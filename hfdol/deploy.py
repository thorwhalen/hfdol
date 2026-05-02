"""
Deploy webapps to HuggingFace Spaces.

This module wraps `huggingface_hub` with opinionated, idempotent helpers for
publishing a Python package's webapp to a Docker-SDK Space. The primary entry
point is :func:`deploy_webapp`, which composes the smaller helpers into a
full create-or-update + restart cycle.

Functions are designed to be composable: each does one thing, fails loudly,
and prints what it's about to do. The module assumes you already have a
package on PyPI (or a Dockerfile that knows how to install it).

Quick reference:

    >>> from hfdol.deploy import deploy_webapp  # doctest: +SKIP
    >>> deploy_webapp(                          # doctest: +SKIP
    ...     repo_id="thorwhalen/typola",
    ...     source_dir="/path/to/staging",
    ... )

For ad-hoc operations:

    >>> from hfdol.deploy import factory_reboot, wait_for_build  # doctest: +SKIP
    >>> factory_reboot("thorwhalen/typola")     # doctest: +SKIP
    >>> wait_for_build("thorwhalen/typola")     # doctest: +SKIP

Token resolution: Operations need a write-scoped HF token. By default,
:func:`ensure_write_token` reads ``HF_WRITE_TOKEN`` from the environment, then
falls back to sourcing ``~/.keys`` (a shell file that exports it). If you keep
your write token elsewhere, pass ``token=...`` to any function explicitly.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


DFLT_KEYS_FILE = pathlib.Path.home() / ".keys"
DFLT_TOKEN_VAR = "HF_WRITE_TOKEN"
DFLT_BUILD_TIMEOUT = 900  # 15 minutes — typical Docker build is 2-5 min
DFLT_POLL_INTERVAL = 15  # seconds between space_info polls
DFLT_DOCKER_PORT = 7860  # HF Spaces convention for Docker SDK
DFLT_IGNORE_PATTERNS = (
    "node_modules/**",
    "__pycache__/**",
    "*.pyc",
    ".DS_Store",
    ".git/**",
)


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def ensure_write_token(
    *,
    env_var: str = DFLT_TOKEN_VAR,
    keys_file: pathlib.Path = DFLT_KEYS_FILE,
) -> str:
    """Return a write-scoped HF token; source ``~/.keys`` if needed.

    HF Spaces operations (create, upload, restart) need a write token. The
    default ``HF_TOKEN`` is often read-only. Convention: keep the write token
    in ``~/.keys`` as ``export HF_WRITE_TOKEN=hf_...``.

    :param env_var: Environment variable name to read first.
    :param keys_file: Shell file to source as fallback (must export ``env_var``).
    :returns: The token string.
    :raises SystemExit: If no token can be resolved.
    """
    tok = os.environ.get(env_var)
    if tok:
        return tok
    if not keys_file.is_file():
        raise SystemExit(
            f"{env_var} not set and {keys_file} does not exist. "
            f"Either export {env_var}=... or create {keys_file} with that line."
        )
    out = subprocess.run(
        ["bash", "-c", f"source {keys_file} && printf '%s' \"${env_var}\""],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not out:
        raise SystemExit(
            f"{keys_file} does not export {env_var}. "
            f"Add a line like: export {env_var}=hf_..."
        )
    os.environ[env_var] = out
    return out


def _resolve_token(token: Optional[str]) -> str:
    """Resolve token argument: explicit value wins, else fall back to env/keys."""
    return token if token is not None else ensure_write_token()


# ---------------------------------------------------------------------------
# Space lifecycle
# ---------------------------------------------------------------------------


def create_or_update_space(
    repo_id: str,
    *,
    sdk: str = "docker",
    private: bool = False,
    token: Optional[str] = None,
    exist_ok: bool = True,
):
    """Create an HF Space if it doesn't exist; otherwise return its info.

    Idempotent. Safe to call before every deploy.

    :param repo_id: ``owner/space-name``.
    :param sdk: Space SDK — ``docker`` (recommended for custom apps),
        ``gradio``, ``streamlit``, ``static``.
    :param private: If True, create as private (only matters on first creation).
    :param token: Write token; defaults to :func:`ensure_write_token`.
    :param exist_ok: If True (default), an already-existing repo is fine.
    :returns: ``SpaceInfo`` for the (now-existing) Space.
    """
    from huggingface_hub import HfApi, create_repo

    tok = _resolve_token(token)
    api = HfApi(token=tok)

    print(f"Ensuring Space {repo_id} exists (sdk={sdk}) ...")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk=sdk,
        private=private,
        token=tok,
        exist_ok=exist_ok,
    )
    return api.space_info(repo_id)


def upload_app_dir(
    repo_id: str,
    folder_path: str | pathlib.Path,
    *,
    message: str = "Update app contents",
    token: Optional[str] = None,
    ignore_patterns: Iterable[str] = DFLT_IGNORE_PATTERNS,
    delete_patterns: Optional[Iterable[str]] = None,
) -> None:
    """Upload a local folder to an HF Space, replacing the live files.

    Wraps :func:`huggingface_hub.HfApi.upload_folder` with sensible ignore
    patterns and a printed plan. Existing files on the Space that are not in
    the local folder are NOT deleted unless you pass ``delete_patterns``.

    :param repo_id: ``owner/space-name``.
    :param folder_path: Local directory to upload.
    :param message: Commit message on the Space repo.
    :param token: Write token; defaults to :func:`ensure_write_token`.
    :param ignore_patterns: Glob patterns to skip.
    :param delete_patterns: Optional glob patterns of remote files to delete
        before upload (e.g., ``["webapp/ui/dist/**"]`` to ensure stale build
        artifacts are removed).
    """
    from huggingface_hub import HfApi

    folder_path = pathlib.Path(folder_path)
    if not folder_path.is_dir():
        raise SystemExit(f"Source folder does not exist: {folder_path}")

    api = HfApi(token=_resolve_token(token))
    print(f"Uploading {folder_path} → space {repo_id} ...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="space",
        commit_message=message,
        ignore_patterns=list(ignore_patterns),
        delete_patterns=list(delete_patterns) if delete_patterns else None,
    )
    print("Upload complete.")


def factory_reboot(repo_id: str, *, token: Optional[str] = None) -> None:
    """Trigger a from-scratch rebuild of the Space (busts Docker layer cache).

    A normal restart reuses cached Docker layers. A factory reboot blows the
    cache away — required when ``pip install`` should re-fetch a new package
    version, or when system deps in the Dockerfile changed.

    :param repo_id: ``owner/space-name``.
    :param token: Write token; defaults to :func:`ensure_write_token`.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=_resolve_token(token))
    print(f"Triggering factory reboot of {repo_id} ...")
    api.restart_space(repo_id, factory_reboot=True)
    print(f"Reboot signaled. Logs: https://huggingface.co/spaces/{repo_id}/logs")


@dataclass
class BuildResult:
    """Outcome of waiting for a Space build."""

    final_stage: str
    elapsed_seconds: int
    timed_out: bool = False
    stages_seen: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.final_stage in ("RUNNING", "RUNNING_BUILDING")


def wait_for_build(
    repo_id: str,
    *,
    token: Optional[str] = None,
    timeout: int = DFLT_BUILD_TIMEOUT,
    poll_interval: int = DFLT_POLL_INTERVAL,
    on_stage_change: Optional[Callable[[str, int], None]] = None,
) -> BuildResult:
    """Poll Space status until it lands in a terminal state or times out.

    Terminal states: ``RUNNING``, ``RUNNING_BUILDING`` (success); ``BUILD_ERROR``,
    ``RUNTIME_ERROR`` (failure). Times out after ``timeout`` seconds.

    :param repo_id: ``owner/space-name``.
    :param token: Write token (read access also fine); defaults to env.
    :param timeout: Max seconds to wait.
    :param poll_interval: Seconds between polls.
    :param on_stage_change: Callback ``(stage, elapsed_seconds) -> None``
        invoked each time the stage transitions. Default prints to stdout.
    :returns: :class:`BuildResult`.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=_resolve_token(token))

    if on_stage_change is None:
        on_stage_change = lambda stage, elapsed: print(f"  t={elapsed}s stage={stage}")

    deadline = time.time() + timeout
    last = None
    seen: list[str] = []
    started = time.time()

    while time.time() < deadline:
        info = api.space_info(repo_id)
        stage = info.runtime.stage
        if stage != last:
            elapsed = int(time.time() - started)
            on_stage_change(stage, elapsed)
            seen.append(stage)
            last = stage
        if stage in ("RUNNING", "RUNNING_BUILDING"):
            return BuildResult(stage, int(time.time() - started), False, seen)
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR"):
            return BuildResult(stage, int(time.time() - started), False, seen)
        time.sleep(poll_interval)

    return BuildResult(last or "UNKNOWN", int(time.time() - started), True, seen)


# ---------------------------------------------------------------------------
# Composed workflow
# ---------------------------------------------------------------------------


def deploy_webapp(
    repo_id: str,
    source_dir: str | pathlib.Path,
    *,
    sdk: str = "docker",
    private: bool = False,
    token: Optional[str] = None,
    message: str = "Deploy webapp",
    ignore_patterns: Iterable[str] = DFLT_IGNORE_PATTERNS,
    delete_patterns: Optional[Iterable[str]] = None,
    rebuild: bool = True,
    wait: bool = True,
    timeout: int = DFLT_BUILD_TIMEOUT,
) -> Optional[BuildResult]:
    """End-to-end: create-if-missing → upload → factory reboot → wait for build.

    The most common deploy entry point. ``source_dir`` should contain
    everything that goes on the Space repo: ``Dockerfile``, ``README.md``
    (with HF YAML frontmatter), and your app source (e.g., ``webapp/api/``,
    ``webapp/ui/dist/``).

    :param repo_id: ``owner/space-name``.
    :param source_dir: Local staging directory.
    :param sdk: Space SDK (only matters on first creation).
    :param private: Create as private (only matters on first creation).
    :param token: Write token; defaults to :func:`ensure_write_token`.
    :param message: Commit message.
    :param ignore_patterns: Glob patterns to skip during upload.
    :param delete_patterns: Glob patterns to delete from the Space before
        uploading (e.g., to clear stale build artifacts).
    :param rebuild: If True (default), trigger a factory reboot after upload.
        Set False to skip — useful when only README/metadata changed.
    :param wait: If True (default), poll until the build settles.
    :param timeout: Max seconds to wait for the build.
    :returns: :class:`BuildResult` if ``wait=True``, else ``None``.
    """
    tok = _resolve_token(token)
    create_or_update_space(repo_id, sdk=sdk, private=private, token=tok)
    upload_app_dir(
        repo_id,
        source_dir,
        message=message,
        token=tok,
        ignore_patterns=ignore_patterns,
        delete_patterns=delete_patterns,
    )
    if not rebuild:
        return None
    factory_reboot(repo_id, token=tok)
    if not wait:
        return None
    result = wait_for_build(repo_id, token=tok, timeout=timeout)
    if result.succeeded:
        space_url = f"https://{repo_id.replace('/', '-')}.hf.space/"
        print(f"\nSpace is live: {space_url}")
    elif result.timed_out:
        print(f"\nTimeout after {result.elapsed_seconds}s (still {result.final_stage}).")
    else:
        print(f"\nBuild failed: stage={result.final_stage}")
        print(f"Logs: https://huggingface.co/spaces/{repo_id}/logs")
    return result


# ---------------------------------------------------------------------------
# Dockerfile templates
# ---------------------------------------------------------------------------


def render_pypi_webapp_dockerfile(
    package: str,
    *,
    extras: str = "web",
    version_spec: str = "",
    python_version: str = "3.12",
    api_module: str = "webapp.api.main:app",
    port: int = DFLT_DOCKER_PORT,
    api_dir: str = "webapp/api",
    ui_dist_dir: str = "webapp/ui/dist",
    extra_run_lines: Iterable[str] = (),
) -> str:
    """Render a Dockerfile for the canonical "PyPI package + FastAPI + React" pattern.

    Assumes:
    - Your Python package is on PyPI (with optional ``[extras]``).
    - Your FastAPI app is in ``api_dir/`` (will be COPYed into the image).
    - Your React UI is pre-built in ``ui_dist_dir/`` (will be COPYed in).

    Why pre-build the UI locally instead of in the Dockerfile? Because
    ``package-lock.json`` may have ``file:`` deps to a local monorepo that
    doesn't exist on the build machine. Pre-building sidesteps the issue.

    :param package: PyPI package name.
    :param extras: Extras spec (without brackets), e.g. ``"web"`` →
        ``package[web]``. Pass empty string to omit.
    :param version_spec: PEP 440 version specifier, e.g. ``"~=0.1.1"``.
        Empty string means "latest".
    :param python_version: Python base image tag (``"3.12"`` → ``python:3.12-slim``).
    :param api_module: ``module.path:app_attr`` for uvicorn.
    :param port: Port to expose (HF Spaces uses 7860 for Docker SDK).
    :param api_dir: Local path of the FastAPI source, relative to the Space repo root.
    :param ui_dist_dir: Local path of the pre-built React dist, relative to the Space repo root.
    :param extra_run_lines: Additional ``RUN`` lines to insert (e.g., to
        pre-cache datasets so cold start is sub-second).
    :returns: The Dockerfile contents as a string.
    """
    extras_spec = f"[{extras}]" if extras else ""
    pip_target = f"{package}{extras_spec}{version_spec}"
    extras_block = "\n".join(f"RUN {line}" for line in extra_run_lines)
    extras_block = f"\n{extras_block}\n" if extras_block else ""

    return (
        f"FROM python:{python_version}-slim\n"
        f"WORKDIR /app\n"
        f"\n"
        f"# Install {package} from PyPI.\n"
        f'RUN pip install --no-cache-dir "{pip_target}"\n'
        f"\n"
        f"# FastAPI app source + prebuilt React UI.\n"
        f"COPY {api_dir}/ /app/{api_dir}/\n"
        f"COPY {ui_dist_dir}/ /app/{ui_dist_dir}/\n"
        f"{extras_block}"
        f"\n"
        f"EXPOSE {port}\n"
        f'CMD ["uvicorn", "{api_module}", "--host", "0.0.0.0", "--port", "{port}"]\n'
    )


def render_space_readme(
    title: str,
    *,
    color_from: str = "blue",
    color_to: str = "green",
    sdk: str = "docker",
    port: int = DFLT_DOCKER_PORT,
    pinned: bool = False,
    body: str = "",
) -> str:
    """Render an HF Space README with the YAML frontmatter HF requires.

    The frontmatter sets the Space's metadata (sdk, port, etc.) — required
    even for Docker SDK Spaces. Body is appended after.

    :param title: Display title for the Space.
    :param color_from: Gradient start color (HF picks from a small palette).
    :param color_to: Gradient end color.
    :param sdk: ``"docker"``, ``"gradio"``, ``"streamlit"``, ``"static"``.
    :param port: ``app_port`` (Docker SDK uses this; ignored otherwise).
    :param pinned: Whether to pin to the user's HF profile.
    :param body: Markdown content to append after the frontmatter.
    """
    return (
        f"---\n"
        f"title: {title}\n"
        f"colorFrom: {color_from}\n"
        f"colorTo: {color_to}\n"
        f"sdk: {sdk}\n"
        f"app_port: {port}\n"
        f"pinned: {'true' if pinned else 'false'}\n"
        f"---\n"
        f"\n"
        f"{body}".rstrip() + "\n"
    )


# ---------------------------------------------------------------------------
# Convenience: stage a webapp directory
# ---------------------------------------------------------------------------


def stage_webapp(
    *,
    staging_dir: str | pathlib.Path,
    api_src: str | pathlib.Path,
    ui_dist_src: str | pathlib.Path,
    api_dir_in_space: str = "webapp/api",
    ui_dist_dir_in_space: str = "webapp/ui/dist",
    dockerfile_text: Optional[str] = None,
    readme_text: Optional[str] = None,
    extra_files: Optional[dict[str, str]] = None,
) -> pathlib.Path:
    """Lay out the contents of a Space repo in a local staging directory.

    Idempotent: re-running replaces the API and UI dirs but reuses what's
    already there for things you didn't pass (e.g., an existing Dockerfile).

    :param staging_dir: Where to assemble the Space contents.
    :param api_src: Local source directory for the FastAPI app.
    :param ui_dist_src: Local source directory for the pre-built React dist.
    :param api_dir_in_space: Path inside the Space repo for the API.
    :param ui_dist_dir_in_space: Path inside the Space repo for the UI dist.
    :param dockerfile_text: Dockerfile contents (use :func:`render_pypi_webapp_dockerfile`).
        If None, only writes one if the staging dir doesn't already have it.
    :param readme_text: README contents (use :func:`render_space_readme`).
        Same conditional behavior as ``dockerfile_text``.
    :param extra_files: ``{relative_path: contents}`` extras (e.g., ``.gitignore``).
    :returns: The staging directory path.
    """
    import shutil

    staging_dir = pathlib.Path(staging_dir)
    api_src = pathlib.Path(api_src)
    ui_dist_src = pathlib.Path(ui_dist_src)

    if not api_src.is_dir():
        raise SystemExit(f"Missing API source: {api_src}")
    if not ui_dist_src.is_dir():
        raise SystemExit(f"Missing UI dist (build first): {ui_dist_src}")

    staging_dir.mkdir(parents=True, exist_ok=True)

    api_dst = staging_dir / api_dir_in_space
    if api_dst.exists():
        shutil.rmtree(api_dst)
    api_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(api_src, api_dst)

    ui_dst = staging_dir / ui_dist_dir_in_space
    if ui_dst.exists():
        shutil.rmtree(ui_dst)
    ui_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ui_dist_src, ui_dst)

    dockerfile = staging_dir / "Dockerfile"
    if dockerfile_text is not None:
        dockerfile.write_text(dockerfile_text)
    elif not dockerfile.exists():
        raise SystemExit(
            f"No Dockerfile in {staging_dir} and none provided. "
            f"Pass dockerfile_text=render_pypi_webapp_dockerfile(...)."
        )

    readme = staging_dir / "README.md"
    if readme_text is not None:
        readme.write_text(readme_text)
    elif not readme.exists():
        raise SystemExit(
            f"No README.md in {staging_dir} and none provided. "
            f"Pass readme_text=render_space_readme(...)."
        )

    if extra_files:
        for relpath, contents in extra_files.items():
            target = staging_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)

    return staging_dir
