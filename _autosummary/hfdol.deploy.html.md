# hfdol.deploy

Deploy webapps to HuggingFace Spaces.

This module wraps `huggingface_hub` with opinionated, idempotent helpers for
publishing a Python package’s webapp to a Docker-SDK Space. The primary entry
point is [`deploy_webapp()`](#hfdol.deploy.deploy_webapp), which composes the smaller helpers into a
full create-or-update + restart cycle.

Functions are designed to be composable: each does one thing, fails loudly,
and prints what it’s about to do. The module assumes you already have a
package on PyPI (or a Dockerfile that knows how to install it).

Quick reference:

```pycon
>>> from hfdol.deploy import deploy_webapp
>>> deploy_webapp(
...     repo_id="thorwhalen/typola",
...     source_dir="/path/to/staging",
... )
```

For ad-hoc operations:

```pycon
>>> from hfdol.deploy import factory_reboot, wait_for_build
>>> factory_reboot("thorwhalen/typola")
>>> wait_for_build("thorwhalen/typola")
```

Token resolution: Operations need a write-scoped HF token. By default,
[`ensure_write_token()`](#hfdol.deploy.ensure_write_token) reads `HF_WRITE_TOKEN` from the environment, then
falls back to sourcing `~/.keys` (a shell file that exports it). If you keep
your write token elsewhere, pass `token=...` to any function explicitly.

### Functions

| [`create_or_update_space`](#hfdol.deploy.create_or_update_space)(repo_id, \*[, sdk, ...])    | Create an HF Space if it doesn't exist; otherwise return its info.              |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`deploy_webapp`](#hfdol.deploy.deploy_webapp)(repo_id, source_dir, \*[, sdk, ...]) | End-to-end: create-if-missing → upload → factory reboot → wait for build.       |
| [`ensure_write_token`](#hfdol.deploy.ensure_write_token)(\*[, env_var, keys_file])       | Return a write-scoped HF token; source `~/.keys` if needed.                     |
| [`factory_reboot`](#hfdol.deploy.factory_reboot)(repo_id, \*[, token])               | Trigger a from-scratch rebuild of the Space (busts Docker layer cache).         |
| [`render_pypi_webapp_dockerfile`](#hfdol.deploy.render_pypi_webapp_dockerfile)(package, \*[, ...])  | Render a Dockerfile for the canonical "PyPI package + FastAPI + React" pattern. |
| [`render_space_readme`](#hfdol.deploy.render_space_readme)(title, \*[, color_from, ...])  | Render an HF Space README with the YAML frontmatter HF requires.                |
| [`stage_webapp`](#hfdol.deploy.stage_webapp)(\*, staging_dir, api_src, ...[, ...]) | Lay out the contents of a Space repo in a local staging directory.              |
| [`upload_app_dir`](#hfdol.deploy.upload_app_dir)(repo_id, folder_path, \*[, ...])    | Upload a local folder to an HF Space, replacing the live files.                 |
| [`wait_for_build`](#hfdol.deploy.wait_for_build)(repo_id, \*[, token, timeout, ...]) | Poll Space status until it lands in a terminal state or times out.              |

### Classes

| [`BuildResult`](#hfdol.deploy.BuildResult)(final_stage, elapsed_seconds[, ...])   | Outcome of waiting for a Space build.   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------|

### *class* hfdol.deploy.BuildResult(final_stage, elapsed_seconds, timed_out=False, stages_seen=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Outcome of waiting for a Space build.

### hfdol.deploy.create_or_update_space(repo_id, , sdk='docker', private=False, token=None, exist_ok=True)

Create an HF Space if it doesn’t exist; otherwise return its info.

Idempotent. Safe to call before every deploy.

* **Parameters:**
  * **repo_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `owner/space-name`.
  * **sdk** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Space SDK — `docker` (recommended for custom apps),
    `gradio`, `streamlit`, `static`.
  * **private** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, create as private (only matters on first creation).
  * **token** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Write token; defaults to [`ensure_write_token()`](#hfdol.deploy.ensure_write_token).
  * **exist_ok** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), an already-existing repo is fine.
* **Returns:**
  `SpaceInfo` for the (now-existing) Space.

### hfdol.deploy.deploy_webapp(repo_id, source_dir, , sdk='docker', private=False, token=None, message='Deploy webapp', ignore_patterns=('node_modules/\*\*', '_\_pycache_\_/\*\*', '\*.pyc', '.DS_Store', '.git/\*\*'), delete_patterns=None, rebuild=True, wait=True, timeout=900)

End-to-end: create-if-missing → upload → factory reboot → wait for build.

The most common deploy entry point. `source_dir` should contain
everything that goes on the Space repo: `Dockerfile`, `README.md`
(with HF YAML frontmatter), and your app source (e.g., `webapp/api/`,
`webapp/ui/dist/`).

* **Parameters:**
  * **repo_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `owner/space-name`.
  * **source_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Local staging directory.
  * **sdk** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Space SDK (only matters on first creation).
  * **private** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Create as private (only matters on first creation).
  * **token** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Write token; defaults to [`ensure_write_token()`](#hfdol.deploy.ensure_write_token).
  * **message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Commit message.
  * **ignore_patterns** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Glob patterns to skip during upload.
  * **delete_patterns** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Glob patterns to delete from the Space before
    uploading (e.g., to clear stale build artifacts).
  * **rebuild** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), trigger a factory reboot after upload.
    Set False to skip — useful when only README/metadata changed.
  * **wait** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), poll until the build settles.
  * **timeout** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max seconds to wait for the build.
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`BuildResult`](#hfdol.deploy.BuildResult)]
* **Returns:**
  [`BuildResult`](#hfdol.deploy.BuildResult) if `wait=True`, else `None`.

### hfdol.deploy.ensure_write_token(, env_var='HF_WRITE_TOKEN', keys_file=PosixPath('/home/runner/.keys'))

Return a write-scoped HF token; source `~/.keys` if needed.

HF Spaces operations (create, upload, restart) need a write token. The
default `HF_TOKEN` is often read-only. Convention: keep the write token
in `~/.keys` as `export HF_WRITE_TOKEN=hf_...`.

* **Parameters:**
  * **env_var** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Environment variable name to read first.
  * **keys_file** ([`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Shell file to source as fallback (must export `env_var`).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The token string.
* **Raises:**
  [**SystemExit**](https://docs.python.org/3/builtins/exceptions.html#SystemExit) – If no token can be resolved.

### hfdol.deploy.factory_reboot(repo_id, , token=None)

Trigger a from-scratch rebuild of the Space (busts Docker layer cache).

A normal restart reuses cached Docker layers. A factory reboot blows the
cache away — required when `pip install` should re-fetch a new package
version, or when system deps in the Dockerfile changed.

* **Parameters:**
  * **repo_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `owner/space-name`.
  * **token** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Write token; defaults to [`ensure_write_token()`](#hfdol.deploy.ensure_write_token).
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### hfdol.deploy.render_pypi_webapp_dockerfile(package, , extras='web', version_spec='', python_version='3.12', api_module='webapp.api.main:app', port=7860, api_dir='webapp/api', ui_dist_dir='webapp/ui/dist', extra_run_lines=())

Render a Dockerfile for the canonical “PyPI package + FastAPI + React” pattern.

Assumes:

- Your Python package is on PyPI (with optional `[extras]`).
- Your FastAPI app is in `api_dir/` (will be COPYed into the image).
- Your React UI is pre-built in `ui_dist_dir/` (will be COPYed in).

Why pre-build the UI locally instead of in the Dockerfile? Because
`package-lock.json` may have `file:` deps to a local monorepo that
doesn’t exist on the build machine. Pre-building sidesteps the issue.

* **Parameters:**
  * **package** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – PyPI package name.
  * **extras** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Extras spec (without brackets), e.g. `"web"` →
    `package[web]`. Pass empty string to omit.
  * **version_spec** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – PEP 440 version specifier, e.g. `"~=0.1.1"`.
    Empty string means “latest”.
  * **python_version** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Python base image tag (`"3.12"` → `python:3.12-slim`).
  * **api_module** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `module.path:app_attr` for uvicorn.
  * **port** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Port to expose (HF Spaces uses 7860 for Docker SDK).
  * **api_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Local path of the FastAPI source, relative to the Space repo root.
  * **ui_dist_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Local path of the pre-built React dist, relative to the Space repo root.
  * **extra_run_lines** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Additional `RUN` lines to insert (e.g., to
    pre-cache datasets so cold start is sub-second).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The Dockerfile contents as a string.

### hfdol.deploy.render_space_readme(title, , color_from='blue', color_to='green', sdk='docker', port=7860, pinned=False, body='')

Render an HF Space README with the YAML frontmatter HF requires.

The frontmatter sets the Space’s metadata (sdk, port, etc.) — required
even for Docker SDK Spaces. Body is appended after.

* **Parameters:**
  * **title** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Display title for the Space.
  * **color_from** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Gradient start color (HF picks from a small palette).
  * **color_to** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Gradient end color.
  * **sdk** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"docker"`, `"gradio"`, `"streamlit"`, `"static"`.
  * **port** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `app_port` (Docker SDK uses this; ignored otherwise).
  * **pinned** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to pin to the user’s HF profile.
  * **body** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Markdown content to append after the frontmatter.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### hfdol.deploy.stage_webapp(, staging_dir, api_src, ui_dist_src, api_dir_in_space='webapp/api', ui_dist_dir_in_space='webapp/ui/dist', dockerfile_text=None, readme_text=None, extra_files=None)

Lay out the contents of a Space repo in a local staging directory.

Idempotent: re-running replaces the API and UI dirs but reuses what’s
already there for things you didn’t pass (e.g., an existing Dockerfile).

* **Parameters:**
  * **staging_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Where to assemble the Space contents.
  * **api_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Local source directory for the FastAPI app.
  * **ui_dist_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Local source directory for the pre-built React dist.
  * **api_dir_in_space** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path inside the Space repo for the API.
  * **ui_dist_dir_in_space** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path inside the Space repo for the UI dist.
  * **dockerfile_text** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Dockerfile contents (use [`render_pypi_webapp_dockerfile()`](#hfdol.deploy.render_pypi_webapp_dockerfile)).
    If None, only writes one if the staging dir doesn’t already have it.
  * **readme_text** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – README contents (use [`render_space_readme()`](#hfdol.deploy.render_space_readme)).
    Same conditional behavior as `dockerfile_text`.
  * **extra_files** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `{relative_path: contents}` extras (e.g., `.gitignore`).
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The staging directory path.

### hfdol.deploy.upload_app_dir(repo_id, folder_path, , message='Update app contents', token=None, ignore_patterns=('node_modules/\*\*', '_\_pycache_\_/\*\*', '\*.pyc', '.DS_Store', '.git/\*\*'), delete_patterns=None)

Upload a local folder to an HF Space, replacing the live files.

Wraps `huggingface_hub.HfApi.upload_folder()` with sensible ignore
patterns and a printed plan. Existing files on the Space that are not in
the local folder are NOT deleted unless you pass `delete_patterns`.

* **Parameters:**
  * **repo_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `owner/space-name`.
  * **folder_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Local directory to upload.
  * **message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Commit message on the Space repo.
  * **token** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Write token; defaults to [`ensure_write_token()`](#hfdol.deploy.ensure_write_token).
  * **ignore_patterns** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Glob patterns to skip.
  * **delete_patterns** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional glob patterns of remote files to delete
    before upload (e.g., `["webapp/ui/dist/**"]` to ensure stale build
    artifacts are removed).
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### hfdol.deploy.wait_for_build(repo_id, , token=None, timeout=900, poll_interval=15, on_stage_change=None)

Poll Space status until it lands in a terminal state or times out.

Terminal states: `RUNNING`, `RUNNING_BUILDING` (success); `BUILD_ERROR`,
`RUNTIME_ERROR` (failure). Times out after `timeout` seconds.

* **Parameters:**
  * **repo_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `owner/space-name`.
  * **token** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Write token (read access also fine); defaults to env.
  * **timeout** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max seconds to wait.
  * **poll_interval** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Seconds between polls.
  * **on_stage_change** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – Callback `(stage, elapsed_seconds) -> None`
    invoked each time the stage transitions. Default prints to stdout.
* **Return type:**
  [`BuildResult`](#hfdol.deploy.BuildResult)
* **Returns:**
  [`BuildResult`](#hfdol.deploy.BuildResult).
