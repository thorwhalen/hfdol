#!/usr/bin/env python3
"""deploy_webapp.py — push a webapp to a HuggingFace Space.

Full cycle: (optional) npm build → stage → upload → factory reboot → wait.

The work is done by `hfdol.deploy`. This script is a CLI wrapper for ad-hoc
terminal use.

USAGE:
    python deploy_webapp.py --space OWNER/NAME --repo /path/to/repo --package PKG
    python deploy_webapp.py --space OWNER/NAME --repo . --package PKG --pin '~=0.2.0'
    python deploy_webapp.py --space OWNER/NAME --repo . --package PKG --skip-build --no-restart

Assumes the standard layout:
    REPO/webapp/api/         FastAPI source (gets COPYed into the image)
    REPO/webapp/ui/          React + Vite (dist/ is what gets shipped)
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

from hfdol.deploy import (
    deploy_webapp,
    render_pypi_webapp_dockerfile,
    render_space_readme,
    stage_webapp,
)


def build_ui(repo_dir: pathlib.Path) -> None:
    ui_dir = repo_dir / "webapp" / "ui"
    if not ui_dir.is_dir():
        sys.exit(f"webapp/ui/ not found in {repo_dir}")
    print(f"Building UI in {ui_dir} ...")
    subprocess.run(["npm", "run", "build"], cwd=ui_dir, check=True)
    print("UI build complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--space", required=True, help='HF Space repo, e.g. "thorwhalen/typola".')
    ap.add_argument("--repo", required=True, help="Local source repo directory.")
    ap.add_argument("--package", required=True, help="PyPI package name to install in the Dockerfile.")
    ap.add_argument("--extras", default="web", help='PyPI extras (default: "web").')
    ap.add_argument("--pin", default="~=0.1.0", help='PyPI version spec (default: "~=0.1.0").')
    ap.add_argument("--title", default=None, help="Space display title (defaults to package name capitalised).")
    ap.add_argument("--staging", default=None, help="Staging dir (default: /tmp/{package}-space).")
    ap.add_argument("--message", default="Update webapp source", help="Commit message.")
    ap.add_argument("--skip-build", action="store_true", help="Don't run npm; reuse existing dist/.")
    ap.add_argument("--no-restart", action="store_true", help="Upload only; don't factory reboot.")
    ap.add_argument("--no-wait", action="store_true", help="Don't wait for the build to settle.")
    ap.add_argument("--timeout", type=int, default=900, help="Seconds to wait for build.")
    ap.add_argument(
        "--cache",
        action="append",
        default=[],
        help="A 'pre-cache' shell command to RUN in the Dockerfile (repeatable).",
    )
    args = ap.parse_args()

    repo_dir = pathlib.Path(args.repo).expanduser().resolve()
    staging_dir = pathlib.Path(args.staging or f"/tmp/{args.package}-space").expanduser()
    title = args.title or args.package.capitalize()

    if not args.skip_build:
        build_ui(repo_dir)

    stage_webapp(
        staging_dir=staging_dir,
        api_src=repo_dir / "webapp" / "api",
        ui_dist_src=repo_dir / "webapp" / "ui" / "dist",
        dockerfile_text=render_pypi_webapp_dockerfile(
            package=args.package,
            extras=args.extras,
            version_spec=args.pin,
            extra_run_lines=args.cache,
        ),
        readme_text=render_space_readme(
            title=title,
            body=f"# {title}\n\n- Source: (your GitHub repo)\n- Package: https://pypi.org/project/{args.package}/",
        ),
    )

    result = deploy_webapp(
        repo_id=args.space,
        source_dir=staging_dir,
        message=args.message,
        rebuild=not args.no_restart,
        wait=not args.no_wait,
        timeout=args.timeout,
    )

    if result is not None and not result.succeeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
