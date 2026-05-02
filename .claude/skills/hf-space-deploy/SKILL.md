---
name: hf-space-deploy
description: Deploy a Python webapp (FastAPI backend + frontend) to a Hugging Face Space using the Docker SDK. Use when the user wants to "publish to HF", "put this on Hugging Face Spaces", "ship a Space", "host the app on HF", "make a Space for X", or "deploy a Gradio/FastAPI/Streamlit app to Hugging Face." Also triggers on rebuild/refresh requests for an existing Space, factory reboots, swapping out a Space's PyPI version pin, or fixing build failures. The skill assumes the package is on PyPI; if it isn't yet, suggest publishing first (or ship source via the Dockerfile).
---

# Deploy a Webapp to a HuggingFace Space

End-to-end recipe for publishing a Python package's webapp as a **Docker SDK** Hugging Face Space. The skill is opinionated for the `pip-installable package + FastAPI API + React/SPA frontend` pattern, but the helpers cover other shapes too.

## Underlying library: `hfdol.deploy`

This skill is a thin wrapper. The actual work happens in `hfdol.deploy` — composable functions that any Python code (not just this skill's scripts) can call:

```python
from hfdol.deploy import (
    deploy_webapp,           # full workflow: create → upload → reboot → wait
    create_or_update_space,  # idempotent space creation
    upload_app_dir,          # upload a folder, replacing live files
    factory_reboot,          # bust the Docker layer cache
    wait_for_build,          # poll until RUNNING / failure
    ensure_write_token,      # resolve HF_WRITE_TOKEN from env or ~/.keys
    render_pypi_webapp_dockerfile,  # generate a Dockerfile
    render_space_readme,     # generate the YAML-frontmatter README
    stage_webapp,            # lay out a staging dir for upload
)
```

If your task fits one of these primitives, prefer them over the scripts. The scripts here are CLI thin-wrappers for ad-hoc terminal use.

## Decision tree

| You want to... | Action |
|---|---|
| Initial deploy of an app whose package is already on PyPI | "Initial deploy" below |
| Refresh the Space to pull a newly-published patch version | `scripts/rebuild_space.py` (factory reboot) |
| Push updated frontend / API source after a code change | `scripts/deploy_webapp.py` |
| Bump the PyPI version pin in the Dockerfile (e.g., 0.1.x → 0.2.x) | "Bumping the version pin" below |
| Deploy a Gradio or Streamlit app instead | "Other SDKs" below |
| Diagnose a stuck/failed build | "Diagnosing failures" below |

## Initial deploy

The standard "package is on PyPI, has a webapp" recipe. Five steps:

1. **Verify package is on PyPI**: `curl -s https://pypi.org/pypi/PKG/json | jq .info.version`. If not, publish first.

2. **Build the UI locally** (don't try to build inside Docker — see Gotcha 1):
   ```bash
   cd /path/to/repo/webapp/ui && npm run build
   ```

3. **Stage the Space contents** in a working directory:
   ```python
   from hfdol.deploy import stage_webapp, render_pypi_webapp_dockerfile, render_space_readme

   stage_webapp(
       staging_dir="/tmp/PKG-space",
       api_src="/path/to/repo/webapp/api",
       ui_dist_src="/path/to/repo/webapp/ui/dist",
       dockerfile_text=render_pypi_webapp_dockerfile(
           package="PKG",
           extras="web",          # → PKG[web]
           version_spec="~=0.1.1", # compatible-release
           extra_run_lines=[
               # Pre-cache datasets so cold start is sub-second.
               "python -c \"from PKG import load; load('something')\"",
           ],
       ),
       readme_text=render_space_readme(
           title="MyApp",
           body="Description here.\n\n- Source: https://github.com/owner/PKG\n- Package: https://pypi.org/project/PKG/",
       ),
   )
   ```

4. **Deploy** (creates if missing, uploads, reboots, waits):
   ```python
   from hfdol.deploy import deploy_webapp
   result = deploy_webapp(
       repo_id="OWNER/PKG",
       source_dir="/tmp/PKG-space",
   )
   # result.succeeded → True if Space is RUNNING
   ```

5. **Verify**: `curl -s https://OWNER-PKG.hf.space/api/...` (replace with your endpoint).

For ad-hoc terminal use, the same flow is in `scripts/deploy_webapp.py`.

## Bumping the version pin

The Dockerfile pins the package with a compatible-release spec like `~=0.1.1` (allows `0.1.x` patches, blocks `0.2.0`). To move to a new minor:

1. Re-stage with a new `version_spec`:
   ```python
   stage_webapp(
       staging_dir="/tmp/PKG-space",
       ...,
       dockerfile_text=render_pypi_webapp_dockerfile(package="PKG", version_spec="~=0.2.0"),
   )
   ```
2. Re-deploy (`deploy_webapp(...)`). The factory reboot is essential — without it, Docker's layer cache will skip re-running `pip install` and you'll keep the old version.

**Don't pin too loosely** (e.g. `>=0.1.0`): caching means `pip install` doesn't auto-rerun on new PyPI versions, and the bundled webapp source might encode an old API contract. Pin to a compatible range and bump deliberately.

## Other SDKs

The defaults above target Docker SDK + FastAPI + React. For other shapes:

| Stack | Adjustments |
|---|---|
| **Gradio app** | Pass `sdk="gradio"` to `create_or_update_space()`. Skip the Dockerfile entirely; HF auto-builds Gradio Spaces from `app.py`. Push your Gradio code with `upload_app_dir()`. |
| **Streamlit** | `sdk="streamlit"`, similar — push your `streamlit_app.py`. |
| **Static site** | `sdk="static"`. Push the build output (HTML/CSS/JS) with `upload_app_dir()`. No Dockerfile, no port. |
| **Custom Docker** (different Python version, system deps, etc.) | Write your own Dockerfile and pass `dockerfile_text=open("my.Dockerfile").read()` to `stage_webapp()`. Or skip `stage_webapp` and assemble the staging dir manually. |

See `references/dockerfile_templates.md` for more Dockerfile patterns.

## Diagnosing failures

When a deploy fails:

1. **Check build logs**: `https://huggingface.co/spaces/OWNER/SPACE/logs` (visible in browser, also via `huggingface_hub.HfApi.get_space_runtime`).

2. **Common build errors**:
   - **`pip install` fails** → version doesn't exist on PyPI yet (publish first), or your `version_spec` is too restrictive
   - **`COPY failed: file not found`** → the staging dir is missing the file, or its path differs from what the Dockerfile expects
   - **`npm ci` errors** → you tried to build the UI inside Docker; switch to pre-building locally (see Gotcha 1)

3. **Common runtime errors**:
   - **Container exits immediately** → check `CMD` line, port mismatch (HF Spaces Docker SDK expects 7860 by default), missing env vars
   - **502/connection refused** → app is binding to `127.0.0.1` instead of `0.0.0.0`; also check `app_port` in README frontmatter matches `EXPOSE`

4. **Cache-related**: If a fresh `pip install` should pick up a new version but isn't, the Docker layer is cached. Run `factory_reboot()` (NOT a normal restart) to bust the cache.

## Gotchas

These are real failures we've hit, not theoretical. Reading them first will save you 15+ minutes each.

### 1. Don't `npm ci` inside Docker if your lockfile has `file:` deps

If `package-lock.json` references local-path deps (e.g., `"@org/pkg": "file:../../monorepo/pkg"`), those paths don't exist on the HF Space build machine. `npm ci` fails with a confusing "lockfileVersion" error that's actually about unresolvable `file:` deps.

**Solution**: build `dist/` locally and ship it pre-built. The default Dockerfile from `render_pypi_webapp_dockerfile()` does this — it `COPY`s `webapp/ui/dist/` and never runs `npm`.

### 2. HF_TOKEN is often read-only

The default HF token (e.g., `HF_TOKEN`) may be scoped to "download data only" and lacks write permissions. Space ops (`create_repo`, `upload_folder`, `restart_space`) need a write token.

**Convention**: keep a separate write token at `HF_WRITE_TOKEN` in `~/.keys`:
```bash
# ~/.keys
export HF_WRITE_TOKEN=hf_...
```
The `ensure_write_token()` function reads `HF_WRITE_TOKEN` from env first, then sources `~/.keys` as fallback. To create a write token: HF settings → Access Tokens → "New token" → role "write".

### 3. Factory reboot vs restart

`api.restart_space(repo_id)` reuses cached Docker layers — fast, but won't re-fetch a new PyPI version. `api.restart_space(repo_id, factory_reboot=True)` blows the cache → forces a full rebuild. Use `factory_reboot=True` whenever the rebuild needs to actually re-run `pip install`. The `factory_reboot()` helper in `hfdol.deploy` always passes `factory_reboot=True`.

### 4. README YAML frontmatter is required

Even Docker SDK Spaces need a YAML-frontmatter README with `sdk:`, `app_port:`, etc. Without it, HF's UI shows "Configuration error" and the Space won't build. `render_space_readme()` produces a valid one.

### 5. Frontend ↔ Backend version coupling

If your webapp ships **API source** (e.g., `webapp/api/main.py`) AND **prebuilt UI** (`webapp/ui/dist/`) both COPYed into the image, they're frozen at whatever you bundled. The `pip install`d package (e.g., `typola[web]`) might be a different version of the underlying library, but the routes the UI calls are baked in. If you bump the version pin and the API contract changed, you may break the UI.

**Implication**: when bumping a major/minor version, also re-stage with fresh `webapp/api/` and `webapp/ui/dist/` from the version's source tree, not the old ones.

### 6. `webapp/` is NOT in the wheel (typically)

If you control the package, the PyPI wheel ships only the importable Python package — not `webapp/api/` or `webapp/ui/`. Those live in your repo and have to be pushed to the Space repo separately (which is exactly what `deploy_webapp` does). If you change webapp code, you have to push it to **both** the source repo (commit + git push) AND the Space (`deploy_webapp(...)`).

## Scripts

| Script | Purpose |
|---|---|
| `scripts/deploy_webapp.py` | Full cycle: build UI → stage → upload → factory reboot → wait. The everyday command. |
| `scripts/rebuild_space.py` | Factory reboot only (no upload). Use after publishing a new patch version of the package. |

Each script prints what it's about to do and exits non-zero on failure.

## Where things live (after a successful deploy)

| Resource | Pattern |
|---|---|
| Space repo | `https://huggingface.co/spaces/OWNER/NAME` |
| Live URL | `https://OWNER-NAME.hf.space/` |
| Build logs | `https://huggingface.co/spaces/OWNER/NAME/logs` |
| Settings (secrets, hardware, visibility) | `https://huggingface.co/spaces/OWNER/NAME/settings` |
