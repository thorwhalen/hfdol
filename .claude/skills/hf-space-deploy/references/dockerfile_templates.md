# Dockerfile templates for HF Space deploys

The default `render_pypi_webapp_dockerfile()` covers the FastAPI + React case. This file collects alternatives for other shapes.

## Default: PyPI package + FastAPI + prebuilt React UI

What `render_pypi_webapp_dockerfile()` produces. Use this when:
- Your Python package is already on PyPI (with optional `[web]` extras for the API deps).
- You have a FastAPI app at `webapp/api/`.
- You have a Vite/React UI you build locally and ship as `webapp/ui/dist/`.

```dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir "PKG[web]~=0.1.1"

COPY webapp/api/ /app/webapp/api/
COPY webapp/ui/dist/ /app/webapp/ui/dist/

# Optional: pre-cache datasets so cold start is fast.
RUN python -c "from PKG import load; load('something')"

EXPOSE 7860
CMD ["uvicorn", "webapp.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Source-only (no PyPI publish yet)

When the package isn't on PyPI yet, install it from the local sources you ship into the image:

```dockerfile
FROM python:3.12-slim
WORKDIR /app

# Ship the package source + install it.
COPY pkg/ /app/pkg/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Then the usual webapp bits.
COPY webapp/api/ /app/webapp/api/
COPY webapp/ui/dist/ /app/webapp/ui/dist/

EXPOSE 7860
CMD ["uvicorn", "webapp.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Trade-off: every code change requires a full rebuild + reupload. Move to PyPI as soon as the package stabilizes.

## Multi-stage with Node build inside Docker (only when lockfile is clean)

If your `package-lock.json` has NO `file:` deps (no monorepo imports), you CAN build the UI inside Docker — saves you remembering to `npm run build` locally:

```dockerfile
# Stage 1: build the UI.
FROM node:20-alpine AS ui-build
WORKDIR /ui
COPY webapp/ui/package.json webapp/ui/package-lock.json ./
RUN npm ci
COPY webapp/ui/ ./
RUN npm run build

# Stage 2: the runtime image.
FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir "PKG[web]~=0.1.1"

COPY webapp/api/ /app/webapp/api/
COPY --from=ui-build /ui/dist /app/webapp/ui/dist

EXPOSE 7860
CMD ["uvicorn", "webapp.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Don't use this if `package-lock.json` has any `file:` paths** — the build will fail opaquely.

## System dependencies (apt)

If your package needs `ffmpeg`, `libpq`, `git`, etc.:

```dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "PKG[web]~=0.1.1"
# ... rest as before
```

## Larger image with build tools (when wheels aren't available)

Some Python packages have C extensions and don't ship wheels for slim Python. If `pip install` fails with compile errors:

```dockerfile
FROM python:3.12   # NOT slim — has gcc + headers

# Or, on slim, install build deps then remove them (smaller final image but slower build):
# RUN apt-get update && apt-get install -y gcc python3-dev && \
#     pip install ... && \
#     apt-get purge -y gcc python3-dev && apt-get autoremove -y
```

## Gradio (no Dockerfile needed)

Gradio Spaces don't use Docker. Just:
- `sdk: gradio` in the README frontmatter
- `app.py` with a `gr.Blocks` or `gr.Interface` instance named `demo` (or whatever HF expects)
- `requirements.txt` for deps

Use `create_or_update_space(repo_id, sdk="gradio")` and `upload_app_dir()` to push the files.

## Streamlit (no Dockerfile needed)

Same as Gradio:
- `sdk: streamlit`
- `app.py` (or whatever filename)
- `requirements.txt`

## Static (no Dockerfile, no Python)

For pre-rendered HTML/CSS/JS:
- `sdk: static`
- Push the build output to the Space root

Used for landing pages or fully-static SPAs that don't need a backend.

## Picking a Python version

| HF Spaces hardware | Recommended Python |
|---|---|
| CPU basic | 3.12-slim (default) |
| CPU upgrade / ZeroGPU | 3.12 |
| GPU (T4 etc.) | 3.10-3.12 — match your CUDA-aware deps |

Avoid 3.13 until the wider ecosystem catches up. Avoid <3.10 unless a dep forces it.

## Port + EXPOSE

HF Spaces Docker SDK expects port **7860** by default. You can change it via `app_port:` in the README frontmatter, but stick with 7860 unless you have a reason. Make sure:
- `EXPOSE` matches `app_port`
- The `CMD` binds to `0.0.0.0` (not `127.0.0.1`) on that port
