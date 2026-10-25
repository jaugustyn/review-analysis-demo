FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# Only the app's dependencies: the results are precomputed, so no PyTorch, models or LLM here.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-default-groups

COPY . .

EXPOSE 8502
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
