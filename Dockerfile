FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . .

RUN uv sync --locked


EXPOSE 8000

HEALTHCHECK --interval=5m --timeout=10s --start-period=10s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "main.py"]
