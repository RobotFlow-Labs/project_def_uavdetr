FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv pip install -e ".[dev]" --system
COPY . .
CMD ["python", "-m", "anima_inari"]
