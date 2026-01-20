FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
RUN python - <<'PY'
import pathlib
import tomllib

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
deps = data.get("project", {}).get("dependencies", [])
pathlib.Path("/tmp/requirements.txt").write_text("\n".join(deps))
PY
RUN pip install -r /tmp/requirements.txt

COPY src /app/src
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "sp_solution.transport.api:app", "--host", "0.0.0.0", "--port", "8000"]
