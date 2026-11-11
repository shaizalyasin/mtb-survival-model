FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code change does not reinstall the scientific stack.
COPY requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app/src MTB_ARTIFACTS=/app/artifacts

# Nothing is written after startup: the service holds no state and persists no
# patient data. Running unprivileged makes that harder to undo by accident.
RUN useradd --create-home --uid 10001 mtb
USER mtb

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD \
    python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
