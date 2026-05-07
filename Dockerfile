# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Working directory inside container ────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app code and model ───────────────────────────────────────────────────
COPY app/ ./app/
COPY model/ ./model/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the server ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]