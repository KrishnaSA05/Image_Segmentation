# ============================================================
#  Drivable Area Detection — Production Dockerfile
#  Phase 2: multi-stage build, non-root user, health check
# ============================================================

# ── Stage 1: dependency builder ──────────────────────────────────────────────
# Installs all Python packages into a clean prefix so the final image
# only copies compiled wheels — no pip cache, no build tools.
FROM python:3.10-slim AS builder

WORKDIR /install

# System libs needed to compile some wheels (opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# OpenCV runtime system libraries (no build tools needed here)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install/pkg /usr/local

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# ── Copy application source ───────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Streamlit configuration ───────────────────────────────────────────────────
# Disable telemetry and file-watcher for container environments
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
# Docker / ECS marks the container unhealthy if Streamlit stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
