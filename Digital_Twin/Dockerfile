# Multi-stage build for optimized production image

# ----------------------
# Builder stage
# ----------------------
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ----------------------
# Production stage
# ----------------------
FROM python:3.12-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=WEB_APPLICATION/enhanced_flask_app_v2.py \
    FLASK_ENV=production \
    PORT=5000

# Install runtime dependencies
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p DATABASE LOGS SECURITY/audit_logs SECURITY/data_backups SECURITY/keys ANALYTICS/models ANALYTICS/analysis_cache REPORTS/generated && \
    chown -R appuser:appuser /app && \
    chmod +x WEB_APPLICATION/enhanced_flask_app_v2.py

# Switch to non-root user
USER appuser

# Add local Python packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "WEB_APPLICATION.enhanced_flask_app_v2:app"]

# Labels
LABEL maintainer="Digital Twin Team" \
      version="2.0" \
      description="Digital Twin System for Industrial IoT" \
      org.opencontainers.image.source="https://github.com/your-org/digital-twin"
