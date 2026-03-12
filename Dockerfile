FROM python:3.12-slim

# Set UTC timezone for consistent candle timestamps
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user for security and own the app and logs directories
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && mkdir -p /app/logs && chown -R appuser:appgroup /app

USER appuser

# No ports exposed — V2 uses outbound-only connections (Telegram polling + Binance WS/REST)

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD ["python", "healthcheck.py"]

CMD ["python", "-m", "src.main"]
