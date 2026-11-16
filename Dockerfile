FROM python:3.11-slim

# The Azure Speech SDK needs OpenSSL and system certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates libasound2 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py prompt.md knowledge.md ./

ENV PYTHONUNBUFFERED=1
EXPOSE 9092
CMD ["python", "bot.py"]
