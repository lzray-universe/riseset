FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends bash curl wget \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DE_BSP=/data/spk

RUN chmod +x scripts/boot.sh

CMD ["/app/scripts/boot.sh"]
