$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Keep research, authentication, Milvus, chat and first-time on-demand collection,
# while pausing bulk collection and scheduling services that are unnecessary for browsing.
docker compose stop collector worker scheduler
docker compose up -d mysql redis api chat-worker ai-background-worker on-demand-worker frontend
docker compose ps
