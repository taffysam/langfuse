Write-Host "🚀 Starting AI Observability Stack..." -ForegroundColor Green

# =========================
# Start Docker services
# =========================
Write-Host "Starting Docker services..."
docker compose -f ".\docker-compose.build.yml" up -d

# Give services time to initialize
Write-Host "Waiting for services to initialize..."
Start-Sleep -Seconds 5

# =========================
# Check Ollama
# =========================
$ollama = Get-Process ollama -ErrorAction SilentlyContinue

if ($ollama) {
    Write-Host "Ollama already running." -ForegroundColor Cyan
} else {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama"
    Start-Sleep -Seconds 3
}

# =========================
# Final status
# =========================
Write-Host ""
Write-Host "✅ System is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "👉 Run your chat with:"
Write-Host "python .\test-ollama-langfuse.py"