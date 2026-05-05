Write-Host "🛑 Stopping AI Observability Stack..." -ForegroundColor Red

# Stop Docker
Write-Host "Stopping Docker services..."
docker compose -f .\docker-compose.build.yml  down

# Stop Ollama
Write-Host "Stopping Ollama..."
taskkill /IM ollama.exe /F 2>$null

Write-Host "✅ System stopped."