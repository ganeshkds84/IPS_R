# Bootstrap a development environment (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

uv python install 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example"
}

if (Test-Path .git) {
    uv run pre-commit install
} else {
    Write-Host "Not a git repository; skipping pre-commit install (run 'git init' first)."
}

uv run python scripts/verify_setup.py
exit $LASTEXITCODE
