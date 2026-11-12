# Commit all changes and push to GitHub.
# Usage:  .\save.ps1 "what you changed"
param([string]$Message)

Set-Location $PSScriptRoot

if (-not (git status --porcelain)) {
    Write-Host "Nothing to save."
    exit 0
}

if (-not $Message) {
    $files = (git status --porcelain | Measure-Object).Count
    $Message = "Update $files file(s)"
}

git add -A
git commit -m $Message
git push
