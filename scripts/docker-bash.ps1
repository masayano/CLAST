#Requires -Version 5.1
# Git bash (優先) または WSL で docker-*.sh を実行。CLAST_POWERSHELL_BASH_ORDER=wsl-first で WSL 優先。
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [ValidateSet('build', 'test', 'smoke')]
  [string] $Action
)
$ErrorActionPreference = 'Stop'
$sh = switch ($Action) {
  'build' { 'docker-build.sh' }
  'test'  { 'docker-test.sh' }
  'smoke' { 'docker-smoke.sh' }
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path (Join-Path $PSScriptRoot $sh))) { throw "Missing: scripts\$sh" }
$gitBash = @(
  "${env:ProgramFiles}\Git\usr\bin\bash.exe",
  "${env:ProgramFiles}\Git\bin\bash.exe",
  "$env:LocalAppData\Programs\Git\usr\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$wslFirst = $env:CLAST_POWERSHELL_BASH_ORDER -eq "wsl-first"
Push-Location $repo
$rel = "./scripts/" + $sh
function Invoke-WithBash {
  param([string]$BashPath)
  & $BashPath -l $rel
  if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
function Invoke-WslBash {
  wsl.exe bash -l $rel
  if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
try {
  if ($wslFirst) {
    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) { Invoke-WslBash; return }
    if ($gitBash) { Invoke-WithBash $gitBash; return }
  } else {
    if ($gitBash) { Invoke-WithBash $gitBash; return }
    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) { Invoke-WslBash; return }
  }
  throw "Install Git for Windows (https://git-scm.com) or WSL, or run: wsl.exe bash $rel"
} finally { Pop-Location }
