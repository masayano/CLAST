#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'docker-bash.ps1') -Action test
exit $LASTEXITCODE
