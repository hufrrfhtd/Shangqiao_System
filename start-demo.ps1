#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port,

    [switch]$Headless
)

$ErrorActionPreference = 'Stop'
$demoPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $demoPython -PathType Leaf)) {
    throw 'The .venv Python environment is missing. Follow the Windows setup steps in README.md.'
}

$streamlitArguments = @(
    '-m', 'streamlit', 'run', 'app.py',
    '--server.address', '127.0.0.1',
    '--browser.gatherUsageStats', 'false'
)
if ($PSBoundParameters.ContainsKey('Port')) {
    $streamlitArguments += @('--server.port', $Port)
}
if ($Headless) {
    $streamlitArguments += @('--server.headless', 'true')
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $demoPython @streamlitArguments
    $demoExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $demoExitCode
