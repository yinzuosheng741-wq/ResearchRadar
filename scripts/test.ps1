param([string[]]$PytestArgs = @('-q'))

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the project development .venv and install test dependencies first.'
}
$testRoot = Join-Path $projectRoot ('.pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$savedTemp, $savedTmp = $env:TEMP, $env:TMP
try {
    $env:TEMP = $testRoot
    $env:TMP = $testRoot
    Push-Location -LiteralPath $projectRoot
    try {
        & $pythonPath -m pytest @PytestArgs "--basetemp=$(Join-Path $testRoot 'pytest')"
        $testExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    $env:TEMP, $env:TMP = $savedTemp, $savedTmp
}
# Preserve temporary artifacts for diagnosis. They are ignored by Git.
exit $testExitCode
