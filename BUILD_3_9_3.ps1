$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}

$TargetTemp = Join-Path $Root 'pytest_tmp_393_targeted'
$FullTemp = Join-Path $Root 'pytest_tmp_393_full'

Remove-Item -Recurse -Force $TargetTemp -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $FullTemp -ErrorAction SilentlyContinue

Write-Host '== Targeted 3.9.3 tests =='
Push-Location (Join-Path $Root 'source')
try {
    & $Python -m pytest .\tests\test_opponent_policy_runtime.py -q --basetemp $TargetTemp
    if ($LASTEXITCODE -ne 0) {
        throw "Targeted tests failed with exit code $LASTEXITCODE"
    }

    Write-Host '== Full tests =='
    & $Python -m pytest .\tests -q --basetemp $FullTemp
    if ($LASTEXITCODE -ne 0) {
        throw "Full tests failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Set-Location $Root

Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\dist -ErrorAction SilentlyContinue

Write-Host '== Build HeroesWM_Worker_3_9_3 =='

& $Python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name HeroesWM_Worker_3_9_3 `
  --paths .\source `
  --add-data '.\source\cards_catalog.json;.' `
  --add-data '.\source\policy_models.json;.' `
  --add-data '.\source\opponent_policy.json;.' `
  --hidden-import onnxruntime.capi._pybind_state `
  --collect-all ddddocr `
  --collect-all onnxruntime `
  .\source\main.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host ''
Write-Host 'Done:' (Join-Path $Root 'dist\HeroesWM_Worker_3_9_3.exe')