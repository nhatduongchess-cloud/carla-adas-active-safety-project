# run_portfolio_demo.ps1
# One-shot portfolio demo + results runner (stable envelope).
# Usage (from the project root, with CARLA server already running):
#   powershell -ExecutionPolicy Bypass -File .\run_portfolio_demo.ps1
#
# Records go to .\logs\ . Start/stop your screen recorder at the prompts.
# On a native server crash (B01): stop; do NOT rerun a heavier config or raise timeouts.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venvCarLa\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "Khong tim thay interpreter: $py"; exit 1 }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'logs') | Out-Null

function Step($title) { Write-Host "`n==== $title ====" -ForegroundColor Cyan }
function RunStep($label, $arguments) {
    Step $label
    & $py @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] '$label' ket thuc voi ma loi $LASTEXITCODE." -ForegroundColor Yellow
        Write-Host "    Neu la crash native (B01): DUNG lai, giu nguyen crash artifact, dung chay lai cau hinh nang hon." -ForegroundColor Yellow
        $ans = Read-Host "Van tiep tuc buoc sau? (y/n)"
        if ($ans -ne 'y') { Write-Host "Da dung theo yeu cau."; exit $LASTEXITCODE }
    }
}

# 0) Readiness (read-only)
RunStep 'Kiem tra server san sang' @('wait_for_carla.py','--timeout','30')

$common = @('--town','Town02','--seed','42',
            '--performance-profile','low-memory','--runtime-mode','async-stable',
            '--duration','20')

# 1) Clip A - clear-road driving
Read-Host "`nBAT phan mem quay man hinh, roi nhan Enter de chay CLIP A (lai duong thoang)"
RunStep 'Clip A - clear-road' (@('-u','chinh.py') + $common + @('--run-report','logs\demo_clear.json'))
Read-Host "TAT quay (hoac giu quay), nhan Enter de sang Clip B"

# 2) Clip B - AEB / hazard
Read-Host "Nhan Enter de chay CLIP B (phanh gap / AEB)"
RunStep 'Clip B - AEB/hazard' (@('-u','chinh.py','--hazard') + $common + @('--run-report','logs\demo_hazard.json'))
Read-Host "Nhan Enter de sang Clip C"

# 3) Clip C - L3 takeover -> MRM
Read-Host "Nhan Enter de chay CLIP C (takeover -> MRM)"
RunStep 'Clip C - takeover/MRM' (@('-u','chinh.py','--driver-takeover') + $common + @('--run-report','logs\demo_mrm.json'))
Read-Host "TAT quay man hinh. Nhan Enter de chay bo ket qua (khong can quay)"

# 4) Curated core suite, 3 seeds (18 runs) -> results table
RunStep 'Bo core scenario - 3 seed' @('run_scenarios.py','--scenarios','core',
        '--town','Town02','--seeds','42,1337,2026','--report','logs\portfolio_core_3seed.json')

Write-Host "`n==== XONG ====" -ForegroundColor Green
Write-Host "Cac file ket qua nam trong .\logs\ :"
Write-Host "  demo_clear.json, demo_hazard.json, demo_mrm.json, portfolio_core_3seed.json"
Write-Host "Gui/dan cac file nay lai cho Claude de dien bang ket qua vao README."

