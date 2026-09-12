# run_demo_aeb_mrm.ps1
# Chay lai 2 clip con thieu report: AEB va MRM (moi clip 12s de ne B01 ~20s).
# Tu kiem tra server sau moi clip; neu crash -> mo lai server + quay lai clip do.
#
# Dung:  powershell -ExecutionPolicy Bypass -File .\run_demo_aeb_mrm.ps1 [-AebSeconds 25] [-MrmSeconds 15]
# Reports: logs\demo_hazard.json (AEB), logs\demo_mrm.json (MRM)
#
# AEB (spawn 0) la canh TINH -> on dinh, co the chay dai. MRM (chay) de dinh B01 hon.
# Report chi ghi khi run KET THUC (het duration, hoac crash bat duoc); hard-crash thi khong ghi.

param(
    [int]$AebSeconds = 20,
    [int]$MrmSeconds = 20
)

$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot
$root = $PSScriptRoot
$py   = Join-Path $root '.venvCarLa\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

function Test-Server { & $py wait_for_carla.py --timeout 8 *> $null; return ($LASTEXITCODE -eq 0) }

function Ensure-Server {
    if (Test-Server) { return $true }
    Write-Host '[demo] Mo lai server...' -ForegroundColor Yellow
    Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
    Start-Process -FilePath (Join-Path $root 'launch_carla.bat') -WorkingDirectory $root
    for ($i=0; $i -lt 12; $i++) { Start-Sleep 7; if (Test-Server) { Write-Host '[demo] READY' -ForegroundColor Green; return $true } }
    Write-Host '[demo] Khong mo duoc server.' -ForegroundColor Red; return $false
}

function Run-Clip {
    param([string]$Name, [int]$Spawn, [int]$Seconds, [string[]]$Extra, [string]$Report)
    for ($try=1; $try -le 3; $try++) {
        if (-not (Ensure-Server)) { return }
        Read-Host "`n[$Name] (lan $try, ${Seconds}s) BAT quay man hinh roi Enter de chay"
        $a = @('-u','chinh.py','--town','Town02','--seed','42','--spawn-index',"$Spawn",
               '--performance-profile','low-memory','--runtime-mode','async-stable',
               '--duration',"$Seconds",'--run-report',"logs\$Report") + $Extra
        & $py @a
        Read-Host "[$Name] TAT quay. Enter de tiep"
        if (Test-Path "logs\$Report") { Write-Host "[$Name] OK - da ghi logs\$Report" -ForegroundColor Green; return }
        Write-Host "[$Name] Chua co report (co the crash som) -> thu lai." -ForegroundColor Yellow
    }
    Write-Host "[$Name] Da thu 3 lan. Giam --duration xuong 8 neu van kho." -ForegroundColor Yellow
}

Write-Host "==== AEB + MRM ====" -ForegroundColor Cyan
# AEB - dung spawn-index 0: xe do chan san truoc mui, AEB giu phanh (on dinh -> chay dai duoc)
Run-Clip -Name 'AEB' -Spawn 0 -Seconds $AebSeconds -Extra @()                    -Report 'demo_hazard.json'
# MRM - L3 takeover -> minimal-risk maneuver (xe chay -> giu ngan hon)
Run-Clip -Name 'MRM' -Spawn 2 -Seconds $MrmSeconds -Extra @('--driver-takeover') -Report 'demo_mrm.json'

Write-Host "`n==== XONG ====" -ForegroundColor Green
Write-Host "Reports: logs\demo_hazard.json (AEB), logs\demo_mrm.json (MRM)"
Write-Host "Bao Claude doc 2 file nay de dien not bang ket qua vao README."
Write-Host "`n(Neu muon AEB DONG - xe dam vao chuong ngai - doi dong AEB thanh:"
Write-Host "   Run-Clip -Name 'AEB' -Spawn 2 -Extra @('--hazard') -Report 'demo_hazard.json'  )"

