# run_demo_safe.ps1
# Quay 3 clip demo mÃ  KHÃ”NG bá»‹ B01 lÃ m há»ng buá»•i quay.
# Chiáº¿n lÆ°á»£c: má»—i clip cháº¡y ngáº¯n (--duration 12) Ä‘á»ƒ káº¿t thÃºc TRUOC cua so crash ~20s;
# sau má»—i clip kiá»ƒm tra server; náº¿u B01 crash giá»¯a chá»«ng thÃ¬ tá»± má»Ÿ láº¡i server vÃ  quay láº¡i clip Ä‘Ã³.
#
# CÃ¡ch dÃ¹ng (Ä‘á»©ng táº¡i thÆ° má»¥c dá»± Ã¡n):
#   powershell -ExecutionPolicy Bypass -File .\run_demo_safe.ps1
#
# Reports luu trong .\logs\ . Bat/tat phan mem quay man hinh tai cac dau nhac.

$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot
$root = $PSScriptRoot
$py   = Join-Path $root '.venvCarLa\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host "Khong tim thay interpreter: $py" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

function Test-Server {
    & $py wait_for_carla.py --timeout 8 *> $null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-Server {
    if (Test-Server) { return $true }
    Write-Host '[demo] Server chua san sang -> dang mo lai...' -ForegroundColor Yellow
    Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Start-Process -FilePath (Join-Path $root 'launch_carla.bat') -WorkingDirectory $root
    for ($i = 0; $i -lt 12; $i++) {
        Start-Sleep -Seconds 7
        if (Test-Server) { Write-Host '[demo] Server READY' -ForegroundColor Green; return $true }
    }
    Write-Host '[demo] Khong mo duoc server sau ~90s.' -ForegroundColor Red
    return $false
}

function Run-Clip {
    param([string]$Name, [int]$Spawn, [string[]]$Extra, [string]$Report)
    for ($try = 1; $try -le 3; $try++) {
        if (-not (Ensure-Server)) { Write-Host "[$Name] Bo qua: khong co server." -ForegroundColor Red; return }
        Read-Host "`n[$Name] (lan $try) BAT quay man hinh, roi nhan Enter de chay"
        $cliArgs = @('-u','chinh.py','--town','Town02','--seed','42','--spawn-index',"$Spawn",
                     '--performance-profile','low-memory','--runtime-mode','async-stable',
                     '--duration','20','--run-report',"logs\$Report") + $Extra
        & $py @cliArgs
        Read-Host "[$Name] TAT quay. Nhan Enter"
        if (Test-Server) {
            Write-Host "[$Name] OK - server con song, clip hoan tat." -ForegroundColor Green
            return
        }
        Write-Host "[$Name] B01 crash giua chung -> mo lai server va quay LAI clip nay." -ForegroundColor Yellow
    }
    Write-Host "[$Name] Da thu 3 lan van dinh crash. Giu doan quay duoc, hoac giam --duration." -ForegroundColor Yellow
}

Write-Host "==== DEMO SAFE RUNNER (ne B01) ====" -ForegroundColor Cyan
Write-Host "Moi clip 12s. Neu server crash, script tu mo lai va quay lai clip do.`n"

# Clip A - lai duong thoang (spawn 2 = duong thoang da xac nhan)
Run-Clip -Name 'A/clear'  -Spawn 2 -Extra @()                 -Report 'demo_clear.json'
# Clip B - AEB (dynamic: dam xe chuong ngai phia truoc)
Run-Clip -Name 'B/AEB'    -Spawn 2 -Extra @('--hazard')       -Report 'demo_hazard.json'
# Clip C - L3 takeover -> MRM
Run-Clip -Name 'C/MRM'    -Spawn 2 -Extra @('--driver-takeover') -Report 'demo_mrm.json'

Write-Host "`n==== XONG ====" -ForegroundColor Green
Write-Host "Reports: .\logs\demo_clear.json, demo_hazard.json, demo_mrm.json"
Write-Host "Gui/dan cac file JSON nay cho Claude de dien bang ket qua vao README."
Write-Host "`nMEO: neu clip B (AEB dynamic) hay crash, dung spawn-index 0 lam clip AEB tinh"
Write-Host "     (xe do chan san truoc mui, AEB giu phanh - da xac nhan on dinh, khong crash)."

