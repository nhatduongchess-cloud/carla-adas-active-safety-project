# run_until_crash.ps1
# Chay pipeline LIEN TUC (khong gioi han thoi gian) den khi B01 crash HOAC ban nhan 'q'
# tren cua so dashboard. KHONG auto-relaunch. Ghi lai crash (UE4 dump + report + journal).
# Quay man hinh bang OBS trong luc chay.
#
# Dung:  powershell -ExecutionPolicy Bypass -File .\scripts\demo\run_until_crash.ps1 [-SpawnIndex 2] [-Town Town10HD_Opt]

param(
    [int]$SpawnIndex = 2,
    [string]$Town = "Town10HD_Opt"
)

$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -Path $root
$py = Join-Path $root '.venvCarLa\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
$crashDir = Join-Path $env:LOCALAPPDATA 'CarlaUE4\Saved\Crashes'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'

function Test-Server { & $py scripts\tools\wait_for_carla.py --timeout 8 *> $null; return ($LASTEXITCODE -eq 0) }

# 1. Dam bao server dang chay (KHONG kill neu dang chay - de server tu chay)
if (-not (Test-Server)) {
    Write-Host '[demo] Server chua chay -> mo (khong dung/restart neu da chay)...' -ForegroundColor Yellow
    Start-Process -FilePath (Join-Path $root 'launch_carla.bat') -WorkingDirectory $root
    for ($i = 0; $i -lt 12; $i++) { Start-Sleep 7; if (Test-Server) { break } }
}
if (-not (Test-Server)) { Write-Host '[demo] Khong the san sang server.' -ForegroundColor Red; exit 1 }
Write-Host '[demo] Server READY.' -ForegroundColor Green

# 2. Ghi nhan crash dump moi nhat TRUOC khi chay (de phat hien dump moi sau crash)
$before = ''
if (Test-Path $crashDir) {
    $b = Get-ChildItem $crashDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($b) { $before = $b.Name }
}

Write-Host "`n=== CHAY LIEN TUC den khi crash ===" -ForegroundColor Cyan
Write-Host "Nhan 'q' tren cua so 'ADAS Portfolio - Central Dashboard' de dung som." -ForegroundColor Cyan
Write-Host "BAT OBS quay man hinh bay gio.`n"

# 3. Chay LIEN TUC: KHONG --duration -> chay den khi crash hoac 'q'
& $py -u chinh.py --town $Town --seed 42 --spawn-index $SpawnIndex `
    --performance-profile low-memory --runtime-mode async-stable `
    --run-report "logs\until_crash_$ts.json" `
    --crash-journal "logs\until_crash_journal_$ts.jsonl"
$exit = $LASTEXITCODE

# 4. Bao cao (KHONG relaunch)
Write-Host "`n=== KET THUC (exit=$exit) ===" -ForegroundColor Cyan
if (Test-Server) {
    Write-Host "[demo] Server VAN SONG -> ket thuc do 'q' hoac loi phia client, khong phai B01 server crash." -ForegroundColor Green
} else {
    Write-Host "[demo] Server DA CHET -> B01 native crash." -ForegroundColor Yellow
}

# Phat hien crash dump moi
$new = @()
if (Test-Path $crashDir) {
    $new = Get-ChildItem $crashDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne $before -and $_.LastWriteTime -gt (Get-Date).AddMinutes(-30) } |
        Sort-Object LastWriteTime -Descending
}
if ($new) {
    Write-Host '[demo] CRASH DUMP MOI:' -ForegroundColor Yellow
    $new | ForEach-Object { Write-Host ('   ' + $_.FullName) }
} else {
    Write-Host '[demo] Khong thay UE4 crash dump moi (co the la client crash, hoac ban nhan q).'
}
Write-Host ("Report:        logs\until_crash_$ts.json  (chi ghi neu ket thuc mem)")
Write-Host ("Crash journal: logs\until_crash_journal_$ts.jsonl")
Write-Host "`nGui cac file tren (va duong dan crash dump) cho Claude de phan tich."
