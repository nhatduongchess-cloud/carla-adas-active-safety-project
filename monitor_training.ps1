$ErrorActionPreference = 'SilentlyContinue'
$runDir = Join-Path $PSScriptRoot 'runs\object_perception\yolov8n_adas_seed42'
$csvPath = Join-Path $runDir 'results.csv'

Write-Host 'ADAS YOLO training monitor' -ForegroundColor Cyan
Write-Host "Run: $runDir"
Write-Host 'This window only monitors files/processes. Press Ctrl+C to close.'
Write-Host 'Every completed epoch is printed once; no history is hidden.'

$printedEpoch = 0
while ($true) {
    if (Test-Path $csvPath) {
        $rows = @(Import-Csv $csvPath)
        foreach ($row in $rows) {
            $epoch = [int]$row.epoch
            if ($epoch -gt $printedEpoch) {
                Write-Host ("{0} | epoch={1}/100 | box={2} cls={3} dfl={4} | " +
                    "precision={5} recall={6} mAP50={7} mAP50-95={8}" -f `
                    (Get-Date -Format 'HH:mm:ss'), $row.epoch,
                    $row.'train/box_loss', $row.'train/cls_loss', $row.'train/dfl_loss',
                    $row.'metrics/precision(B)', $row.'metrics/recall(B)',
                    $row.'metrics/mAP50(B)', $row.'metrics/mAP50-95(B)') -ForegroundColor Green
                $printedEpoch = $epoch
            }
        }
    } else {
        Write-Host ("{0} | waiting for results.csv..." -f (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Yellow
    }

    $gpu = @(nvidia-smi --query-compute-apps=pid,process_name,used_memory `
        --format=csv,noheader,nounits)
    $python = @(Get-Process -Name python)
    if ($gpu.Count -gt 0 -and $gpu[0]) {
        Write-Host ("{0} | GPU: {1}" -f (Get-Date -Format 'HH:mm:ss'), ($gpu -join '; ')) -ForegroundColor Magenta
    }
    if ($python.Count -eq 0) {
        Write-Host 'No Python process found; training may have ended.' -ForegroundColor Yellow
        break
    }
    Start-Sleep -Seconds 10
}

Read-Host 'Press Enter to close this monitor'
