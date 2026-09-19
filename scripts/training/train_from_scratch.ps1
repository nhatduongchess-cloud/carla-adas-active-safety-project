$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venvCarLa\Scripts\python.exe'
$train = Join-Path $PSScriptRoot 'train.py'
$data = Join-Path $project 'carla_dataset_v1\yolo_adas_v1\data.yaml'
$base = Join-Path $project 'yolov8n.pt'
$projectRuns = Join-Path $project 'runs\object_perception'
$runName = 'yolov8n_adas_seed42_fresh'

Write-Host 'Starting fresh ADAS YOLO training' -ForegroundColor Cyan
Write-Host "Run: $runName"
Write-Host 'The previous checkpoints are preserved.'

& $python -u $train `
    --data $data `
    --base $base `
    --project $projectRuns `
    --name $runName `
    --epochs 100 `
    --batch 16 `
    --imgsz 640 `
    --workers 0 `
    --fraction 1.0 `
    --val-fraction 1.0 `
    --device 0 `
    --patience 15 `
    --seed 42

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host 'Training and post-train validation completed.' -ForegroundColor Green
} else {
    Write-Host "Training exited with code $exitCode; last.pt should be available for resume." -ForegroundColor Yellow
}
Read-Host 'Press Enter to close this training window'
exit $exitCode
