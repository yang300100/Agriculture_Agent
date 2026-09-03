$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$configPath = Join-Path $PSScriptRoot "训练设置.psd1"
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "找不到训练设置文件：$configPath"
}
$config = Import-PowerShellDataFile -LiteralPath $configPath

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "未找到 Python。请安装 Python 3.10 或 3.11，并勾选添加到 PATH。"
    }
    Write-Host "[1/3] 正在创建独立 Python 环境..." -ForegroundColor Cyan
    & py "-$($config.PythonVersion)" -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "指定版本不可用，正在尝试 Python 3.10..." -ForegroundColor Yellow
        & py -3.10 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw "创建 Python 虚拟环境失败。" }
}

if ($config.InstallDependencies) {
    Write-Host "[2/3] 正在检查训练依赖..." -ForegroundColor Cyan
    & $python -c "import torch, torchvision, timm, PIL, numpy, tqdm, matplotlib, sklearn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "升级 pip 失败。" }
        & $python -m pip install -r "model_all\portable_requirements.txt"
        if ($LASTEXITCODE -ne 0) { throw "安装训练依赖失败。" }
    } else {
        Write-Host "训练依赖已经就绪。" -ForegroundColor Green
    }
}

$arguments = @(
    "model_all\train_hmpd.py",
    "--prepared-dir", [string]$config.PreparedDir,
    "--experiment-dir", [string]$config.ExperimentDir,
    "--output", [string]$config.OutputModel,
    "--backbone", [string]$config.Backbone,
    "--image-size", [string]$config.ImageSize,
    "--fusion-channels", [string]$config.FusionChannels,
    "--dropout", [string]$config.Dropout,
    "--consistency-strength", [string]$config.ConsistencyStrength,
    "--epochs", [string]$config.Epochs,
    "--freeze-epochs", [string]$config.FreezeEpochs,
    "--batch-size", [string]$config.BatchSize,
    "--workers", [string]$config.Workers,
    "--learning-rate", [string]$config.LearningRate,
    "--weight-decay", [string]$config.WeightDecay,
    "--crop-loss-weight", [string]$config.CropLossWeight,
    "--disease-loss-weight", [string]$config.DiseaseLossWeight,
    "--severity-loss-weight", [string]$config.SeverityLossWeight,
    "--consistency-loss-weight", [string]$config.ConsistencyLossWeight,
    "--seed", [string]$config.Seed
)

if (-not $config.UsePretrained) { $arguments += "--no-pretrained" }
if ($config.Checkpoint) {
    $arguments += @("--checkpoint", [string]$config.Checkpoint)
} elseif ($config.Resume) {
    $arguments += "--resume"
}
if ([int]$config.MaxTrainBatches -gt 0) {
    $arguments += @("--max-train-batches", [string]$config.MaxTrainBatches)
}
if ([int]$config.MaxEvalBatches -gt 0) {
    $arguments += @("--max-eval-batches", [string]$config.MaxEvalBatches)
}

Write-Host "[3/3] 即将启动 HMPD-Net 训练" -ForegroundColor Cyan
Write-Host "模型：$($config.Backbone) | Epoch：$($config.Epochs) | Batch：$($config.BatchSize) | 图像：$($config.ImageSize)"
Write-Host "实验目录：$($config.ExperimentDir)"
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "训练进程异常结束，退出码：$LASTEXITCODE" }

Write-Host "训练完成，结果位于 $($config.ExperimentDir)" -ForegroundColor Green
