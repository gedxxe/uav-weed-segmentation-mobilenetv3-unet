param(
    [string]$PythonVersion = "3.13",
    [string]$CudaWheel = "cu128",
    [switch]$SkipTorch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:TORCH_HOME = Join-Path $RepoRoot ".cache\torch"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

Write-Host "Creating .venv with py -$PythonVersion"
Invoke-Checked -FilePath "py" -Arguments @("-$PythonVersion", "-m", "venv", ".venv")

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Invoke-Checked -FilePath $Python -Arguments @("--version")

Write-Host "Upgrading pip tooling"
Invoke-Checked -FilePath $Python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel")

if (-not $SkipTorch) {
    $TorchIndex = "https://download.pytorch.org/whl/$CudaWheel"
    Write-Host "Installing PyTorch CUDA wheel from $TorchIndex"
    Invoke-Checked -FilePath $Python -Arguments @("-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", $TorchIndex)
}

Write-Host "Installing project dependencies"
Invoke-Checked -FilePath $Python -Arguments @("-m", "pip", "install", "-r", "requirements.txt")

Write-Host "Verifying CUDA runtime"
Invoke-Checked -FilePath $Python -Arguments @("scripts\verify_cuda.py")

Write-Host "Checking dataset folders"
Invoke-Checked -FilePath $Python -Arguments @("scripts\check_dataset.py")
