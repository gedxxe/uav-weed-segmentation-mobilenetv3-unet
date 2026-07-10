[CmdletBinding()]
param(
    [string]$RootPath = ".",
    [ValidateSet("resnet34", "resnet50")]
    [string[]]$Encoders = @("resnet34", "resnet50"),
    [switch]$Apply,
    [switch]$KeepWebcamOutputs,
    [switch]$KeepProposedV1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $RootPath))
if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "RootPath does not exist: $Root"
}

$BaselineArchitectures = @("fcn8s", "fcn16s", "fcn32s", "unet", "dlplus")
$ProtectedRelativePaths = @(
    ".venv",
    "data",
    "exports",
    "models\unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar",
    "results\training_logs"
)

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-IsUnderRoot {
    param([Parameter(Mandatory = $true)][string]$Path)
    $FullPath = Get-FullPath -Path $Path
    return $FullPath.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-RelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $FullPath = Get-FullPath -Path $Path
    if ($FullPath.Length -eq $Root.Length) {
        return "."
    }
    return $FullPath.Substring($Root.Length).TrimStart("\", "/")
}

function Test-IsProtected {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Relative = Get-RelativePath -Path $Path
    foreach ($Protected in $ProtectedRelativePaths) {
        if (
            $Relative.Equals($Protected, [System.StringComparison]::OrdinalIgnoreCase) -or
            $Relative.StartsWith("$Protected\", [System.StringComparison]::OrdinalIgnoreCase)
        ) {
            return $true
        }
    }
    return $false
}

function Add-Target {
    param(
        [Parameter(Mandatory = $true)][object]$Targets,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Reason
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $FullPath = Get-FullPath -Path $Path
    if (-not (Test-IsUnderRoot -Path $FullPath)) {
        throw "Refusing to include path outside workspace: $FullPath"
    }
    if (Test-IsProtected -Path $FullPath) {
        return
    }
    if (-not $Targets.Contains($FullPath)) {
        $Targets[$FullPath] = $Reason
    }
}

$Targets = [ordered]@{}

Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force |
    Where-Object {
        $_.Name -eq "__pycache__" -and
        (-not $_.FullName.StartsWith((Join-Path $Root ".venv"), [System.StringComparison]::OrdinalIgnoreCase))
    } |
    ForEach-Object { Add-Target -Targets $Targets -Path $_.FullName -Reason "python cache" }

foreach ($CacheDir in @(".pytest_cache", ".cache", "tmp")) {
    Add-Target -Targets $Targets -Path (Join-Path $Root $CacheDir) -Reason "workspace cache/temp"
}

if (-not $KeepWebcamOutputs) {
    Add-Target -Targets $Targets -Path (Join-Path $Root "results\webcam_inference_checks") -Reason "local webcam/demo capture output"
}

foreach ($Encoder in $Encoders) {
    foreach ($Architecture in $BaselineArchitectures) {
        $ModelName = "${Architecture}_${Encoder}"
        Add-Target `
            -Targets $Targets `
            -Path (Join-Path $Root "models\${ModelName}_dil0_bilin1_pre1.pth.tar") `
            -Reason "old baseline checkpoint for retraining"

        Get-ChildItem -LiteralPath (Join-Path $Root "results\predictions") -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                Add-Target -Targets $Targets -Path (Join-Path $_.FullName $ModelName) -Reason "old baseline prediction folder"
            }

        Get-ChildItem -LiteralPath (Join-Path $Root "results\reports") -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                Add-Target -Targets $Targets -Path (Join-Path $_.FullName $ModelName) -Reason "old baseline report folder"
            }
    }

    Get-ChildItem -LiteralPath (Join-Path $Root "results\reports") -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            foreach ($ReportName in @(
                "architecture_comparison_${Encoder}",
                "architecture_comparison_${Encoder}_proposed_v2",
                "baseline_comparison_${Encoder}",
                "full_evaluation_${Encoder}",
                "full_v2_evaluation_${Encoder}"
            )) {
                Add-Target -Targets $Targets -Path (Join-Path $_.FullName $ReportName) -Reason "old combined baseline comparison report"
            }
        }
}

Get-ChildItem -LiteralPath (Join-Path $Root "results") -Filter "*.db" -File -ErrorAction SilentlyContinue |
    Where-Object {
        $Name = $_.Name
        $MatchesEncoder = $false
        foreach ($Encoder in $Encoders) {
            if (
                $Name -like "archcmp_${Encoder}_*.db" -or
                $Name -match "_(fcn8s|fcn16s|fcn32s|unet|dlplus)_${Encoder}_dil0_bilin1_pre1\.db$"
            ) {
                $MatchesEncoder = $true
            }
        }
        return $MatchesEncoder
    } |
    ForEach-Object { Add-Target -Targets $Targets -Path $_.FullName -Reason "old baseline Optuna DB" }

if (-not $KeepProposedV1) {
    Add-Target `
        -Targets $Targets `
        -Path (Join-Path $Root "models\unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar") `
        -Reason "old proposed-v1 checkpoint"

    Get-ChildItem -LiteralPath (Join-Path $Root "results") -Filter "*.db" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.db" } |
        ForEach-Object { Add-Target -Targets $Targets -Path $_.FullName -Reason "old proposed-v1 Optuna DB" }

    foreach ($SubsetRoot in @("results\predictions", "results\reports")) {
        Get-ChildItem -LiteralPath (Join-Path $Root $SubsetRoot) -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                foreach ($Name in @("unet_mobilenetv3", "proposed_model", "proposed_evaluation")) {
                    Add-Target -Targets $Targets -Path (Join-Path $_.FullName $Name) -Reason "old proposed-v1 output"
                }
            }
    }

    Get-ChildItem -LiteralPath (Join-Path $Root "models\_trial_checkpoints") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1" } |
        ForEach-Object { Add-Target -Targets $Targets -Path $_.FullName -Reason "old proposed-v1 trial checkpoint folder" }
}

if ($Targets.Count -eq 0) {
    Write-Host "No baseline retrain cleanup targets found."
    exit 0
}

if ($Apply) {
    Write-Host "Deleting baseline retrain cleanup targets:"
} else {
    Write-Host "Dry run. The following baseline retrain cleanup targets would be deleted:"
}

foreach ($Target in $Targets.Keys | Sort-Object) {
    $Relative = Get-RelativePath -Path $Target
    $Reason = $Targets[$Target]
    Write-Host "  - $Relative [$Reason]"
    if ($Apply) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
}

if (-not $Apply) {
    Write-Host "Run again with -Apply to delete these targets."
}
