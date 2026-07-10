[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("resnet18", "resnet34", "resnet50", "resnet101")]
    [string]$EncoderName,

    [string]$RootPath = ".",
    [int]$BatchSize,
    [int]$NumWorkers = 2,
    [int]$NFolds = 2,
    [int]$MaxEpochs = 20,
    [int]$NTrials = 1,
    [int]$EarlyStopPatience = 10,
    [int]$LrSchedulerPatience = 5,
    [string]$Subset = "test",
    [string]$RunPrefix,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("v1", "v2")]
    [string]$ProposedVariant = "v1",
    [ValidateSet("dice", "ce", "ce_dice", "ce_dice_aux_foreground")]
    [string]$Loss = "dice",
    [ValidateSet("none", "auto")]
    [string]$ClassWeights = "none",
    [ValidateSet("dice", "ce", "ce_dice", "ce_dice_aux_foreground")]
    [string]$ProposedLoss = "ce_dice",
    [ValidateSet("none", "auto")]
    [string]$ProposedClassWeights = "auto",
    [double]$ClassWeightMax = 5.0,
    [ValidateSet("inverse_frequency", "sqrt_inverse")]
    [string]$ClassWeightStrategy = "inverse_frequency",
    [ValidateSet("inverse_frequency", "sqrt_inverse")]
    [string]$ProposedClassWeightStrategy = "inverse_frequency",
    [double]$CeWeight = 1.0,
    [double]$DiceWeight = 1.0,
    [double]$ForegroundAuxWeight = 0.3,
    [ValidateSet("dice", "same", "macro_f1", "weed_f1", "foreground_macro_f1")]
    [string]$ValidationLoss = "dice",
    [ValidateSet("dice", "same", "macro_f1", "weed_f1", "foreground_macro_f1")]
    [string]$ProposedValidationLoss = "macro_f1",
    [switch]$CleanStudy,
    [switch]$NoPretrained,
    [switch]$BaselineOnly,
    [switch]$SkipProposedTraining,
    [switch]$SkipTraining,
    [switch]$SkipPrediction,
    [switch]$SkipEvaluation,
    [switch]$SkipEfficiency,
    [int]$EvaluationInputSize = 480,
    [int]$WarmupIterations = 5,
    [int]$BenchmarkIterations = 20,
    [switch]$CpuLatency,
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv was not found at $Python. Create it first with scripts\setup_windows_cuda.ps1."
}

if (-not $RunPrefix) {
    $RunPrefix = "archcmp_$EncoderName"
}

$Mode = if ($BaselineOnly) {
    "baseline"
} elseif ($ProposedVariant -eq "v2") {
    "full_v2"
} else {
    "full"
}

$ArgsList = @(
    "scripts\run_model_suite.py",
    $Mode,
    "--encoder", $EncoderName,
    "--root_path", $RootPath,
    "--num_workers", "$NumWorkers",
    "--n_folds", "$NFolds",
    "--max_epochs", "$MaxEpochs",
    "--n_trials", "$NTrials",
    "--early_stop_patience", "$EarlyStopPatience",
    "--lr_scheduler_patience", "$LrSchedulerPatience",
    "--subset", $Subset,
    "--run_prefix", $RunPrefix,
    "--device", $Device,
    "--loss", $Loss,
    "--class_weights", $ClassWeights,
    "--proposed_loss", $ProposedLoss,
    "--proposed_class_weights", $ProposedClassWeights,
    "--class_weight_max", "$ClassWeightMax",
    "--class_weight_strategy", $ClassWeightStrategy,
    "--proposed_class_weight_strategy", $ProposedClassWeightStrategy,
    "--ce_weight", "$CeWeight",
    "--dice_weight", "$DiceWeight",
    "--foreground_aux_weight", "$ForegroundAuxWeight",
    "--validation_loss", $ValidationLoss,
    "--proposed_validation_loss", $ProposedValidationLoss,
    "--evaluation_input_size", "$EvaluationInputSize",
    "--warmup_iterations", "$WarmupIterations",
    "--benchmark_iterations", "$BenchmarkIterations"
)

if ($BatchSize -gt 0) {
    $ArgsList += @("--batch_size", "$BatchSize")
}
if ($CleanStudy) {
    $ArgsList += "--clean_study"
}
if ($NoPretrained) {
    $ArgsList += "--no-pretrained"
}
if ($SkipTraining) {
    $ArgsList += "--skip_training"
}
if ($SkipPrediction) {
    $ArgsList += "--skip_prediction"
}
if ($SkipEvaluation) {
    $ArgsList += "--skip_evaluation"
}
if ($SkipEfficiency) {
    $ArgsList += "--skip_efficiency"
}
if ($SkipProposedTraining) {
    $ArgsList += "--skip_proposed_training"
}
if ($CpuLatency) {
    $ArgsList += "--cpu_latency"
}
if ($PlanOnly) {
    $ArgsList += "--plan_only"
}

& $Python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $Python $($ArgsList -join ' ')"
}
