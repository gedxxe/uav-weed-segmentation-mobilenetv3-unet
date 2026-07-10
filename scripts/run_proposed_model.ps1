[CmdletBinding()]
param(
    [string]$RootPath = ".",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 2,
    [int]$NFolds = 2,
    [int]$MaxEpochs = 20,
    [int]$NTrials = 1,
    [int]$EarlyStopPatience = 10,
    [int]$LrSchedulerPatience = 5,
    [string]$Subset = "test",
    [string]$RunPrefix = "proposed_mnv3",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("v1", "v2")]
    [string]$Variant = "v1",
    [ValidateSet("dice", "ce", "ce_dice", "ce_dice_aux_foreground")]
    [string]$Loss = "ce_dice",
    [ValidateSet("none", "auto")]
    [string]$ClassWeights = "auto",
    [double]$ClassWeightMax = 5.0,
    [ValidateSet("inverse_frequency", "sqrt_inverse")]
    [string]$ClassWeightStrategy = "inverse_frequency",
    [double]$CeWeight = 1.0,
    [double]$DiceWeight = 1.0,
    [double]$ForegroundAuxWeight = 0.3,
    [ValidateSet("dice", "same", "macro_f1", "weed_f1", "foreground_macro_f1")]
    [string]$ValidationLoss = "macro_f1",
    [switch]$CleanStudy,
    [switch]$NoPretrained,
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

$Mode = "proposed"
if ($Variant -eq "v2") {
    $Mode = "proposed_v2"
}

$ArgsList = @(
    "scripts\run_model_suite.py",
    $Mode,
    "--root_path", $RootPath,
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--n_folds", "$NFolds",
    "--max_epochs", "$MaxEpochs",
    "--n_trials", "$NTrials",
    "--early_stop_patience", "$EarlyStopPatience",
    "--lr_scheduler_patience", "$LrSchedulerPatience",
    "--subset", $Subset,
    "--run_prefix", $RunPrefix,
    "--device", $Device,
    "--proposed_loss", $Loss,
    "--proposed_class_weights", $ClassWeights,
    "--class_weight_max", "$ClassWeightMax",
    "--proposed_class_weight_strategy", $ClassWeightStrategy,
    "--ce_weight", "$CeWeight",
    "--dice_weight", "$DiceWeight",
    "--foreground_aux_weight", "$ForegroundAuxWeight",
    "--proposed_validation_loss", $ValidationLoss,
    "--evaluation_input_size", "$EvaluationInputSize",
    "--warmup_iterations", "$WarmupIterations",
    "--benchmark_iterations", "$BenchmarkIterations"
)

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
