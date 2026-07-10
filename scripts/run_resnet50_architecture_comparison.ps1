[CmdletBinding()]
param(
    [string]$RootPath = ".",
    [int]$BatchSize = 4,
    [int]$NumWorkers = 2,
    [int]$NFolds = 2,
    [int]$MaxEpochs = 20,
    [int]$NTrials = 1,
    [int]$EarlyStopPatience = 10,
    [int]$LrSchedulerPatience = 5,
    [string]$Subset = "test",
    [string]$RunPrefix = "archcmp_resnet50",
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

$CoreScript = Join-Path $PSScriptRoot "run_architecture_comparison_core.ps1"
& $CoreScript `
    -EncoderName "resnet50" `
    -RootPath $RootPath `
    -BatchSize $BatchSize `
    -NumWorkers $NumWorkers `
    -NFolds $NFolds `
    -MaxEpochs $MaxEpochs `
    -NTrials $NTrials `
    -EarlyStopPatience $EarlyStopPatience `
    -LrSchedulerPatience $LrSchedulerPatience `
    -Subset $Subset `
    -RunPrefix $RunPrefix `
    -Device $Device `
    -ProposedVariant $ProposedVariant `
    -Loss $Loss `
    -ClassWeights $ClassWeights `
    -ProposedLoss $ProposedLoss `
    -ProposedClassWeights $ProposedClassWeights `
    -ClassWeightMax $ClassWeightMax `
    -ClassWeightStrategy $ClassWeightStrategy `
    -ProposedClassWeightStrategy $ProposedClassWeightStrategy `
    -CeWeight $CeWeight `
    -DiceWeight $DiceWeight `
    -ForegroundAuxWeight $ForegroundAuxWeight `
    -ValidationLoss $ValidationLoss `
    -ProposedValidationLoss $ProposedValidationLoss `
    -CleanStudy:$CleanStudy `
    -NoPretrained:$NoPretrained `
    -BaselineOnly:$BaselineOnly `
    -SkipProposedTraining:$SkipProposedTraining `
    -SkipTraining:$SkipTraining `
    -SkipPrediction:$SkipPrediction `
    -SkipEvaluation:$SkipEvaluation `
    -SkipEfficiency:$SkipEfficiency `
    -EvaluationInputSize $EvaluationInputSize `
    -WarmupIterations $WarmupIterations `
    -BenchmarkIterations $BenchmarkIterations `
    -CpuLatency:$CpuLatency `
    -PlanOnly:$PlanOnly
