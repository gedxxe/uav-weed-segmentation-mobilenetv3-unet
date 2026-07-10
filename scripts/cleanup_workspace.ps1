[CmdletBinding()]
param(
    [string]$RootPath = ".",
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $RootPath))
if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "RootPath does not exist: $Root"
}

function Test-IsUnderRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    return $FullPath.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-WorkspaceRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if ($FullPath.Length -eq $Root.Length) {
        return "."
    }
    return $FullPath.Substring($Root.Length).TrimStart("\", "/")
}

function Add-ExistingPath {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Targets,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not (Test-IsUnderRoot -Path $FullPath)) {
        throw "Refusing to include path outside workspace: $FullPath"
    }
    $Relative = Get-WorkspaceRelativePath -Path $FullPath
    if ($Relative -eq ".venv" -or $Relative.StartsWith(".venv\", [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }
    if ($Relative -eq "data" -or $Relative.StartsWith("data\", [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }
    if ($Relative -eq "models" -or $Relative.StartsWith("models\", [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }
    if (-not $Targets.Contains($FullPath)) {
        $Targets.Add($FullPath)
    }
}

$Targets = [System.Collections.Generic.List[string]]::new()

Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force |
    Where-Object {
        $_.Name -eq "__pycache__" -and
        (-not $_.FullName.StartsWith((Join-Path $Root ".venv"), [System.StringComparison]::OrdinalIgnoreCase))
    } |
    ForEach-Object { Add-ExistingPath -Targets $Targets -Path $_.FullName }

foreach ($CacheDir in @(".pytest_cache", ".cache")) {
    Add-ExistingPath -Targets $Targets -Path (Join-Path $Root $CacheDir)
}

Add-ExistingPath -Targets $Targets -Path (Join-Path $Root "tmp")

Get-ChildItem -LiteralPath (Join-Path $Root "results\reports\test") -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name.StartsWith("audit_check_", [System.StringComparison]::OrdinalIgnoreCase) -or
        $_.Name.StartsWith("evaluation_check_", [System.StringComparison]::OrdinalIgnoreCase)
    } |
    ForEach-Object { Add-ExistingPath -Targets $Targets -Path $_.FullName }

foreach ($Orphan in @(
    "results\predictions\cm_subset_test.pdf",
    "results\predictions\cm_subset_test_different_bbch_bbch_15.png",
    "results\predictions\cm_subset_test_different_bbch_bbch_19.png",
    "results\predictions\test\test_01_pred.png",
    "results\predictions\test\test_02_pred.png",
    "results\predictions\test\test_03_pred.png",
    "results\predictions\test_different_bbch\bbch15_img_pred.png",
    "results\predictions\test_different_bbch\bbch19_img_pred.png"
)) {
    Add-ExistingPath -Targets $Targets -Path (Join-Path $Root $Orphan)
}

if ($Targets.Count -eq 0) {
    Write-Host "No conservative cleanup targets found."
    exit 0
}

if ($Apply) {
    Write-Host "Deleting conservative cleanup targets:"
} else {
    Write-Host "Dry run. The following conservative cleanup targets would be deleted:"
}

foreach ($Target in $Targets | Sort-Object) {
    $Relative = Get-WorkspaceRelativePath -Path $Target
    Write-Host "  - $Relative"
    if ($Apply) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
}

if (-not $Apply) {
    Write-Host "Run again with -Apply to delete these targets."
}
