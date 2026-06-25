Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# v0.7 learned risk-routing experiment controller.
# This does not modify datasets and does not require a GPU.
# It creates a reproducible baseline-vs-candidate comparison under a fixed review budget.

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectName = "risk-controlled-vlm-evidence-platform"
$RunId = "routing-v070-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$OutputRoot = Join-Path $RepoRoot "outputs-local"
$RunRoot = Join-Path (Join-Path $OutputRoot "outputs") "routing_experiments"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "[$Name] $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "[$Name] failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repository root was not found: $RepoRoot"
}

if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) {
    Write-Host "Creating a Python 3.11 virtual environment."
    & py -3.11 -m venv (Join-Path $RepoRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv with py -3.11."
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

Push-Location -LiteralPath $RepoRoot
try {
    Invoke-Checked -Name "P01_DEPENDENCIES" -FilePath $Py -Arguments @(
        "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
    )
    Invoke-Checked -Name "P01_PROJECT_INSTALL" -FilePath $Py -Arguments @(
        "-m", "pip", "install", "-e", ".[api,dev,datasets,ml]"
    )
    Invoke-Checked -Name "P01_PIP_CHECK" -FilePath $Py -Arguments @("-m", "pip", "check")
    Invoke-Checked -Name "P02_RUFF" -FilePath $Py -Arguments @("-m", "ruff", "check", ".")
    Invoke-Checked -Name "P02_FORMAT" -FilePath $Py -Arguments @("-m", "ruff", "format", "--check", ".")
    Invoke-Checked -Name "P02_MYPY" -FilePath $Py -Arguments @("-m", "mypy", "src/control_evidence")
    Invoke-Checked -Name "P02_TESTS" -FilePath $Py -Arguments @("-m", "pytest", "-q")

    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

    Write-Host ""
    Write-Host "[P03_EXPERIMENT] Frozen rule baseline vs calibrated learned router"
    & $Py -m control_evidence.cli compare-risk-routing `
        --root $OutputRoot `
        --seed 701 `
        --n-per-family 360 `
        --capacity-fraction 0.20 `
        --bootstrap-samples 1000 `
        --run-id $RunId
    $ExperimentExit = $LASTEXITCODE

    # 0 = candidate promoted; 3 = candidate retained baseline with full artifacts.
    if ($ExperimentExit -notin @(0, 3)) {
        throw "[P03_EXPERIMENT] unexpected exit code $ExperimentExit"
    }

    $ExperimentDir = Join-Path $RunRoot "runs\$RunId"
    Invoke-Checked -Name "P04_VALIDATE_ARTIFACTS" -FilePath $Py -Arguments @(
        "-m", "control_evidence.cli", "validate-run", $ExperimentDir
    )

    $SummaryPath = Join-Path $ExperimentDir "routing_experiment_summary.json"
    $PromotionPath = Join-Path $ExperimentDir "promotion_gate.json"
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $Promotion = Get-Content -LiteralPath $PromotionPath -Raw | ConvertFrom-Json

    $LocalSummary = [ordered]@{
        project_name = $ProjectName
        project_version = $Summary.project_version
        run_id = $RunId
        experiment_dir = $ExperimentDir
        summary_path = $SummaryPath
        promotion_gate_path = $PromotionPath
        decision = $Promotion.decision
        baseline_residual_weighted_risk = $Summary.baseline.routing.residual_weighted_risk
        candidate_residual_weighted_risk = $Summary.candidate.routing.residual_weighted_risk
        baseline_pr_auc = $Summary.baseline.probability_metrics.pr_auc
        candidate_pr_auc = $Summary.candidate.probability_metrics.pr_auc
        bootstrap_ci95 = $Summary.paired_bootstrap
        completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $LocalSummaryPath = Join-Path $ExperimentDir "local_execution_summary.json"
    $LocalSummary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $LocalSummaryPath -Encoding UTF8

    Write-Host ""
    Write-Host "=== LEARNED RISK ROUTING EXPERIMENT COMPLETE ==="
    Write-Host "Decision: $($Promotion.decision)"
    Write-Host "Experiment artifacts: $ExperimentDir"
    Write-Host "Summary: $LocalSummaryPath"
    Write-Host ""
    Write-Host "Claim boundary: synthetic group-held-out operational-error routing only."

    if ($ExperimentExit -eq 3) {
        Write-Host "Candidate was not promoted; baseline is retained. Artifacts remain valid for diagnosis."
        exit 3
    }
}
finally {
    Pop-Location
}
