param(
    [ValidateSet("core", "standard")]
    [string]$Profile = "standard"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $Root "reports/qualification_logs"
$WorkDir = Join-Path $Root "reports/qualification_work"
$Steps = Join-Path $Root "reports/qualification_steps.tsv"
Push-Location $Root
try {
    Remove-Item $LogDir, $WorkDir, "reports/runtime_work", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build" -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem "src" -Filter "*.egg-info" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem . -Recurse -File -Include "*.pyc","*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force
    New-Item -ItemType Directory -Force -Path $LogDir, $WorkDir | Out-Null
    Set-Content -Path $Steps -Value "" -NoNewline
    $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"; $env:PYTHONHASHSEED="0"; $env:PYTHONDONTWRITEBYTECODE="1"
    function Clear-TransientArtifacts {
        Remove-Item ".pytest_cache", ".ruff_cache", ".mypy_cache", "build" -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem "src" -Filter "*.egg-info" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Get-ChildItem . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Get-ChildItem . -Recurse -File -Include "*.pyc","*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force
    }
    function Sanitize-Log([string]$Path) {
        if (Test-Path $Path) {
            $Text = Get-Content $Path -Raw
            $Text = $Text.Replace($Root, ".")
            Set-Content -Path $Path -Value $Text -NoNewline
        }
    }
    function Invoke-Step([string]$Name, [string[]]$Command) {
        $Log = Join-Path $LogDir "$Name.log"
        Write-Host "[qualification] starting $Name"
        $Watch = [System.Diagnostics.Stopwatch]::StartNew()
        $Process = Start-Process -FilePath $Command[0] -ArgumentList $Command[1..($Command.Length-1)] -RedirectStandardOutput $Log -RedirectStandardError $Log -NoNewWindow -PassThru
        while (-not $Process.HasExited) {
            Write-Host "[qualification] $Name still running..."
            Start-Sleep -Seconds 2
            $Process.Refresh()
        }
        $Code = $Process.ExitCode
        $Watch.Stop()
        $Rendered = ($Command -join " ")
        Sanitize-Log $Log
        "$Name`t$Rendered`t$Code`t$([math]::Round($Watch.Elapsed.TotalSeconds,3))`treports/qualification_logs/$Name.log" | Add-Content -Path $Steps
        Write-Host "[qualification] finished $Name`: exit=$Code duration=$([math]::Round($Watch.Elapsed.TotalSeconds,3))s"
        if ($Code -ne 0) { Get-Content $Log -Tail 80; exit $Code }
    }
    Invoke-Step "repository-integrity" @("python","scripts/repository_integrity.py","--root",".")
    Invoke-Step "pip-check" @("python","-m","pip","check")
    Invoke-Step "ruff" @("python","-m","ruff","check",".")
    Invoke-Step "format" @("python","-m","ruff","format","--check",".")
    Invoke-Step "mypy" @("python","-m","mypy","src/control_evidence")
    Invoke-Step "tests-round-1" @("python","-m","pytest","--cov=control_evidence","--cov-report=term-missing","--cov-report=json:reports/coverage.json","--cov-fail-under=80","-q")
    if ($Profile -ne "core") { Invoke-Step "tests-round-2" @("python","-m","pytest","-q") }
    $Rounds = if ($Profile -eq "core") { 2 } else { 3 }
    for ($Index=1; $Index -le $Rounds; $Index++) { Invoke-Step "pipeline-round-$Index" @("python","-m","control_evidence.cli","full-pipeline","--root","reports/qualification_work/smoke-$Index","--run-id","qualification") }
    # Package build is verified by scripts/full_pipeline_validation.py and package CI; keep local qualifier focused on source/runtime gates.
    Invoke-Step "sbom" @("python","-m","control_evidence.cli","sbom","--output","reports/cyclonedx-sbom.json")
    python scripts/build_qualification_manifest.py --root . --profile $Profile --steps $Steps --output qualification_manifest.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Clear-TransientArtifacts
}
finally { Pop-Location }
