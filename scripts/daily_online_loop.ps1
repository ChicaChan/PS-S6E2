param(
    [string]$KernelRef = "chicachan/ps-s6e2-optimized-hybrid-gpu",
    [string]$Competition = "playground-series-s6e2",
    [string]$CloudOutputDir = "kaggle_outputs/cloud_latest",
    [string]$SampleSubmissionPath = "data/raw/sample_submission.csv",
    [string]$TargetCol = "Heart Disease",
    [int]$TopK = 2,
    [double]$MaxCorrelation = 0.998,
    [string]$DailyPickPath = "kaggle_outputs/cloud_latest/daily_pick.json",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
chcp 65001 > $null

function Assert-CommandExists {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $Name"
    }
}

Assert-CommandExists "python"

$kaggleCommand = "kaggle"
if (-not (Get-Command $kaggleCommand -ErrorAction SilentlyContinue)) {
    $candidatePaths = @(
        "$env:APPDATA\Python\Python310\Scripts\kaggle.exe",
        "$env:APPDATA\Python\Python311\Scripts\kaggle.exe",
        "$env:APPDATA\Python\Python312\Scripts\kaggle.exe"
    )
    $resolved = $candidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $resolved) {
        throw "Command not found: kaggle"
    }
    $kaggleCommand = $resolved
}

if (-not $SkipDownload) {
    if (-not (Test-Path $CloudOutputDir)) {
        New-Item -ItemType Directory -Path $CloudOutputDir -Force | Out-Null
    }

    Write-Host "[1/5] Downloading kernel outputs: $KernelRef -> $CloudOutputDir"
    & $kaggleCommand kernels output $KernelRef -p $CloudOutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "kaggle kernels output failed"
    }
}
else {
    Write-Host "[1/5] Skip download enabled; using existing outputs at: $CloudOutputDir"
}

$candidateScoresPath = Join-Path $CloudOutputDir "candidate_scores_cloud.csv"
if (-not (Test-Path $candidateScoresPath)) {
    throw "Missing file: $candidateScoresPath"
}
if (-not (Test-Path $SampleSubmissionPath)) {
    throw "Missing file: $SampleSubmissionPath"
}

Write-Host "[2/5] Validating submission files"
$submissionFiles = Get-ChildItem -Path $CloudOutputDir -Filter "submission_*.csv" -File | Sort-Object Name
if ($submissionFiles.Count -eq 0) {
    throw "No submission_*.csv found in $CloudOutputDir"
}

$validFileNames = [System.Collections.Generic.HashSet[string]]::new()
foreach ($file in $submissionFiles) {
    & python src/online_sprint.py validate-submission `
        --submission-path $file.FullName `
        --sample-sub-path $SampleSubmissionPath `
        --target-col $TargetCol

    if ($LASTEXITCODE -eq 0) {
        [void]$validFileNames.Add($file.Name)
    }
}

Write-Host "[3/5] Filtering candidate scores by validated files"
$candidateRows = Import-Csv -Path $candidateScoresPath
$filteredRows = @($candidateRows | Where-Object { $validFileNames.Contains($_.submission_file) })
if ($filteredRows.Count -eq 0) {
    throw "No validated candidates remain after filtering"
}

$validatedScoresPath = Join-Path $CloudOutputDir "candidate_scores_validated.csv"
$filteredRows | Export-Csv -Path $validatedScoresPath -NoTypeInformation -Encoding UTF8

Write-Host "[4/5] Selecting daily submissions"
$dailyPickDir = Split-Path -Path $DailyPickPath -Parent
if ($dailyPickDir -and -not (Test-Path $dailyPickDir)) {
    New-Item -ItemType Directory -Path $dailyPickDir -Force | Out-Null
}

& python src/online_sprint.py pick-daily `
    --candidate-scores-path $validatedScoresPath `
    --submission-dir $CloudOutputDir `
    --target-col $TargetCol `
    --top-k $TopK `
    --max-correlation $MaxCorrelation `
    --output-path $DailyPickPath

if ($LASTEXITCODE -ne 0) {
    throw "daily selection failed"
}

Write-Host "[5/5] Suggested submission commands"
$pick = Get-Content -Path $DailyPickPath -Raw | ConvertFrom-Json
foreach ($item in $pick.selected) {
    $submissionPath = Join-Path $CloudOutputDir $item.submission_file
    $message = "{0}|cv={1}|reason={2}" -f $item.candidate, ([double]$item.cv_auc).ToString("F6"), $item.selection_reason
    Write-Host ("kaggle competitions submit -c {0} -f `"{1}`" -m `"{2}`"" -f $Competition, $submissionPath, $message)
}

Write-Host "Done. Daily selection file: $DailyPickPath"
