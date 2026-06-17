param(
    [string]$DatasetDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "datasets"),
    [string]$HfEndpoint = "https://hf-mirror.com",
    [switch]$InstallMissingTools,
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Require-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (Test-Command $Name) {
        return
    }

    if ($InstallMissingTools) {
        Write-Host "Installing missing tool for command: $Name"
        Invoke-Expression $InstallHint
        if (Test-Command $Name) {
            return
        }
    }

    throw "Command '$Name' was not found. Install it first: $InstallHint"
}

$DatasetDir = [System.IO.Path]::GetFullPath($DatasetDir)
New-Item -ItemType Directory -Force -Path $DatasetDir | Out-Null

$env:HF_ENDPOINT = $HfEndpoint

Require-Command -Name "modelscope" -InstallHint "python -m pip install modelscope"
Require-Command -Name "huggingface-cli" -InstallHint "python -m pip install huggingface_hub"
Require-Command -Name "tar" -InstallHint "Use Windows 10/11 built-in tar, or install tar with your package manager."

Write-Host "Dataset directory: $DatasetDir"
Write-Host "HF_ENDPOINT: $env:HF_ENDPOINT"

$seqMonkeyArchive = Join-Path $DatasetDir "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2"

Write-Host "`n[1/3] Downloading seq-monkey pretrain corpus..."
modelscope download `
    --dataset ddzhu123/seq-monkey `
    mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 `
    --local_dir "$DatasetDir"

if (-not $SkipExtract) {
    Write-Host "`n[2/3] Extracting seq-monkey archive..."
    tar -xvf "$seqMonkeyArchive" -C "$DatasetDir"
}
else {
    Write-Host "`n[2/3] Skipping extraction because -SkipExtract was provided."
}

$belleDir = Join-Path $DatasetDir "BelleGroup"

Write-Host "`n[3/3] Downloading BelleGroup/train_3.5M_CN SFT corpus..."
huggingface-cli download `
    --repo-type dataset `
    --resume-download `
    BelleGroup/train_3.5M_CN `
    --local-dir "$belleDir"

Write-Host "`nDone."
Write-Host "Downloaded source datasets to: $DatasetDir"
Write-Host ""
Write-Host "Next step: convert source files into training jsonl files, for example:"
Write-Host "  seq_monkey_datawhale.jsonl"
Write-Host "  BelleGroup_sft.jsonl"
