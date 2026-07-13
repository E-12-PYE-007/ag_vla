param(
    [string]$RepoRoot = "C:\Users\miahv\Documents\Capstone_Project\ag_vla",
    [string]$OutRoot = "D:\Capstone\vla_datasets\huron",
    [ValidateSet("balanced", "pilot", "full")]
    [string]$Mode = "balanced",
    [int]$MaxBagsPerFolder = 10,
    [double]$MaxTotalGB = 25.0,
    [int]$MinBagMB = 30,
    [switch]$DryRun,
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"

function Convert-WindowsPathToWsl {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -notmatch "^([A-Za-z]):\\(.*)$") {
        throw "Expected a Windows drive path, got: $Path"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $rest = $Matches[2] -replace "\\", "/"
    return "/mnt/$drive/$rest"
}

$repoWsl = Convert-WindowsPathToWsl $RepoRoot
$outWsl = Convert-WindowsPathToWsl $OutRoot
$dry = if ($DryRun) { "1" } else { "0" }

$bashCommand = @"
set -euo pipefail
cd '$repoWsl'
python3 scripts/data_processing/download_huron_balanced_subset_wsl.py \
  --out-root '$outWsl' \
  --mode '$Mode' \
  --max-bags-per-folder '$MaxBagsPerFolder' \
  --max-total-gb '$MaxTotalGB' \
  --min-bag-mb '$MinBagMB' \
  $(if ($DryRun) { "--dry-run" } else { "" })
"@

Write-Host "Running HuRoN balanced downloader through WSL ($Distro)"
Write-Host "Repo: $repoWsl"
Write-Host "Output: $outWsl"
wsl -d $Distro -- bash -lc $bashCommand
