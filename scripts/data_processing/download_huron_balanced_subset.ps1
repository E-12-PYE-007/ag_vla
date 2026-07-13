param(
    [string]$OutRoot = "D:\Capstone\vla_datasets\huron",
    [ValidateSet("balanced", "pilot", "full")]
    [string]$Mode = "balanced",
    [int]$MaxBagsPerFolder = 10,
    [double]$MaxTotalGB = 25.0,
    [int]$MinBagMB = 30,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$BaseUrl = "https://rail.eecs.berkeley.edu/datasets/huron"
$FallbackBaseUrl = "http://rail.eecs.berkeley.edu/datasets/huron"

# Balanced default: paired standard/intloss folders across several environments.
# This gives useful policy/environment diversity without pulling the full 75-hour
# release. Increase MaxBagsPerFolder or MaxTotalGB after validating conversion.
$BalancedFolders = @(
    "Dec-06-2022-bww8",
    "Dec-09-2022-bww8",
    "Feb-03-2023-bww8-intloss",
    "Feb-06-2023-bww8-intloss",
    "Feb-15-2023-bww1",
    "Feb-16-2023-bww1-intloss",
    "Feb-15-2023-cory1",
    "Feb-16-2023-cory1-intloss",
    "Feb-17-2023-bww2",
    "Feb-20-2023-bww2-intloss",
    "Feb-17-2023-soda3",
    "Feb-23-2023-soda3-intloss"
)

$PilotFolders = @(
    "Feb-15-2023-cory1",
    "Feb-16-2023-cory1-intloss"
)

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Convert-SizeToBytes {
    param([string]$Text)
    $value = $Text.Trim()
    if (-not $value -or $value -eq "-") { return $null }
    if ($value -match "^([0-9]+(?:\.[0-9]+)?)([KMG])$") {
        $number = [double]$Matches[1]
        switch ($Matches[2]) {
            "K" { return [int64]($number * 1KB) }
            "M" { return [int64]($number * 1MB) }
            "G" { return [int64]($number * 1GB) }
        }
    }
    return $null
}

function Get-UrlContent {
    param([string]$Url)
    $content = & curl.exe --noproxy "*" --ssl-no-revoke -L --retry 3 --silent --show-error $Url
    if ($LASTEXITCODE -ne 0) {
        $fallback = $Url -replace "^https://", "http://"
        Write-Warning "HTTPS index read failed; trying HTTP fallback: $fallback"
        $content = & curl.exe --noproxy "*" -L --retry 3 --silent --show-error $fallback
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed while reading index: $Url"
        }
    }
    return ($content -join "`n")
}

function Get-DirectoryLinks {
    param([string]$Url)
    $html = Get-UrlContent $Url
    $matches = [regex]::Matches($html, 'href="([^"]+/)"')
    $folders = @()
    foreach ($match in $matches) {
        $href = $match.Groups[1].Value.TrimEnd("/")
        if ($href -and $href -ne ".." -and $href -ne "Parent Directory") {
            $folders += [System.Uri]::UnescapeDataString($href)
        }
    }
    return $folders | Sort-Object -Unique
}

function Get-BagFiles {
    param(
        [string]$Folder,
        [string]$Url
    )
    $html = Get-UrlContent $Url
    $pattern = 'href="([^"]+\.bag)"\s*>([^<]+)</a>\s+([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})\s+([0-9.]+[KMG])'
    $matches = [regex]::Matches($html, $pattern)
    $bags = @()
    foreach ($match in $matches) {
        $name = [System.Uri]::UnescapeDataString($match.Groups[1].Value)
        $sizeText = $match.Groups[4].Value
        $sizeBytes = Convert-SizeToBytes $sizeText
        if ($null -eq $sizeBytes) { continue }
        $bags += [pscustomobject]@{
            Folder = $Folder
            Name = $name
            Url = "$Url/$name"
            SizeText = $sizeText
            SizeBytes = $sizeBytes
            InteractionLoss = $Folder -like "*intloss*"
            LocalPath = Join-Path $OutRoot ("raw\{0}\{1}" -f $Folder, $name)
        }
    }
    return $bags | Sort-Object Name
}

function Test-NotHtml {
    param([string]$Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $take = [Math]::Min(512, $bytes.Length)
    $prefix = [System.Text.Encoding]::ASCII.GetString($bytes, 0, $take).ToLowerInvariant()
    if ($prefix -match "<html|<!doctype html|not found|access denied|forbidden|login|sign in") {
        throw "Downloaded file looks like an HTML/error page: $Path"
    }
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Download-Bag {
    param([pscustomobject]$Bag)
    $dest = $Bag.LocalPath
    $part = "$dest.part"
    $destDir = Split-Path -Parent $dest
    Ensure-Directory $destDir

    if ((Test-Path -LiteralPath $dest) -and -not $Force) {
        $existingSize = (Get-Item -LiteralPath $dest).Length
        if ($existingSize -eq $Bag.SizeBytes) {
            Test-NotHtml $dest
            Write-Host "Already complete: $dest"
            return
        }
        Write-Host "Existing file size differs; resuming into .part: $dest"
        Move-Item -Force -LiteralPath $dest -Destination $part
    }

    Write-Host "Downloading $($Bag.Folder)/$($Bag.Name) [$($Bag.SizeText)]"
    curl.exe --noproxy "*" -L -C - -o $part $Bag.Url
    Test-NotHtml $part
    Move-Item -Force -LiteralPath $part -Destination $dest

    $actualSize = (Get-Item -LiteralPath $dest).Length
    if ($actualSize -ne $Bag.SizeBytes) {
        Write-Warning "Size mismatch for $dest expected=$($Bag.SizeBytes) actual=$actualSize"
    }
}

Ensure-Directory $OutRoot
Ensure-Directory (Join-Path $OutRoot "raw")
Ensure-Directory (Join-Path $OutRoot "logs")
Ensure-Directory (Join-Path $OutRoot "manifests")

$driveName = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $OutRoot).Path)
$driveInfo = [System.IO.DriveInfo]::new($driveName)
$freeGB = [Math]::Round($driveInfo.AvailableFreeSpace / 1GB, 2)
Write-Host "HuRoN output root: $OutRoot"
Write-Host "Free space on $driveName ${freeGB} GB"

if ($freeGB -lt ($MaxTotalGB + 5)) {
    throw "Not enough free space for requested cap. Need at least $($MaxTotalGB + 5) GB free."
}

if ($Mode -eq "full") {
    $folders = Get-DirectoryLinks $BaseUrl
} elseif ($Mode -eq "pilot") {
    $folders = $PilotFolders
} else {
    $folders = $BalancedFolders
}

Write-Host "Mode: $Mode"
Write-Host "Folders:"
$folders | ForEach-Object { Write-Host "  $_" }

$allBags = @()
foreach ($folder in $folders) {
    $url = "$BaseUrl/$folder"
    Write-Host "Indexing $url"
    $bags = Get-BagFiles -Folder $folder -Url $url
    $bags = $bags | Where-Object { $_.SizeBytes -ge ($MinBagMB * 1MB) }
    if ($MaxBagsPerFolder -gt 0) {
        $bags = $bags | Select-Object -First $MaxBagsPerFolder
    }
    $allBags += $bags
}

$selected = @()
$runningBytes = [int64]0
$maxBytes = [int64]($MaxTotalGB * 1GB)
foreach ($bag in $allBags) {
    if (($runningBytes + $bag.SizeBytes) -gt $maxBytes) {
        Write-Host "Skipping due to MaxTotalGB cap: $($bag.Folder)/$($bag.Name)"
        continue
    }
    $selected += $bag
    $runningBytes += $bag.SizeBytes
}

$planCsv = Join-Path $OutRoot "manifests\huron-balanced-download-plan.csv"
$selected | Export-Csv -NoTypeInformation -Path $planCsv

$planJson = Join-Path $OutRoot "manifests\huron-balanced-download-plan.json"
$plan = [pscustomobject]@{
    created_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    base_url = $BaseUrl
    mode = $Mode
    max_bags_per_folder = $MaxBagsPerFolder
    max_total_gb = $MaxTotalGB
    min_bag_mb = $MinBagMB
    selected_count = $selected.Count
    selected_size_gb = [Math]::Round($runningBytes / 1GB, 3)
    folders = $folders
    bags = $selected
}
$plan | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $planJson

Write-Host ""
Write-Host "Selected $($selected.Count) bags, estimated size $([Math]::Round($runningBytes / 1GB, 2)) GB"
Write-Host "Plan CSV: $planCsv"
Write-Host "Plan JSON: $planJson"

if ($DryRun) {
    Write-Host "Dry run only. No files downloaded."
    exit 0
}

$shaPath = Join-Path $OutRoot "manifests\huron-balanced-sha256.txt"
foreach ($bag in $selected) {
    Download-Bag $bag
    $sha = Get-Sha256 $bag.LocalPath
    "$sha  $($bag.LocalPath)" | Add-Content -Encoding ASCII -Path $shaPath
}

$downloaded = foreach ($bag in $selected) {
    $path = $bag.LocalPath
    if (Test-Path -LiteralPath $path) {
        [pscustomobject]@{
            folder = $bag.Folder
            name = $bag.Name
            url = $bag.Url
            local_path = $path
            size_bytes = (Get-Item -LiteralPath $path).Length
            sha256 = Get-Sha256 $path
            interaction_loss = $bag.InteractionLoss
        }
    }
}

$manifestPath = Join-Path $OutRoot "manifests\huron-balanced-download-manifest.json"
[pscustomobject]@{
    created_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    base_url = $BaseUrl
    mode = $Mode
    downloaded_count = $downloaded.Count
    downloaded_size_gb = [Math]::Round((($downloaded | Measure-Object -Property size_bytes -Sum).Sum) / 1GB, 3)
    items = $downloaded
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $manifestPath

Write-Host ""
Write-Host "Done."
Write-Host "Manifest: $manifestPath"
Write-Host "SHA-256: $shaPath"
