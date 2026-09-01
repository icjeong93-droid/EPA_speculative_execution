# Transfer the offline bundle and the SpokenWOZ dataset to the HPC from Windows.
#
# Windows has no rsync (and Git Bash does not ship one), so transfer.sh cannot run
# here. This uses only what Windows 10+ provides out of the box: OpenSSH scp/ssh.
#
# Resumable by design: before sending a directory it lists what is already on the
# remote and sends only the missing files. Re-run after any interruption.
#
#   .\scripts\transfer.ps1 -Dest user@hpc:/scratch/$env:USERNAME/EPA_incoming
#   .\scripts\transfer.ps1 -Dest user@hpc:/path -DataOnly
#   .\scripts\transfer.ps1 -Dest user@hpc:/path -BundleOnly
#   .\scripts\transfer.ps1 -Dest user@hpc:/scratch/$env:USERNAME/EPA_incoming `
#        -DataDest /home/sr5/SR_AISolution_ACU/database/EPA
#
# -DataDest puts SpokenWOZ/ somewhere other than <remote>/data -- use it when the
# dataset lives in a shared database area rather than next to the working copy.
#
# If the HPC needs a jump host or a key, configure it once in ~/.ssh/config and
# refer to the host by its alias -- this script just calls ssh/scp.

param(
    [Parameter(Mandatory = $true)][string]$Dest,   # user@host:/remote/path
    [switch]$DataOnly,
    [switch]$BundleOnly,
    [string]$DataDest = "",                        # remote dir to hold SpokenWOZ/
                                                   # default: <remote>/data

    [int]$BatchSize = 150                          # files per scp call
)

$ErrorActionPreference = "Stop"

if ($Dest -notmatch '^([^:]+):(.+)$') {
    throw "Dest must look like user@host:/remote/path  (got: $Dest)"
}
$RemoteHost = $Matches[1]
$RemoteRoot = $Matches[2].TrimEnd('/')
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "host   : $RemoteHost"
Write-Host "remote : $RemoteRoot"
$DataRoot = if ($DataDest) { $DataDest.TrimEnd('/') } else { "$RemoteRoot/data" }

Write-Host "local  : $Root"
Write-Host "data   : $DataRoot/SpokenWOZ`n"

function Invoke-Remote([string]$Cmd) {
    $out = & ssh $RemoteHost $Cmd 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ssh failed: $Cmd`n$out" }
    return $out
}

function Send-Directory([string]$LocalDir, [string]$RemoteDir, [string]$Label) {
    if (-not (Test-Path $LocalDir)) {
        Write-Host "  skip $Label (not present locally)"
        return
    }
    $local = Get-ChildItem -File $LocalDir
    if ($local.Count -eq 0) { Write-Host "  skip $Label (empty)"; return }

    Invoke-Remote "mkdir -p '$RemoteDir'" | Out-Null
    # one listing instead of one stat per file -- 4700 round trips would be glacial
    $remote = @{}
    foreach ($n in (Invoke-Remote "ls -1 '$RemoteDir' 2>/dev/null")) {
        if ("$n".Trim()) { $remote["$n".Trim()] = $true }
    }
    $todo = @($local | Where-Object { -not $remote.ContainsKey($_.Name) })

    $haveGB = [math]::Round((($local | Where-Object { $remote.ContainsKey($_.Name) } |
                Measure-Object Length -Sum).Sum / 1GB), 2)
    $todoGB = [math]::Round((($todo | Measure-Object Length -Sum).Sum / 1GB), 2)
    Write-Host ("  {0}: {1} files, {2} already there, sending {3} ({4} GB)" -f `
            $Label, $local.Count, ($local.Count - $todo.Count), $todo.Count, $todoGB)
    if ($todo.Count -eq 0) { return }

    $sent = 0
    for ($i = 0; $i -lt $todo.Count; $i += $BatchSize) {
        $batch = $todo[$i..([math]::Min($i + $BatchSize - 1, $todo.Count - 1))]
        # scp takes many sources at once; batching keeps the command line sane
        & scp -q @($batch.FullName) "${RemoteHost}:${RemoteDir}/"
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed at file $($i + 1)/$($todo.Count). Re-run this script -- it resumes."
        }
        $sent += $batch.Count
        Write-Progress -Activity $Label -Status "$sent / $($todo.Count)" `
            -PercentComplete ($sent * 100 / $todo.Count)
    }
    Write-Progress -Activity $Label -Completed
    Write-Host "    done ($sent files)"
}

# --- connectivity ---------------------------------------------------------
Write-Host "checking ssh ..."
Invoke-Remote "echo ok" | Out-Null
Invoke-Remote "mkdir -p '$RemoteRoot'" | Out-Null
Write-Host "  ok`n"

# --- bundle ---------------------------------------------------------------
if (-not $DataOnly) {
    $b = Join-Path $Root "offline_bundle"
    if (-not (Test-Path $b)) {
        throw "offline_bundle not found. Build it first:`n  python scripts\build_offline_bundle.py --root . --out offline_bundle"
    }
    Write-Host "== bundle =="
    Send-Directory (Join-Path $b "wheels") "$RemoteRoot/bundle/wheels" "wheels"
    # hf_cache and repo are nested trees; scp -r handles them and is cheap enough
    foreach ($sub in "hf_cache", "repo", "epa") {
        $p = Join-Path $b $sub
        if (Test-Path $p) {
            Write-Host "  $sub ..."
            Invoke-Remote "mkdir -p '$RemoteRoot/bundle'" | Out-Null
            & scp -q -r $p "${RemoteHost}:${RemoteRoot}/bundle/"
            if ($LASTEXITCODE -ne 0) { throw "scp -r $sub failed" }
        }
    }
    foreach ($f in "MANIFEST.txt", "setup_offline.sh", "repo_local_changes.patch") {
        $p = Join-Path $b $f
        if (Test-Path $p) { & scp -q $p "${RemoteHost}:${RemoteRoot}/bundle/" }
    }
    Write-Host ""
}

# --- dataset --------------------------------------------------------------
if (-not $BundleOnly) {
    $d = Join-Path $Root "data\SpokenWOZ"
    if (-not (Test-Path $d)) {
        throw "data\SpokenWOZ not found. Download it first:`n  python scripts\download_spokenwoz.py --root ."
    }
    Write-Host "== dataset (~29 GB) =="
    Send-Directory (Join-Path $d "text_5700_train_dev") "$DataRoot/SpokenWOZ/text_5700_train_dev" "text_train_dev"
    Send-Directory (Join-Path $d "text_5700_test")      "$DataRoot/SpokenWOZ/text_5700_test"      "text_test"
    Send-Directory (Join-Path $d "audio_5700_train_dev") "$DataRoot/SpokenWOZ/audio_5700_train_dev" "audio_train_dev"
    Send-Directory (Join-Path $d "audio_5700_test")      "$DataRoot/SpokenWOZ/audio_5700_test"      "audio_test"
    Write-Host ""
}

# --- verify ---------------------------------------------------------------
Write-Host "== remote check =="
$checks = @{
    "audio_5700_train_dev" = 4700
    "audio_5700_test"      = 1000
}
$bad = $false
foreach ($k in $checks.Keys) {
    $n = [int](Invoke-Remote "ls -1 '$DataRoot/SpokenWOZ/$k' 2>/dev/null | wc -l")
    $okk = ($n -eq $checks[$k])
    if (-not $okk) { $bad = $true }
    Write-Host ("  [{0}] {1}: {2} (expected {3})" -f $(if ($okk) { "OK " } else { "BAD" }), $k, $n, $checks[$k])
}
$nw = [int](Invoke-Remote "ls -1 '$RemoteRoot/bundle/wheels' 2>/dev/null | wc -l")
Write-Host ("  [{0}] bundle/wheels: {1} wheels" -f $(if ($nw -ge 80) { "OK " } else { "BAD" }), $nw)

Write-Host ""
if ($bad) {
    Write-Host "INCOMPLETE - re-run this script, it sends only what is missing." -ForegroundColor Yellow
    exit 1
}
Write-Host "transferred. next, on the HPC:" -ForegroundColor Green
Write-Host "  cd $RemoteRoot/bundle && bash setup_offline.sh /scratch/`$USER/EPA"
Write-Host "  bash /scratch/`$USER/EPA/scripts/preflight.sh /scratch/`$USER/EPA"
