# One-time, recoverable relocation of historical material. No files are deleted.
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repoPrefix = $repoRoot.TrimEnd('\') + '\'
$moves = [Collections.Generic.List[object]]::new()
function Add-Relocations($names, $folder) {
    foreach ($name in $names) {
        $moves.Add([pscustomobject]@{from=$name; to=($folder + '/' + [IO.Path]::GetFileName($name))})
    }
}
Add-Relocations @('grouped_probe_results.json', 'intervention_paired_reanalysis.json') 'results'
Add-Relocations @('mesoscale_results.json', 'confirmation_results.json', 'prefill_all_results.json') 'results/interventions'
Add-Relocations @(
    'capture_runs_ews.py', 'colab_setup.py', 'setup_pod.sh',
    'run_sdq_commitment.py', 'run_sdq_commitment_map.py', 'run_sdq_commitment_intervention.py',
    'run_sdq_subspace_intervention.py', 'run_sdq_mode_intervention.py',
    'run_sdq_mesoscale_intervention.py', 'run_sdq_cross_model.py', 'run_sdq_decodability.py',
    'run_sdq_ews.py', 'run_sdq_ews_probe.py', 'run_sdq_ews_t0_ablation.py',
    'commitment_results_updated.json', 'commitment_map_results.json',
    'ews_results.json', 'ews_results2.json', 'ews_probe_results.json',
    'ews_probe_results2.json', 'ews_t0_ablation_results.json', 'commitment_direction.pt'
) 'archive/legacy_experiments'
Add-Relocations @('ews_model.pt', 'ews_model2.pt', 'sdq_v5_checkpoint_9.pt', 'sdq_v5_1_results_colab3_model.safetensors') 'archive/local_checkpoints'
Add-Relocations @('sdq_refactor.md', 'sdq_writeup.md', 'sdq_writeup_astra.md', 'SDQ.md', 'SESSION_CONTEXT.md',
    'docs/preprint_plan.md', '.cursor/plans/sdq_early-warning_refactor_7a82b709.plan.md') 'archive/project_notes'
Add-Relocations @('docs/assumptions.md', 'docs/certification.md', 'docs/experimental_protocol.md', 'docs/sdq_formalism.md') 'archive/legacy_docs'
$moves.Add([pscustomobject]@{from='visualizations'; to='archive/visualizations'})
$manifestPath = Join-Path $PSScriptRoot 'relocation_manifest.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'Relocation manifest already exists; this is a one-time migration.' }

# Check every resolved target before performing any move, and refuse overwrites.
foreach ($move in $moves) {
    $source = [IO.Path]::GetFullPath((Join-Path $repoRoot $move.from))
    $dest = [IO.Path]::GetFullPath((Join-Path $repoRoot $move.to))
    if (-not $source.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $dest.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes workspace.' }
    if (-not (Test-Path -LiteralPath $source) -and -not (Test-Path -LiteralPath $dest)) { throw "Missing source and destination: $source" }
    if ((Test-Path -LiteralPath $source) -and (Test-Path -LiteralPath $dest)) { throw "Source and destination both exist: $dest" }
}
$manifest = [Collections.Generic.List[object]]::new()
foreach ($move in $moves) {
    $source = [IO.Path]::GetFullPath((Join-Path $repoRoot $move.from))
    $dest = [IO.Path]::GetFullPath((Join-Path $repoRoot $move.to))
    $alreadyMoved = -not (Test-Path -LiteralPath $source)
    $readPath = if ($alreadyMoved) { $dest } else { $source }
    $item = Get-Item -LiteralPath $readPath
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $readPath -File -Recurse) } else { @($item) }
    foreach ($file in $files) {
        $suffix = if ($item.PSIsContainer) { '/' + $file.FullName.Substring($readPath.Length + 1).Replace('\', '/') } else { '' }
        $manifest.Add([pscustomobject]@{
            old_path=($move.from + $suffix); new_path=($move.to + $suffix)
            sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
            resumed_after_move=$alreadyMoved
        })
    }
    if (-not $alreadyMoved) {
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($dest)) -Force | Out-Null
        Move-Item -LiteralPath $source -Destination $dest
    }
}
foreach ($entry in $manifest) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $repoRoot $entry.new_path) -Algorithm SHA256).Hash
    if ($actual -ne $entry.sha256) { throw "Hash mismatch: $($entry.new_path)" }
}
# Generated provenance, not a modification of any source result.
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output "Relocated and hash-verified $($manifest.Count) files; originals remain recoverable at their new paths."
