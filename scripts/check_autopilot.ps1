# Check GLAUCO autopilot progress (run anytime)
$status = Get-Content "results\autopilot_status.json" | ConvertFrom-Json
Write-Host "Phase: $($status.phase)  Updated: $($status.updated_at)"
$status.jobs | Format-Table id, status, elapsed_s -AutoSize
if ($status.cross_evals.Count -gt 0) { $status.cross_evals | Format-Table label, status -AutoSize }
if ($status.errors.Count -gt 0) { Write-Host "Errors:" $status.errors }
