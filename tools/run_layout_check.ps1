# Runs `--type layout-check` against every fixture in layout_fixtures/
# and aggregates pass/fail counts. Each fixture is a self-contained
# HTML file with an embedded TB-LAYOUT-EXPECT block declaring the
# expected box dimensions for elements identified by `id`.

$ErrorActionPreference = 'Continue'
$exe = "$PSScriptRoot\..\target\x86_64-pc-windows-msvc\release\tb_browser.exe"
$dir = "$PSScriptRoot\layout_fixtures"
$files = Get-ChildItem -Path $dir -Filter "*.html" | Sort-Object Name
$total_files = 0
$ok_files = 0
$fail_files = 0
$total_pass = 0
$total_fail = 0

foreach ($f in $files) {
    $total_files += 1
    Write-Host "[layout-check] $($f.Name)" -ForegroundColor Yellow
    $out = & $exe --type layout-check $f.FullName 2>&1
    $exit = $LASTEXITCODE
    foreach ($line in $out) {
        Write-Host "    $line"
    }
    # Parse the summary line: "<path> pass=N fail=N"
    $summary = $out | Where-Object { $_ -match 'pass=(\d+)\s+fail=(\d+)' }
    if ($summary) {
        $m = [regex]::Match([string]$summary, 'pass=(\d+)\s+fail=(\d+)')
        $total_pass += [int]$m.Groups[1].Value
        $total_fail += [int]$m.Groups[2].Value
    }
    if ($exit -eq 0) {
        $ok_files += 1
    } else {
        $fail_files += 1
    }
}

Write-Host ""
Write-Host "=== layout-check summary ===" -ForegroundColor Cyan
Write-Host "fixtures:     $total_files"
Write-Host "fully pass:   $ok_files"
Write-Host "any fail:     $fail_files"
Write-Host "check PASS:   $total_pass"
Write-Host "check FAIL:   $total_fail"
