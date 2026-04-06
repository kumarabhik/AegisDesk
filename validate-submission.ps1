param(
    [Parameter(Mandatory = $true)]
    [string]$PingUrl,

    [Parameter(Mandatory = $false)]
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"
$ImageTag = "support-ops-env-validate"

function Invoke-StrictExternal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    $command = Get-Command $FilePath -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Required command not found on PATH: $FilePath"
    }

    & $command.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code ${LASTEXITCODE}: {0} {1}" -f $FilePath, ($Arguments -join ' '))
    }
}

Write-Host "[validate] checking root endpoint: $PingUrl"
Invoke-RestMethod -Uri $PingUrl -Method Get | Out-Null

Write-Host "[validate] checking reset endpoint"
Invoke-RestMethod -Uri ($PingUrl.TrimEnd("/") + "/reset") -Method Post -ContentType "application/json" -Body '{"task_id":"billing_seat_adjustment","seed":1}' | Out-Null

Push-Location $RepoDir
try {
    Write-Host "[validate] running openenv validate"
    Invoke-StrictExternal -FilePath "openenv" -Arguments @("validate")

    Write-Host "[validate] building docker image"
    Invoke-StrictExternal -FilePath "docker" -Arguments @("build", "-t", $ImageTag, ".")
}
finally {
    Pop-Location
}

Write-Host "[validate] all checks passed"
