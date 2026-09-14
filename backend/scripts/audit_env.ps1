<#
    ETAPE 0A - AUDIT SYSTEME (Windows)

    Lancer depuis la racine du projet, dans PowerShell :

        powershell -ExecutionPolicy Bypass -File backend\scripts\audit_env.ps1

    Le rapport s'affiche a l'ecran ET est ecrit dans :
        backend\data\metrics\audit_systeme.txt

    Ce script ne modifie rien : il ne fait que lire l'etat de la machine.
#>

$ErrorActionPreference = "Continue"
$outDir  = Join-Path $PSScriptRoot "..\data\metrics"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$outFile = Join-Path $outDir "audit_systeme.txt"
$report  = New-Object System.Collections.Generic.List[string]

function Section($t) {
    $report.Add("")
    $report.Add("=== $t ===")
}

function Probe($label, $cmd) {
    # Execute une commande et capture sa sortie, sans planter si l'outil est absent.
    try {
        $out = & cmd /c "$cmd 2>&1"
        if ($LASTEXITCODE -eq 0 -and $out) { $report.Add(("{0,-22}: {1}" -f $label, ($out -join ' ').Trim())) }
        else { $report.Add(("{0,-22}: ABSENT" -f $label)) }
    } catch {
        $report.Add(("{0,-22}: ABSENT" -f $label))
    }
}

Section "IDENTIFICATION"
$report.Add("Date                  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$os = Get-CimInstance Win32_OperatingSystem
$report.Add("Systeme               : $($os.Caption) build $($os.BuildNumber)")
$report.Add("Architecture          : $($os.OSArchitecture)")

Section "OUTILS"
Probe "python --version"  "python --version"
Probe "conda --version"   "conda --version"
Probe "git --version"     "git --version"
Probe "node --version"    "node --version"
Probe "npm --version"     "npm --version"
Probe "java -version"     "java -version"
Probe "docker --version"  "docker --version"

Section "VERSIONS PYTHON INSTALLEES (py -0p)"
try {
    $py = & cmd /c "py -0p 2>&1"
    if ($LASTEXITCODE -eq 0) { $py | ForEach-Object { $report.Add($_) } }
    else { $report.Add("py launcher : ABSENT (normal si Python installe via Anaconda uniquement)") }
} catch { $report.Add("py launcher : ABSENT") }

Section "ENVIRONNEMENTS CONDA"
try {
    $envs = & cmd /c "conda env list 2>&1"
    if ($LASTEXITCODE -eq 0) { $envs | ForEach-Object { $report.Add($_) } }
    else { $report.Add("conda absent : voir correctif dans le rapport d'etape 0") }
} catch { $report.Add("conda absent") }

Section "NEO4J DESKTOP"
$neoPaths = @(
    "$env:LOCALAPPDATA\Programs\Neo4j Desktop",
    "$env:ProgramFiles\Neo4j Desktop",
    "${env:ProgramFiles(x86)}\Neo4j Desktop",
    "$env:APPDATA\Neo4j Desktop"
)
$found = $false
foreach ($p in $neoPaths) {
    if (Test-Path $p) { $report.Add("Trouve : $p"); $found = $true }
}
if (-not $found) { $report.Add("Neo4j Desktop : NON DETECTE dans les emplacements usuels") }
$neoProc = Get-Process -Name "neo4j*", "java" -ErrorAction SilentlyContinue
if ($neoProc) { $report.Add("Processus java/neo4j en cours : $($neoProc.Name -join ', ')") }

Section "PORTS NEO4J 7474 (HTTP) ET 7687 (BOLT)"
foreach ($port in 7474, 7687) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        # $pid est une variable automatique en lecture seule : utiliser un autre nom.
        $ownerPid = $conn[0].OwningProcess
        $pname = (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue).ProcessName
        $report.Add("Port $port : OCCUPE par PID $ownerPid ($pname)")
    } else {
        $report.Add("Port $port : LIBRE")
    }
}

Section "MEMOIRE ET DISQUE"
$ramTotal = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$ramFree  = [math]::Round($os.FreePhysicalMemory   / 1MB, 1)
$report.Add("RAM totale            : $ramTotal Go")
$report.Add("RAM disponible        : $ramFree Go")
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    if ($_.Free -ne $null) {
        $report.Add(("Disque {0}: libre {1} Go / total {2} Go" -f $_.Name,
            [math]::Round($_.Free/1GB,1), [math]::Round(($_.Used + $_.Free)/1GB,1)))
    }
}

Section "ACCES INTERNET"
foreach ($h in "pypi.org", "huggingface.co", "files.pythonhosted.org") {
    try {
        $r = Invoke-WebRequest -Uri "https://$h" -Method Head -TimeoutSec 12 -UseBasicParsing
        $report.Add(("{0,-26}: OK (HTTP {1})" -f $h, $r.StatusCode))
    } catch {
        $report.Add(("{0,-26}: ECHEC - {1}" -f $h, $_.Exception.Message))
    }
}

Section "CORPUS"
$pdf = Join-Path $PSScriptRoot "..\data\raw\these_vagues_froid.pdf"
if (Test-Path $pdf) {
    $mo = [math]::Round((Get-Item $pdf).Length / 1MB, 1)
    $report.Add("these_vagues_froid.pdf : PRESENT ($mo Mo)  -- attendu ~45,2 Mo")
} else {
    $report.Add("these_vagues_froid.pdf : ABSENT -> voir backend\data\raw\PLACER_LE_PDF_ICI.md")
}

$report | ForEach-Object { Write-Host $_ }
$report | Out-File -FilePath $outFile -Encoding utf8
Write-Host ""
Write-Host "Rapport ecrit dans : $outFile"
Write-Host "-> Colle son contenu dans la conversation pour que l'audit soit finalise."
