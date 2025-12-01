# AURA Installation Script for Windows
# Adds AURA CLI to system PATH so you can use 'aura' command anywhere

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Check
)

$AURA_DIR = $PSScriptRoot
$AURA_BAT = Join-Path $AURA_DIR "aura.bat"

function Install-AuraCommand {
    Write-Host "Installing AURA CLI..." -ForegroundColor Green
    
    # Check if aura.bat exists
    if (-not (Test-Path $AURA_BAT)) {
        Write-Host "Error: aura.bat not found in $AURA_DIR" -ForegroundColor Red
        return $false
    }
    
    # Get current PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    
    # Check if already in PATH
    if ($currentPath.Split(";") -contains $AURA_DIR) {
        Write-Host "AURA is already installed in PATH" -ForegroundColor Yellow
        return $true
    }
    
    # Add to PATH
    $newPath = $currentPath + ";" + $AURA_DIR
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    
    Write-Host "AURA CLI installed successfully!" -ForegroundColor Green
    Write-Host "Added to PATH: $AURA_DIR" -ForegroundColor Cyan
    Write-Host "Please restart your terminal to use 'aura' command" -ForegroundColor Yellow
    
    return $true
}

function Remove-AuraCommand {
    Write-Host "Uninstalling AURA CLI..." -ForegroundColor Yellow
    
    # Get current PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    
    # Remove AURA directory from PATH
    $pathArray = $currentPath.Split(";") | Where-Object { $_ -ne $AURA_DIR }
    $newPath = $pathArray -join ";"
    
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    
    Write-Host "AURA CLI uninstalled successfully!" -ForegroundColor Green
    Write-Host "Please restart your terminal" -ForegroundColor Yellow
}

function Test-AuraInstallation {
    Write-Host "Checking AURA installation..." -ForegroundColor Cyan
    
    # Check if aura.bat exists
    if (Test-Path $AURA_BAT) {
        Write-Host "aura.bat found: $AURA_BAT" -ForegroundColor Green
    } else {
        Write-Host "aura.bat not found: $AURA_BAT" -ForegroundColor Red
        return $false
    }
    
    # Check if aura.py exists
    $auraPy = Join-Path $AURA_DIR "aura.py"
    if (Test-Path $auraPy) {
        Write-Host "aura.py found: $auraPy" -ForegroundColor Green
    } else {
        Write-Host "aura.py not found: $auraPy" -ForegroundColor Red
        return $false
    }
    
    # Check PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath.Split(";") -contains $AURA_DIR) {
        Write-Host "AURA directory is in PATH" -ForegroundColor Green
    } else {
        Write-Host "AURA directory not in PATH" -ForegroundColor Red
        return $false
    }
    
    # Check virtual environment
    $venvPython = Join-Path $AURA_DIR ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Host "Virtual environment found" -ForegroundColor Green
    } else {
        Write-Host "Virtual environment not found at .venv\Scripts\python.exe" -ForegroundColor Red
        return $false
    }
    
    Write-Host "AURA installation looks good!" -ForegroundColor Green
    return $true
}

function Show-Usage {
    Write-Host "AURA CLI Installer" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor White
    Write-Host "  .\install-aura.ps1 -Install    # Install AURA CLI to PATH"
    Write-Host "  .\install-aura.ps1 -Uninstall  # Remove AURA CLI from PATH"
    Write-Host "  .\install-aura.ps1 -Check      # Check installation status"
    Write-Host ""
    Write-Host "After installation, you can use these commands:"
    Write-Host "  aura run deepseek-coder:6.7b   # Run a model interactively"
    Write-Host "  aura list                       # List available models"
    Write-Host "  aura pull llama2:7b             # Download a model"
    Write-Host "  aura show tinyllama             # Show model info"
    Write-Host "  aura ps                         # List running models"
    Write-Host ""
}

# Main execution
if ($Install) {
    Install-AuraCommand
} elseif ($Uninstall) {
    Remove-AuraCommand
} elseif ($Check) {
    Test-AuraInstallation
} else {
    Show-Usage
}
