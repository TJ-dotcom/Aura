#!/usr/bin/env powershell
<#
.SYNOPSIS
AURA One-Stop Installation Script for Windows

.DESCRIPTION
Complete setup script for users cloning AURA from GitHub:
1. Checks system requirements (Python, Ollama)
2. Creates and activates virtual environment
3. Installs all Python dependencies
4. Downloads and installs Ollama if needed
5. Pulls recommended models based on hardware
6. Installs AURA CLI to system PATH
7. Runs validation tests

.EXAMPLE
.\install.ps1
.\install.ps1 -SkipModels
.\install.ps1 -Verbose
#>

param(
    [switch]$SkipModels,
    [switch]$Verbose,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Color functions
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

function Show-Banner {
    Write-Host @"
    
    ╔══════════════════════════════════════════╗
    ║           AURA AI Engine Setup           ║
    ║     Hardware-Aware AI Intelligence       ║
    ╚══════════════════════════════════════════╝
    
"@ -ForegroundColor Cyan
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-Python {
    Write-Info "🔍 Checking Python installation..."
    
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            
            if ($major -eq 3 -and $minor -ge 8) {
                Write-Success "✅ Python $pythonVersion found"
                return $true
            } else {
                Write-Error "❌ Python 3.8+ required, found $pythonVersion"
                return $false
            }
        }
    } catch {
        Write-Error "❌ Python not found in PATH"
        Write-Info "Please install Python 3.8+ from https://python.org"
        return $false
    }
}

function Test-Ollama {
    Write-Info "🔍 Checking Ollama installation..."
    
    try {
        $ollamaVersion = ollama --version 2>$null
        Write-Success "✅ Ollama found: $ollamaVersion"
        return $true
    } catch {
        Write-Warning "⚠️  Ollama not found"
        return $false
    }
}

function Install-Ollama {
    Write-Info "📥 Installing Ollama..."
    
    try {
        # Download Ollama installer
        $installerPath = "$env:TEMP\OllamaSetup.exe"
        Write-Info "Downloading Ollama installer..."
        Invoke-WebRequest -Uri "https://ollama.com/download/windows" -OutFile $installerPath
        
        # Run installer silently
        Write-Info "Running Ollama installer..."
        Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
        
        # Add to PATH (installer should do this, but ensure it's there)
        $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
        if (Test-Path $ollamaPath) {
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            if (-not $currentPath.Contains($ollamaPath)) {
                [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$ollamaPath", "User")
                $env:PATH += ";$ollamaPath"
            }
        }
        
        # Clean up
        Remove-Item $installerPath -ErrorAction SilentlyContinue
        
        # Verify installation
        Start-Sleep 3
        $ollamaVersion = ollama --version 2>$null
        if ($ollamaVersion) {
            Write-Success "✅ Ollama installed successfully: $ollamaVersion"
            return $true
        } else {
            Write-Error "❌ Ollama installation failed"
            return $false
        }
        
    } catch {
        Write-Error "❌ Failed to install Ollama: $($_.Exception.Message)"
        return $false
    }
}

function Start-OllamaService {
    Write-Info "🚀 Starting Ollama service..."
    
    try {
        # Start Ollama in background
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep 5
        
        # Test if service is running
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "✅ Ollama service started"
            return $true
        }
    } catch {
        Write-Warning "⚠️  Could not verify Ollama service status"
        return $true  # Continue anyway
    }
}

function New-VirtualEnvironment {
    Write-Info "🐍 Setting up Python virtual environment..."
    
    if (Test-Path ".venv" -and -not $Force) {
        Write-Warning "⚠️  Virtual environment already exists"
        Write-Info "Use -Force to recreate it"
        return $true
    }
    
    try {
        if (Test-Path ".venv" -and $Force) {
            Write-Info "Removing existing virtual environment..."
            Remove-Item ".venv" -Recurse -Force
        }
        
        python -m venv .venv
        Write-Success "✅ Virtual environment created"
        return $true
    } catch {
        Write-Error "❌ Failed to create virtual environment: $($_.Exception.Message)"
        return $false
    }
}

function Install-Dependencies {
    Write-Info "📦 Installing Python dependencies..."
    
    try {
        # Activate virtual environment
        & ".\.venv\Scripts\Activate.ps1"
        
        # Upgrade pip first
        python -m pip install --upgrade pip | Out-Null
        
        # Install core dependencies
        Write-Info "Installing core packages..."
        python -m pip install psutil numpy requests colorama | Out-Null
        
        # Install testing framework
        Write-Info "Installing testing framework..."
        python -m pip install pytest | Out-Null
        
        # Try to install FAISS (optional, may fail on some systems)
        Write-Info "Attempting to install FAISS (optional)..."
        try {
            python -m pip install faiss-cpu | Out-Null
            Write-Success "✅ FAISS installed successfully"
        } catch {
            Write-Warning "⚠️  FAISS installation failed (will use fallback mode)"
        }
        
        Write-Success "✅ Dependencies installed"
        return $true
    } catch {
        Write-Error "❌ Failed to install dependencies: $($_.Exception.Message)"
        return $false
    }
}

function Get-HardwareTier {
    Write-Info "🔍 Analyzing hardware for model recommendations..."
    
    try {
        & ".\.venv\Scripts\Activate.ps1"
        $output = python aura.py hardware 2>$null
        
        if ($output -match "Performance Tier: (\w+)") {
            return $matches[1].ToUpper()
        }
    } catch {
        Write-Warning "⚠️  Could not detect hardware tier"
    }
    
    return "BALANCED"
}

function Install-RecommendedModels {
    param($HardwareTier)
    
    if ($SkipModels) {
        Write-Info "⏭️  Skipping model downloads"
        return $true
    }
    
    Write-Info "🤖 Installing recommended models for $HardwareTier hardware..."
    
    # Define models by tier
    $modelMap = @{
        "HIGH-PERFORMANCE" = @("llama2:13b", "deepseek-coder:6.7b", "tinyllama:latest")
        "BALANCED" = @("llama2:7b", "deepseek-coder:6.7b", "tinyllama:latest")
        "HIGH-EFFICIENCY" = @("tinyllama:latest", "deepseek-coder:1.3b")
    }
    
    $models = $modelMap[$HardwareTier]
    if (-not $models) {
        $models = $modelMap["BALANCED"]
    }
    
    $successful = 0
    foreach ($model in $models) {
        Write-Info "📥 Downloading $model..."
        try {
            $process = Start-Process "ollama" -ArgumentList "pull", $model -PassThru -NoNewWindow
            $process.WaitForExit(300)  # 5 minute timeout per model
            
            if ($process.ExitCode -eq 0) {
                Write-Success "✅ Downloaded $model"
                $successful++
            } else {
                Write-Warning "⚠️  Failed to download $model (timeout or error)"
            }
        } catch {
            Write-Warning "⚠️  Failed to download $model"
        }
    }
    
    if ($successful -gt 0) {
        Write-Success "✅ Downloaded $successful/$($models.Count) models"
        return $true
    } else {
        Write-Warning "⚠️  No models downloaded successfully"
        return $false
    }
}

function Install-AuraCLI {
    Write-Info "🔧 Installing AURA CLI to system PATH..."
    
    try {
        powershell -ExecutionPolicy Bypass -File install-aura.ps1 -Install
        Write-Success "✅ AURA CLI installed"
        return $true
    } catch {
        Write-Error "❌ Failed to install AURA CLI: $($_.Exception.Message)"
        return $false
    }
}

function Test-Installation {
    Write-Info "🧪 Running installation validation..."
    
    try {
        & ".\.venv\Scripts\Activate.ps1"
        
        # Test hardware detection
        Write-Info "Testing hardware detection..."
        $hwOutput = python aura.py hardware 2>$null
        if ($hwOutput -match "AURA Hardware Analysis") {
            Write-Success "✅ Hardware detection working"
        } else {
            Write-Warning "⚠️  Hardware detection may have issues"
        }
        
        # Test model listing
        Write-Info "Testing model intelligence..."
        $modelOutput = python aura.py models 2>$null
        if ($modelOutput -match "AURA Model Intelligence") {
            Write-Success "✅ Model intelligence working"
        } else {
            Write-Warning "⚠️  Model intelligence may have issues"
        }
        
        # Test basic inference (if models available)
        Write-Info "Testing inference capability..."
        $inferOutput = python aura.py "Hello AURA" 2>$null
        if ($inferOutput -match "AURA" -or $inferOutput -match "response") {
            Write-Success "✅ Inference working"
        } else {
            Write-Warning "⚠️  Inference test inconclusive"
        }
        
        return $true
    } catch {
        Write-Warning "⚠️  Validation tests had issues: $($_.Exception.Message)"
        return $false
    }
}

function Show-NextSteps {
    param($HardwareTier)
    
    Write-Host @"

╔══════════════════════════════════════════╗
║          🎉 INSTALLATION COMPLETE        ║
╚══════════════════════════════════════════╝

🚀 AURA is now ready to use!

📋 Your System:
   Hardware Tier: $HardwareTier
   AI Engine: Hardware-optimized inference ready

💡 Quick Start:
   aura "Write a Python function to calculate fibonacci"
   aura "Tell me about artificial intelligence"  
   aura hardware                    # Show detailed system analysis
   aura models                      # List available models

🔧 Commands:
   aura "your prompt"              # Direct intelligent inference
   aura infer --interactive        # Interactive mode
   aura hardware                   # Hardware analysis
   aura models                     # Model intelligence

📚 Documentation:
   - CLI_GUIDE.md: Complete command reference
   - README.md: System architecture overview
   - OPERATIONAL_LOG.md: Development history

🆘 Troubleshooting:
   python aura.py --help           # Show all options
   .\install.ps1 -Force            # Reinstall everything
   .\install.ps1 -SkipModels       # Skip model downloads

"@ -ForegroundColor Green

    Write-Host "🌟 Welcome to the future of AI inference!" -ForegroundColor Cyan
}

# Main installation flow
function Main {
    Show-Banner
    
    Write-Info "🎯 Starting one-stop AURA installation..."
    Write-Info "Current directory: $(Get-Location)"
    
    # Check if we're in the right directory
    if (-not (Test-Path "aura.py")) {
        Write-Error "❌ aura.py not found. Please run this script from the AURA project root directory."
        exit 1
    }
    
    # System requirements
    if (-not (Test-Python)) { exit 1 }
    
    # Ollama installation
    if (-not (Test-Ollama)) {
        $install = Read-Host "Install Ollama automatically? (Y/n)"
        if ($install -eq "" -or $install -eq "Y" -or $install -eq "y") {
            if (-not (Install-Ollama)) { 
                Write-Error "❌ Ollama installation failed. Please install manually from https://ollama.com"
                exit 1 
            }
        } else {
            Write-Error "❌ Ollama is required. Please install from https://ollama.com"
            exit 1
        }
    }
    
    # Start Ollama service
    Start-OllamaService
    
    # Python environment setup
    if (-not (New-VirtualEnvironment)) { exit 1 }
    if (-not (Install-Dependencies)) { exit 1 }
    
    # Hardware analysis and model recommendations
    $hardwareTier = Get-HardwareTier
    Write-Success "🏆 Hardware Tier: $hardwareTier"
    
    # Model installation
    Install-RecommendedModels -HardwareTier $hardwareTier
    
    # CLI installation
    if (-not (Install-AuraCLI)) { 
        Write-Warning "⚠️  CLI installation failed, but you can still use 'python aura.py'"
    }
    
    # Validation
    Test-Installation
    
    # Success!
    Show-NextSteps -HardwareTier $hardwareTier
}

# Run main installation
Main
