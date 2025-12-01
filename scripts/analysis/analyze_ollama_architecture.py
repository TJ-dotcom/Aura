"""
Comprehensive Ollama CPU vs GPU Usage Analysis
Determines why Ollama uses CPU despite GPU configuration
"""

import subprocess
import time
import json
import sys

def run_command(cmd, shell=True):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=shell, timeout=30)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 1
    except Exception as e:
        return "", str(e), 1

def get_gpu_info():
    """Get detailed GPU information."""
    stdout, stderr, code = run_command('nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits')
    if code == 0:
        parts = stdout.split(', ')
        return {
            'name': parts[0] if len(parts) > 0 else 'Unknown',
            'memory_used': int(parts[1]) if len(parts) > 1 else 0,
            'memory_total': int(parts[2]) if len(parts) > 2 else 0,
            'utilization': int(parts[3]) if len(parts) > 3 else 0,
            'temperature': int(parts[4]) if len(parts) > 4 else 0
        }
    return None

def get_cpu_usage():
    """Get CPU usage percentage."""
    stdout, stderr, code = run_command('powershell -Command "(Get-Counter \\"\\Processor(_Total)\\% Processor Time\\").CounterSamples.CookedValue"')
    if code == 0:
        try:
            return float(stdout)
        except:
            pass
    return 0.0

def get_ollama_processes():
    """Get Ollama process information."""
    stdout, stderr, code = run_command('powershell -Command "Get-Process ollama -ErrorAction SilentlyContinue | ConvertTo-Json"')
    if code == 0 and stdout:
        try:
            processes = json.loads(stdout)
            if isinstance(processes, list):
                return processes
            else:
                return [processes]
        except json.JSONDecodeError:
            pass
    return []

def test_ollama_direct():
    """Test direct Ollama usage."""
    print("🔍 Testing Direct Ollama Usage...")
    
    cpu_before = get_cpu_usage()
    gpu_before = get_gpu_info()
    
    start_time = time.time()
    stdout, stderr, code = run_command('ollama run deepseek-coder:6.7b "print(\\"test\\")"')
    end_time = time.time()
    
    cpu_after = get_cpu_usage()
    gpu_after = get_gpu_info()
    
    duration = end_time - start_time
    
    print(f"   Duration: {duration:.1f}s")
    print(f"   CPU: {cpu_before:.1f}% → {cpu_after:.1f}% (Δ{cpu_after-cpu_before:+.1f}%)")
    if gpu_before and gpu_after:
        print(f"   GPU Util: {gpu_before['utilization']}% → {gpu_after['utilization']}%")
        print(f"   GPU Mem: {gpu_before['memory_used']}MB → {gpu_after['memory_used']}MB")
    
    return {
        'duration': duration,
        'cpu_change': cpu_after - cpu_before,
        'gpu_util_change': (gpu_after['utilization'] - gpu_before['utilization']) if gpu_before and gpu_after else 0
    }

def test_aura_middleware():
    """Test AURA (with Ollama middleware)."""
    print("🔍 Testing AURA Middleware Usage...")
    
    cpu_before = get_cpu_usage()
    gpu_before = get_gpu_info()
    
    start_time = time.time()
    stdout, stderr, code = run_command('python aura.py "print(\\"test\\")"')
    end_time = time.time()
    
    cpu_after = get_cpu_usage()
    gpu_after = get_gpu_info()
    
    duration = end_time - start_time
    
    print(f"   Duration: {duration:.1f}s")
    print(f"   CPU: {cpu_before:.1f}% → {cpu_after:.1f}% (Δ{cpu_after-cpu_before:+.1f}%)")
    if gpu_before and gpu_after:
        print(f"   GPU Util: {gpu_before['utilization']}% → {gpu_after['utilization']}%")
        print(f"   GPU Mem: {gpu_before['memory_used']}MB → {gpu_after['memory_used']}MB")
    
    return {
        'duration': duration,
        'cpu_change': cpu_after - cpu_before,
        'gpu_util_change': (gpu_after['utilization'] - gpu_before['utilization']) if gpu_before and gpu_after else 0
    }

def analyze_ollama_architecture():
    """Analyze Ollama's architecture and configuration."""
    print("🏗️ Ollama Architecture Analysis...")
    
    # Check Ollama version
    stdout, stderr, code = run_command('ollama --version')
    if code == 0:
        print(f"   Version: {stdout}")
    
    # Check available models
    stdout, stderr, code = run_command('ollama list')
    if code == 0:
        print("   Available Models:")
        for line in stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                print(f"     {line}")
    
    # Check running models
    stdout, stderr, code = run_command('ollama ps')
    if code == 0:
        print("   Running Models:")
        for line in stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                print(f"     {line}")
    
    # Check environment variables
    env_vars = ['OLLAMA_NUM_GPU', 'OLLAMA_GPU_LAYERS', 'CUDA_VISIBLE_DEVICES']
    print("   Environment Variables:")
    for var in env_vars:
        stdout, stderr, code = run_command(f'powershell -Command "$env:{var}"')
        value = stdout if stdout else "Not Set"
        print(f"     {var}: {value}")

def main():
    """Run comprehensive analysis."""
    print("🎯 Comprehensive Ollama CPU vs GPU Analysis")
    print("=" * 60)
    
    # System information
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"🎮 GPU: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['memory_used']}/{gpu_info['memory_total']}MB")
        print(f"   Utilization: {gpu_info['utilization']}%")
        print(f"   Temperature: {gpu_info['temperature']}°C")
    else:
        print("❌ Could not get GPU information")
    
    print(f"🖥️ CPU Baseline: {get_cpu_usage():.1f}%")
    
    # Ollama process analysis
    processes = get_ollama_processes()
    if processes:
        print(f"🔧 Ollama Processes: {len(processes)}")
        for proc in processes:
            cpu = proc.get('CPU', 0)
            memory = proc.get('WorkingSet', 0) // (1024*1024)  # Convert to MB
            print(f"   CPU: {cpu} units, Memory: {memory}MB")
    
    print("\n" + "=" * 60)
    
    # Architecture analysis
    analyze_ollama_architecture()
    
    print("\n" + "=" * 60)
    
    # Performance tests
    try:
        direct_results = test_ollama_direct()
        time.sleep(3)  # Brief pause between tests
        aura_results = test_aura_middleware()
        
        print("\n📊 COMPARISON ANALYSIS")
        print("=" * 60)
        
        print(f"Direct Ollama:")
        print(f"   CPU Change: {direct_results['cpu_change']:+.1f}%")
        print(f"   Duration: {direct_results['duration']:.1f}s")
        print(f"   GPU Util Change: {direct_results['gpu_util_change']:+d}%")
        
        print(f"AURA Middleware:")
        print(f"   CPU Change: {aura_results['cpu_change']:+.1f}%")
        print(f"   Duration: {aura_results['duration']:.1f}s")
        print(f"   GPU Util Change: {aura_results['gpu_util_change']:+d}%")
        
        # Calculate middleware overhead
        cpu_overhead = aura_results['cpu_change'] - direct_results['cpu_change']
        duration_overhead = aura_results['duration'] - direct_results['duration']
        
        print(f"\n🔍 MIDDLEWARE OVERHEAD:")
        print(f"   Additional CPU: {cpu_overhead:+.1f}%")
        print(f"   Additional Time: {duration_overhead:+.1f}s")
        
        print(f"\n💡 CONCLUSIONS:")
        if abs(cpu_overhead) < 10:
            print("   ✅ Middleware overhead is minimal (<10% CPU)")
        elif abs(cpu_overhead) < 25:
            print("   ⚠️ Moderate middleware overhead (10-25% CPU)")
        else:
            print("   ❌ High middleware overhead (>25% CPU)")
        
        if direct_results['gpu_util_change'] > 0:
            print("   ✅ GPU is being utilized for inference")
        else:
            print("   ⚠️ GPU utilization not detected in test")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
