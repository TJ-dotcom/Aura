#!/usr/bin/env python3
"""
Comprehensive Model Performance Benchmark
Measures TPS, CPU, and GPU consumption for all AURA models
"""

import time
import psutil
import subprocess
import json
import threading
from typing import Dict, List, Tuple
import requests
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aura_engine.orchestrator.model_manager import ModelManager
from aura_engine.performance.monitor import PerformanceMonitor

class ModelBenchmark:
    def __init__(self):
        # All models available in AURA
        self.models = {
            'deepseek-coder:6.7b': 'Complex coding tasks',
            'deepseek-r1:1.5b': 'Math and reasoning',
            'phi3.5:3.8b': 'General purpose',
            'llama2:7b-chat': 'General chat',
            'codellama:7b-instruct': 'Code instruction'
        }
        
        # Test prompts for different scenarios
        self.test_prompts = {
            'short': "What is Python?",
            'medium': "Write a Python function to calculate the factorial of a number using recursion.",
            'long': "Explain the concept of machine learning, provide examples of supervised and unsupervised learning, and write a complete Python script that demonstrates a simple linear regression model using scikit-learn with sample data, including data visualization and model evaluation metrics."
        }
        
        self.results = []
        
    def get_gpu_usage(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def measure_ollama_inference(self, model: str, prompt: str) -> Dict:
        """Measure performance of Ollama model inference"""
        print(f"🔍 Testing {model} with prompt length: {len(prompt)} chars")
        
        # Pre-test measurements
        cpu_before = self.get_cpu_usage()
        gpu_before = self.get_gpu_usage()
        memory_before = self.get_memory_usage()
        
        # Start monitoring
        cpu_readings = []
        gpu_readings = []
        
        def monitor_usage():
            for _ in range(30):  # Monitor for 30 seconds max
                cpu_readings.append(psutil.cpu_percent(interval=0.1))
                gpu_readings.append(self.get_gpu_usage())
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_usage)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Make request to Ollama
        start_time = time.time()
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_gpu': 1,
                        'gpu_layers': 999
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                response_text = data.get('response', '')
                token_count = len(response_text.split())
                tps = token_count / total_time if total_time > 0 else 0
                
                # Post-test measurements
                cpu_after = self.get_cpu_usage()
                gpu_after = self.get_gpu_usage()
                memory_after = self.get_memory_usage()
                
                # Calculate averages during inference
                avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else cpu_after
                max_cpu = max(cpu_readings) if cpu_readings else cpu_after
                avg_gpu = sum(gpu_readings) / len(gpu_readings) if gpu_readings else gpu_after
                max_gpu = max(gpu_readings) if gpu_readings else gpu_after
                
                return {
                    'model': model,
                    'prompt_length': len(prompt),
                    'response_length': len(response_text),
                    'token_count': token_count,
                    'total_time_seconds': round(total_time, 2),
                    'tokens_per_second': round(tps, 2),
                    'cpu_usage': {
                        'before': round(cpu_before, 1),
                        'average': round(avg_cpu, 1),
                        'max': round(max_cpu, 1),
                        'after': round(cpu_after, 1)
                    },
                    'gpu_usage': {
                        'before': round(gpu_before, 1),
                        'average': round(avg_gpu, 1),
                        'max': round(max_gpu, 1),
                        'after': round(gpu_after, 1)
                    },
                    'memory_usage': {
                        'before_gb': round(memory_before['used_gb'], 2),
                        'after_gb': round(memory_after['used_gb'], 2),
                        'delta_gb': round(memory_after['used_gb'] - memory_before['used_gb'], 2)
                    },
                    'success': True,
                    'eval_count': data.get('eval_count', 0),
                    'eval_duration': data.get('eval_duration', 0)
                }
            else:
                return {'model': model, 'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'model': model, 'success': False, 'error': str(e)}
    
    def run_comprehensive_benchmark(self):
        """Run benchmark tests for all models and prompt types"""
        print("🚀 Starting Comprehensive AURA Model Benchmark")
        print("=" * 60)
        
        total_tests = len(self.models) * len(self.test_prompts)
        current_test = 0
        
        for model_name, description in self.models.items():
            print(f"\n📊 BENCHMARKING: {model_name}")
            print(f"Description: {description}")
            print("-" * 50)
            
            model_results = []
            
            for prompt_type, prompt in self.test_prompts.items():
                current_test += 1
                print(f"[{current_test}/{total_tests}] Testing {prompt_type} prompt...")
                
                # Wait for system to stabilize
                time.sleep(2)
                
                result = self.measure_ollama_inference(model_name, prompt)
                result['prompt_type'] = prompt_type
                result['description'] = description
                
                model_results.append(result)
                
                if result['success']:
                    print(f"  ✅ TPS: {result['tokens_per_second']}, "
                          f"CPU: {result['cpu_usage']['average']}%, "
                          f"GPU: {result['gpu_usage']['average']}%")
                else:
                    print(f"  ❌ Error: {result.get('error', 'Unknown')}")
                
                # Brief pause between tests
                time.sleep(1)
            
            self.results.extend(model_results)
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*80)
        
        # Summary table
        print("\n🏆 PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Model':<20} {'Prompt':<8} {'TPS':<6} {'CPU%':<6} {'GPU%':<6} {'Time(s)':<8} {'Status':<8}")
        print("-" * 80)
        
        for result in self.results:
            if result['success']:
                print(f"{result['model']:<20} {result['prompt_type']:<8} "
                      f"{result['tokens_per_second']:<6.1f} "
                      f"{result['cpu_usage']['average']:<6.1f} "
                      f"{result['gpu_usage']['average']:<6.1f} "
                      f"{result['total_time_seconds']:<8.1f} ✅")
            else:
                print(f"{result['model']:<20} {result['prompt_type']:<8} "
                      f"{'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<8} ❌")
        
        # Best performers
        successful_results = [r for r in self.results if r['success']]
        
        if successful_results:
            print("\n🥇 TOP PERFORMERS")
            print("-" * 40)
            
            # Highest TPS
            best_tps = max(successful_results, key=lambda x: x['tokens_per_second'])
            print(f"🚀 Fastest TPS: {best_tps['model']} ({best_tps['tokens_per_second']:.1f} TPS)")
            
            # Lowest CPU usage
            best_cpu = min(successful_results, key=lambda x: x['cpu_usage']['average'])
            print(f"💚 Lowest CPU: {best_cpu['model']} ({best_cpu['cpu_usage']['average']:.1f}%)")
            
            # Highest GPU usage
            best_gpu = max(successful_results, key=lambda x: x['gpu_usage']['average'])
            print(f"⚡ Best GPU Util: {best_gpu['model']} ({best_gpu['gpu_usage']['average']:.1f}%)")
        
        # Save detailed results
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: benchmark_results.json")
        
        return self.results

if __name__ == "__main__":
    benchmark = ModelBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.generate_report()
