#!/usr/bin/env python3
"""
Focused AURA Model Performance Benchmark
Measures TPS, CPU, and GPU consumption for available models
"""

import time
import psutil
import subprocess
import json
import threading
from typing import Dict, List
import requests
import statistics

class FocusedModelBenchmark:
    def __init__(self):
        # Available models from Ollama
        self.models = {
            'phi3.5:3.8b': {'size': '3.8B', 'quantization': 'Q4_0', 'family': 'phi3'},
            'deepseek-coder:6.7b': {'size': '7B', 'quantization': 'Q4_0', 'family': 'llama'},
            'deepseek-r1:1.5b': {'size': '1.8B', 'quantization': 'Q4_K_M', 'family': 'qwen2'},
            'qwen2.5:7b': {'size': '7.6B', 'quantization': 'Q4_K_M', 'family': 'qwen2'},
            'llama2:7b': {'size': '7B', 'quantization': 'Q4_0', 'family': 'llama'},
            'phi:latest': {'size': '3B', 'quantization': 'Q4_0', 'family': 'phi2'},
            'tinyllama:latest': {'size': '1B', 'quantization': 'Q4_0', 'family': 'llama'}
        }
        
        # Standard test prompts
        self.test_prompts = {
            'simple_question': "What is artificial intelligence?",
            'coding_task': "Write a Python function to sort a list of numbers",
            'reasoning_task': "Explain step by step how to solve: 2x + 5 = 15"
        }
        
        self.results = []
    
    def get_gpu_metrics(self) -> Dict:
        """Get detailed GPU metrics"""
        try:
            # Get GPU utilization
            util_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if util_result.returncode == 0:
                values = util_result.stdout.strip().split(', ')
                return {
                    'utilization': float(values[0]),
                    'memory_used_mb': float(values[1]),
                    'memory_total_mb': float(values[2]),
                    'temperature': float(values[3]),
                    'memory_percent': (float(values[1]) / float(values[2])) * 100
                }
        except:
            pass
        return {'utilization': 0, 'memory_used_mb': 0, 'memory_total_mb': 0, 'temperature': 0, 'memory_percent': 0}
    
    def monitor_system_during_inference(self, duration_seconds: int = 10):
        """Monitor system metrics during inference"""
        cpu_readings = []
        gpu_readings = []
        memory_readings = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            cpu_readings.append(psutil.cpu_percent(interval=0.1))
            gpu_metrics = self.get_gpu_metrics()
            gpu_readings.append(gpu_metrics['utilization'])
            memory_readings.append(psutil.virtual_memory().percent)
            time.sleep(0.5)
        
        return {
            'cpu': {
                'min': min(cpu_readings) if cpu_readings else 0,
                'max': max(cpu_readings) if cpu_readings else 0,
                'avg': statistics.mean(cpu_readings) if cpu_readings else 0,
                'readings': len(cpu_readings)
            },
            'gpu': {
                'min': min(gpu_readings) if gpu_readings else 0,
                'max': max(gpu_readings) if gpu_readings else 0,
                'avg': statistics.mean(gpu_readings) if gpu_readings else 0,
                'readings': len(gpu_readings)
            },
            'memory': {
                'min': min(memory_readings) if memory_readings else 0,
                'max': max(memory_readings) if memory_readings else 0,
                'avg': statistics.mean(memory_readings) if memory_readings else 0
            }
        }
    
    def benchmark_model(self, model_name: str, prompt: str, prompt_type: str) -> Dict:
        """Benchmark a specific model with monitoring"""
        print(f"  🔍 Testing {model_name} ({prompt_type})")
        
        # Pre-inference baseline
        baseline_gpu = self.get_gpu_metrics()
        baseline_cpu = psutil.cpu_percent(interval=1)
        
        # Start monitoring thread
        monitoring_active = True
        monitoring_results = {'cpu': [], 'gpu': []}
        
        def monitor_loop():
            while monitoring_active:
                monitoring_results['cpu'].append(psutil.cpu_percent(interval=0.1))
                monitoring_results['gpu'].append(self.get_gpu_metrics()['utilization'])
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Perform inference
        start_time = time.time()
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_gpu': 1,
                        'gpu_layers': 999,
                        'num_predict': 200  # Limit response length for consistent comparison
                    }
                },
                timeout=180
            )
            
            end_time = time.time()
            monitoring_active = False  # Stop monitoring
            
            if response.status_code == 200:
                data = response.json()
                
                # Calculate metrics
                inference_time = end_time - start_time
                response_text = data.get('response', '')
                token_count = len(response_text.split())
                tps = token_count / inference_time if inference_time > 0 else 0
                
                # System metrics during inference
                cpu_stats = {
                    'baseline': baseline_cpu,
                    'avg': statistics.mean(monitoring_results['cpu']) if monitoring_results['cpu'] else 0,
                    'max': max(monitoring_results['cpu']) if monitoring_results['cpu'] else 0,
                    'min': min(monitoring_results['cpu']) if monitoring_results['cpu'] else 0
                }
                
                gpu_stats = {
                    'baseline': baseline_gpu['utilization'],
                    'avg': statistics.mean(monitoring_results['gpu']) if monitoring_results['gpu'] else 0,
                    'max': max(monitoring_results['gpu']) if monitoring_results['gpu'] else 0,
                    'min': min(monitoring_results['gpu']) if monitoring_results['gpu'] else 0
                }
                
                # GPU memory info
                final_gpu = self.get_gpu_metrics()
                
                result = {
                    'model': model_name,
                    'model_info': self.models.get(model_name, {}),
                    'prompt_type': prompt_type,
                    'success': True,
                    'inference_time': round(inference_time, 2),
                    'tokens_generated': token_count,
                    'tokens_per_second': round(tps, 2),
                    'cpu_usage': {
                        'baseline_percent': round(cpu_stats['baseline'], 1),
                        'average_percent': round(cpu_stats['avg'], 1),
                        'peak_percent': round(cpu_stats['max'], 1)
                    },
                    'gpu_usage': {
                        'baseline_percent': round(gpu_stats['baseline'], 1),
                        'average_percent': round(gpu_stats['avg'], 1),
                        'peak_percent': round(gpu_stats['max'], 1)
                    },
                    'gpu_memory': {
                        'used_mb': round(final_gpu['memory_used_mb'], 1),
                        'total_mb': round(final_gpu['memory_total_mb'], 1),
                        'percent_used': round(final_gpu['memory_percent'], 1)
                    },
                    'gpu_temperature': round(final_gpu['temperature'], 1),
                    'ollama_metrics': {
                        'eval_count': data.get('eval_count', 0),
                        'eval_duration_ns': data.get('eval_duration', 0),
                        'prompt_eval_count': data.get('prompt_eval_count', 0),
                        'prompt_eval_duration_ns': data.get('prompt_eval_duration', 0)
                    }
                }
                
                print(f"    ✅ {tps:.1f} TPS | CPU: {cpu_stats['avg']:.1f}% | GPU: {gpu_stats['avg']:.1f}% | {inference_time:.1f}s")
                return result
                
            else:
                monitoring_active = False
                print(f"    ❌ HTTP {response.status_code}")
                return {
                    'model': model_name,
                    'prompt_type': prompt_type,
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            monitoring_active = False
            print(f"    ❌ {str(e)[:50]}...")
            return {
                'model': model_name,
                'prompt_type': prompt_type,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self):
        """Run benchmark for all available models"""
        print("🚀 AURA MODEL PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Testing {len(self.models)} models with {len(self.test_prompts)} prompt types")
        print("=" * 60)
        
        for model_name, model_info in self.models.items():
            print(f"\n📊 {model_name}")
            print(f"   Size: {model_info['size']}, Quantization: {model_info['quantization']}")
            print("-" * 50)
            
            for prompt_type, prompt in self.test_prompts.items():
                result = self.benchmark_model(model_name, prompt, prompt_type)
                self.results.append(result)
                
                # Cool down between tests
                time.sleep(2)
        
        return self.results
    
    def generate_performance_report(self):
        """Generate detailed performance analysis"""
        successful_results = [r for r in self.results if r['success']]
        
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Performance Summary Table
        print(f"\n{'Model':<20} {'Type':<12} {'TPS':<6} {'CPU%':<6} {'GPU%':<6} {'GPU°C':<6} {'Time':<6}")
        print("-" * 80)
        
        for result in self.results:
            if result['success']:
                r = result
                print(f"{r['model']:<20} {r['prompt_type']:<12} "
                      f"{r['tokens_per_second']:<6.1f} "
                      f"{r['cpu_usage']['average_percent']:<6.1f} "
                      f"{r['gpu_usage']['average_percent']:<6.1f} "
                      f"{r['gpu_temperature']:<6.1f} "
                      f"{r['inference_time']:<6.1f}")
            else:
                print(f"{result['model']:<20} {result['prompt_type']:<12} FAILED")
        
        if successful_results:
            print(f"\n🏆 PERFORMANCE CHAMPIONS")
            print("-" * 40)
            
            # Best TPS
            fastest = max(successful_results, key=lambda x: x['tokens_per_second'])
            print(f"🚀 Fastest: {fastest['model']} ({fastest['tokens_per_second']:.1f} TPS)")
            
            # Most efficient (best TPS/CPU ratio)
            most_efficient = max(successful_results, key=lambda x: x['tokens_per_second'] / max(x['cpu_usage']['average_percent'], 1))
            efficiency = most_efficient['tokens_per_second'] / max(most_efficient['cpu_usage']['average_percent'], 1)
            print(f"⚡ Most Efficient: {most_efficient['model']} ({efficiency:.2f} TPS/CPU%)")
            
            # Best GPU utilization
            best_gpu = max(successful_results, key=lambda x: x['gpu_usage']['average_percent'])
            print(f"🎯 Best GPU Usage: {best_gpu['model']} ({best_gpu['gpu_usage']['average_percent']:.1f}%)")
            
            # Lowest CPU usage
            lowest_cpu = min(successful_results, key=lambda x: x['cpu_usage']['average_percent'])
            print(f"💚 Lowest CPU: {lowest_cpu['model']} ({lowest_cpu['cpu_usage']['average_percent']:.1f}%)")
        
        # Save detailed results
        output_file = 'comprehensive_model_benchmark.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Full results: {output_file}")
        return self.results

if __name__ == "__main__":
    benchmark = FocusedModelBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.generate_performance_report()
