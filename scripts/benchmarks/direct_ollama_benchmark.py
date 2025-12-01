#!/usr/bin/env python3
"""
Direct Ollama API Performance Benchmark
Measures raw Ollama performance without AURA middleware overhead
"""

import time
import psutil
import subprocess
import json
import threading
import requests
import statistics
from typing import Dict, List

class DirectOllamaBenchmark:
    def __init__(self):
        # Same models as AURA benchmark for direct comparison
        self.models = {
            'phi3.5:3.8b': {'size': '3.8B', 'quantization': 'Q4_0'},
            'deepseek-coder:6.7b': {'size': '7B', 'quantization': 'Q4_0'},
            'deepseek-r1:1.5b': {'size': '1.8B', 'quantization': 'Q4_K_M'},
            'qwen2.5:7b': {'size': '7.6B', 'quantization': 'Q4_K_M'},
            'llama2:7b': {'size': '7B', 'quantization': 'Q4_0'},
            'phi:latest': {'size': '3B', 'quantization': 'Q4_0'},
            'tinyllama:latest': {'size': '1B', 'quantization': 'Q4_0'}
        }
        
        # Same test prompts for fair comparison
        self.test_prompts = {
            'simple_question': "What is artificial intelligence?",
            'coding_task': "Write a Python function to sort a list of numbers",
            'reasoning_task': "Explain step by step how to solve: 2x + 5 = 15"
        }
        
        self.results = []
        self.ollama_url = "http://localhost:11434/api"
    
    def get_gpu_metrics(self) -> Dict:
        """Get GPU utilization and temperature"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'utilization': float(values[0]),
                    'memory_used_mb': float(values[1]),
                    'memory_total_mb': float(values[2]),
                    'temperature': float(values[3])
                }
        except:
            pass
        return {'utilization': 0, 'memory_used_mb': 0, 'memory_total_mb': 0, 'temperature': 0}
    
    def monitor_system_metrics(self, duration_seconds: int = 30):
        """Monitor system during inference with thread-safe collection"""
        metrics = {
            'cpu_readings': [],
            'gpu_readings': [],
            'active': True
        }
        
        def collect_metrics():
            while metrics['active']:
                metrics['cpu_readings'].append(psutil.cpu_percent(interval=0.1))
                metrics['gpu_readings'].append(self.get_gpu_metrics()['utilization'])
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=collect_metrics)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return metrics
    
    def direct_ollama_generate(self, model: str, prompt: str) -> Dict:
        """Make direct API call to Ollama generate endpoint"""
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'num_gpu': 1,
                'gpu_layers': 999,
                'num_predict': 200,  # Consistent response length
                'temperature': 0.7
            }
        }
        
        response = requests.post(
            f"{self.ollama_url}/generate",
            json=payload,
            timeout=180
        )
        
        return response
    
    def direct_ollama_chat(self, model: str, prompt: str) -> Dict:
        """Make direct API call to Ollama chat endpoint"""
        payload = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'stream': False,
            'options': {
                'num_gpu': 1,
                'gpu_layers': 999,
                'num_predict': 200,
                'temperature': 0.7
            }
        }
        
        response = requests.post(
            f"{self.ollama_url}/chat",
            json=payload,
            timeout=180
        )
        
        return response
    
    def benchmark_direct_ollama(self, model: str, prompt: str, prompt_type: str, api_type: str = 'generate') -> Dict:
        """Benchmark direct Ollama API performance"""
        print(f"  🔍 Direct Ollama {api_type.upper()}: {model} ({prompt_type})")
        
        # Start system monitoring
        monitor_metrics = self.monitor_system_metrics()
        
        # Pre-test baseline
        baseline_gpu = self.get_gpu_metrics()
        baseline_cpu = psutil.cpu_percent(interval=1)
        
        start_time = time.time()
        try:
            # Choose API endpoint
            if api_type == 'chat':
                response = self.direct_ollama_chat(model, prompt)
            else:
                response = self.direct_ollama_generate(model, prompt)
            
            end_time = time.time()
            
            # Stop monitoring
            monitor_metrics['active'] = False
            time.sleep(0.2)  # Allow final readings
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response text based on API type
                if api_type == 'chat':
                    response_text = data.get('message', {}).get('content', '')
                else:
                    response_text = data.get('response', '')
                
                # Calculate performance metrics
                inference_time = end_time - start_time
                token_count = len(response_text.split())
                tps = token_count / inference_time if inference_time > 0 else 0
                
                # Calculate system usage statistics
                cpu_readings = monitor_metrics['cpu_readings']
                gpu_readings = monitor_metrics['gpu_readings']
                
                cpu_stats = {
                    'baseline': baseline_cpu,
                    'avg': statistics.mean(cpu_readings) if cpu_readings else 0,
                    'max': max(cpu_readings) if cpu_readings else 0,
                    'min': min(cpu_readings) if cpu_readings else 0
                }
                
                gpu_stats = {
                    'baseline': baseline_gpu['utilization'],
                    'avg': statistics.mean(gpu_readings) if gpu_readings else 0,
                    'max': max(gpu_readings) if gpu_readings else 0,
                    'min': min(gpu_readings) if gpu_readings else 0
                }
                
                final_gpu = self.get_gpu_metrics()
                
                result = {
                    'model': model,
                    'api_type': api_type,
                    'prompt_type': prompt_type,
                    'success': True,
                    'inference_time': round(inference_time, 2),
                    'tokens_generated': token_count,
                    'tokens_per_second': round(tps, 2),
                    'cpu_usage': {
                        'baseline_percent': round(cpu_stats['baseline'], 1),
                        'average_percent': round(cpu_stats['avg'], 1),
                        'peak_percent': round(cpu_stats['max'], 1),
                        'readings_count': len(cpu_readings)
                    },
                    'gpu_usage': {
                        'baseline_percent': round(gpu_stats['baseline'], 1),
                        'average_percent': round(gpu_stats['avg'], 1),
                        'peak_percent': round(gpu_stats['max'], 1),
                        'readings_count': len(gpu_readings)
                    },
                    'gpu_temperature': round(final_gpu['temperature'], 1),
                    'response_preview': response_text[:100] + "..." if len(response_text) > 100 else response_text,
                    'ollama_native_metrics': {
                        'eval_count': data.get('eval_count', 0),
                        'eval_duration_ns': data.get('eval_duration', 0),
                        'prompt_eval_count': data.get('prompt_eval_count', 0),
                        'prompt_eval_duration_ns': data.get('prompt_eval_duration', 0),
                        'load_duration_ns': data.get('load_duration', 0),
                        'total_duration_ns': data.get('total_duration', 0)
                    }
                }
                
                print(f"    ✅ {tps:.1f} TPS | CPU: {cpu_stats['avg']:.1f}% | GPU: {gpu_stats['avg']:.1f}% | {inference_time:.1f}s")
                return result
                
            else:
                monitor_metrics['active'] = False
                print(f"    ❌ HTTP {response.status_code}")
                return {
                    'model': model,
                    'api_type': api_type,
                    'prompt_type': prompt_type,
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            monitor_metrics['active'] = False
            print(f"    ❌ Error: {str(e)[:50]}...")
            return {
                'model': model,
                'api_type': api_type,
                'prompt_type': prompt_type,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_direct_benchmark(self):
        """Run benchmark using both Ollama API endpoints"""
        print("🔥 DIRECT OLLAMA API PERFORMANCE BENCHMARK")
        print("=" * 60)
        print("Testing pure Ollama performance (no AURA middleware)")
        print(f"Models: {len(self.models)} | Prompts: {len(self.test_prompts)} | APIs: 2")
        print("=" * 60)
        
        test_count = 0
        total_tests = len(self.models) * len(self.test_prompts) * 2  # 2 API types
        
        for model_name, model_info in self.models.items():
            print(f"\n📊 {model_name} ({model_info['size']}, {model_info['quantization']})")
            print("-" * 55)
            
            for prompt_type, prompt in self.test_prompts.items():
                # Test both API endpoints
                for api_type in ['generate', 'chat']:
                    test_count += 1
                    print(f"[{test_count}/{total_tests}]", end=" ")
                    
                    result = self.benchmark_direct_ollama(model_name, prompt, prompt_type, api_type)
                    self.results.append(result)
                    
                    # Brief cooldown between tests
                    time.sleep(1.5)
        
        return self.results
    
    def generate_direct_performance_report(self):
        """Generate comprehensive direct Ollama performance report"""
        successful_results = [r for r in self.results if r['success']]
        
        print("\n" + "="*85)
        print("🔥 DIRECT OLLAMA PERFORMANCE ANALYSIS")
        print("="*85)
        
        # Performance comparison table
        print(f"\n{'Model':<20} {'API':<8} {'Type':<12} {'TPS':<6} {'CPU%':<6} {'GPU%':<6} {'Time':<6}")
        print("-" * 85)
        
        for result in self.results:
            if result['success']:
                r = result
                print(f"{r['model']:<20} {r['api_type']:<8} {r['prompt_type']:<12} "
                      f"{r['tokens_per_second']:<6.1f} "
                      f"{r['cpu_usage']['average_percent']:<6.1f} "
                      f"{r['gpu_usage']['average_percent']:<6.1f} "
                      f"{r['inference_time']:<6.1f}")
            else:
                print(f"{result['model']:<20} {result['api_type']:<8} {result['prompt_type']:<12} FAILED")
        
        if successful_results:
            print(f"\n🚀 DIRECT OLLAMA CHAMPIONS")
            print("-" * 50)
            
            # Performance leaders
            fastest = max(successful_results, key=lambda x: x['tokens_per_second'])
            most_efficient = max(successful_results, key=lambda x: x['tokens_per_second'] / max(x['cpu_usage']['average_percent'], 1))
            best_gpu = max(successful_results, key=lambda x: x['gpu_usage']['average_percent'])
            lowest_cpu = min(successful_results, key=lambda x: x['cpu_usage']['average_percent'])
            
            print(f"🥇 Fastest TPS: {fastest['model']} ({fastest['api_type']}) - {fastest['tokens_per_second']:.1f} TPS")
            print(f"⚡ Most Efficient: {most_efficient['model']} ({most_efficient['api_type']}) - {most_efficient['tokens_per_second'] / max(most_efficient['cpu_usage']['average_percent'], 1):.3f} TPS/CPU%")
            print(f"🎯 Best GPU Usage: {best_gpu['model']} ({best_gpu['api_type']}) - {best_gpu['gpu_usage']['average_percent']:.1f}%")
            print(f"💚 Lowest CPU: {lowest_cpu['model']} ({lowest_cpu['api_type']}) - {lowest_cpu['cpu_usage']['average_percent']:.1f}%")
            
            # API comparison
            generate_results = [r for r in successful_results if r['api_type'] == 'generate']
            chat_results = [r for r in successful_results if r['api_type'] == 'chat']
            
            if generate_results and chat_results:
                generate_avg_tps = statistics.mean([r['tokens_per_second'] for r in generate_results])
                chat_avg_tps = statistics.mean([r['tokens_per_second'] for r in chat_results])
                
                print(f"\n📡 API ENDPOINT COMPARISON")
                print("-" * 30)
                print(f"Generate API: {generate_avg_tps:.1f} TPS average")
                print(f"Chat API: {chat_avg_tps:.1f} TPS average")
                print(f"Difference: {abs(generate_avg_tps - chat_avg_tps):.1f} TPS")
        
        # Save results
        output_file = 'direct_ollama_benchmark.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Complete results: {output_file}")
        return self.results

if __name__ == "__main__":
    benchmark = DirectOllamaBenchmark()
    results = benchmark.run_comprehensive_direct_benchmark()
    benchmark.generate_direct_performance_report()
