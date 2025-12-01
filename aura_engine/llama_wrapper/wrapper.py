"""
llama.cpp integration wrapper for subprocess execution and output parsing.
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import shutil

from ..models import EngineConfig


logger = logging.getLogger(__name__)


@dataclass
class InferenceOutput:
    """Raw output from llama.cpp subprocess execution."""
    stdout: str
    stderr: str
    returncode: int
    execution_time: float


@dataclass
class ProcessedOutput:
    """Processed and parsed output from llama.cpp."""
    response: str
    tokens_generated: int
    model_load_time: Optional[float]
    inference_time: float
    error_message: Optional[str]


class LlamaWrapper:
    """
    Wrapper for llama.cpp binary execution with proper error handling
    and output parsing.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validate_binary()
    
    def _validate_binary(self) -> None:
        """
        Validate that llama.cpp binary exists and is executable.
        
        Raises:
            FileNotFoundError: If llama.cpp binary is not found
        """
        # Check if the binary exists in PATH or at specified path
        binary_path = shutil.which(self.config.llama_cpp_path)
        if binary_path is None and not os.path.isfile(self.config.llama_cpp_path):
            raise FileNotFoundError(
                f"llama.cpp binary not found at '{self.config.llama_cpp_path}'. "
                f"Please ensure llama.cpp is installed and accessible."
            )
        
        self.logger.info(f"llama.cpp binary validated at: {binary_path or self.config.llama_cpp_path}")
    
    def build_command(self, model_path: str, prompt: str, gpu_layers: int = 0) -> List[str]:
        """
        Build llama.cpp command with optimal parameters.
        
        Args:
            model_path: Path to the GGUF model file
            prompt: Input prompt for inference
            gpu_layers: Number of GPU layers to use
            
        Returns:
            List[str]: Complete command arguments for subprocess
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        command = [
            self.config.llama_cpp_path,
            "-m", model_path,
            "-p", prompt,
            "-n", str(self.config.max_tokens),
            "--temp", str(self.config.temperature),
            "-c", "2048",  # Context size
            "--no-display-prompt",  # Don't echo the prompt in output
        ]
        
        # Add GPU layers if specified
        if gpu_layers > 0:
            command.extend(["-ngl", str(gpu_layers)])
        
        # Add threading for CPU performance
        command.extend(["-t", "8"])  # Use 8 threads
        
        self.logger.debug(f"Built command: {' '.join(command)}")
        return command
    
    def execute_inference(self, command: List[str], timeout: int = 300) -> InferenceOutput:
        """
        Execute llama.cpp subprocess with proper stream handling.
        
        Args:
            command: Command arguments for subprocess
            timeout: Maximum execution time in seconds
            
        Returns:
            InferenceOutput: Raw subprocess output and metadata
            
        Raises:
            subprocess.TimeoutExpired: If execution exceeds timeout
            subprocess.SubprocessError: If subprocess fails
        """
        self.logger.info("Starting llama.cpp inference...")
        start_time = time.time()
        
        try:
            # Execute subprocess with proper stream capture
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'  # Handle encoding issues gracefully
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"llama.cpp execution completed in {execution_time:.2f}s")
            self.logger.debug(f"Return code: {result.returncode}")
            
            if result.stderr:
                self.logger.debug(f"stderr output: {result.stderr[:500]}...")  # Log first 500 chars
            
            return InferenceOutput(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            self.logger.error(f"llama.cpp execution timed out after {timeout}s")
            raise subprocess.TimeoutExpired(e.cmd, e.timeout, e.output, e.stderr)
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"llama.cpp execution failed: {e}")
            raise subprocess.SubprocessError(f"Failed to execute llama.cpp: {e}")
    
    def parse_output(self, raw_output: InferenceOutput) -> ProcessedOutput:
        """
        Parse and process raw llama.cpp output.
        
        Args:
            raw_output: Raw subprocess output
            
        Returns:
            ProcessedOutput: Parsed and structured output
        """
        # Check for execution errors
        if raw_output.returncode != 0:
            error_msg = f"llama.cpp failed with return code {raw_output.returncode}"
            if raw_output.stderr:
                error_msg += f": {raw_output.stderr}"
            
            return ProcessedOutput(
                response="",
                tokens_generated=0,
                model_load_time=None,
                inference_time=raw_output.execution_time,
                error_message=error_msg
            )
        
        # Extract model response from stdout
        response = self._extract_response(raw_output.stdout)
        
        # Parse metadata from stderr (llama.cpp logs to stderr)
        model_load_time = self._extract_model_load_time(raw_output.stderr)
        tokens_generated = self._count_tokens(response)
        
        return ProcessedOutput(
            response=response,
            tokens_generated=tokens_generated,
            model_load_time=model_load_time,
            inference_time=raw_output.execution_time,
            error_message=None
        )
    
    def _extract_response(self, stdout: str) -> str:
        """
        Extract the actual model response from stdout.
        
        Args:
            stdout: Raw stdout from llama.cpp
            
        Returns:
            str: Cleaned model response
        """
        if not stdout:
            return ""
        
        # llama.cpp output format varies, but generally the response
        # comes after system messages and before timing info
        lines = stdout.strip().split('\n')
        
        # Filter out system messages and metadata
        response_lines = []
        for line in lines:
            # Skip empty lines and system messages
            if (line.strip() and 
                not line.startswith('llama_') and 
                not line.startswith('system_info:') and
                not 'tokens per second' in line.lower() and
                not 'load time:' in line.lower()):
                response_lines.append(line)
        
        response = '\n'.join(response_lines).strip()
        self.logger.debug(f"Extracted response ({len(response)} chars): {response[:100]}...")
        return response
    
    def _extract_model_load_time(self, stderr: str) -> Optional[float]:
        """
        Extract model load time from stderr output.
        
        Args:
            stderr: Raw stderr from llama.cpp
            
        Returns:
            Optional[float]: Model load time in seconds, or None if not found
        """
        if not stderr:
            return None
        
        # Look for load time patterns in stderr
        # llama.cpp typically outputs: "llama_load_model_from_file: loaded meta data with X key-value pairs and Y tensors from Z in Xs"
        import re
        
        patterns = [
            r'loaded.*in\s+(\d+\.?\d*)\s*ms',
            r'load time:\s*(\d+\.?\d*)\s*ms',
            r'model loaded in\s+(\d+\.?\d*)\s*s',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, stderr, re.IGNORECASE)
            if match:
                time_value = float(match.group(1))
                # Convert ms to seconds if needed
                if 'ms' in pattern:
                    time_value /= 1000.0
                self.logger.debug(f"Extracted model load time: {time_value}s")
                return time_value
        
        return None
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for the generated text.
        
        Args:
            text: Generated text
            
        Returns:
            int: Estimated token count
        """
        if not text:
            return 0
        
        # Simple approximation: ~4 characters per token for English text
        # This is a rough estimate; actual tokenization would be more accurate
        estimated_tokens = len(text) // 4
        return max(1, estimated_tokens)  # At least 1 token if there's text
    
    def run_inference(self, model_path: str, prompt: str, gpu_layers: int = 0) -> ProcessedOutput:
        """
        Complete inference workflow: build command, execute, and parse output.
        
        Args:
            model_path: Path to the GGUF model file
            prompt: Input prompt for inference
            gpu_layers: Number of GPU layers to use
            
        Returns:
            ProcessedOutput: Complete processed inference result
        """
        self.logger.info(f"Running inference with model: {os.path.basename(model_path)}")
        self.logger.info(f"GPU layers: {gpu_layers}, Max tokens: {self.config.max_tokens}")
        
        # Build and execute command
        command = self.build_command(model_path, prompt, gpu_layers)
        raw_output = self.execute_inference(command)
        
        # Parse and return processed output
        processed = self.parse_output(raw_output)
        
        if processed.error_message:
            self.logger.error(f"Inference failed: {processed.error_message}")
        else:
            self.logger.info(f"Inference successful: {processed.tokens_generated} tokens generated")
        
        return processed