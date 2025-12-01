"""
Unit tests for llama.cpp wrapper functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import subprocess
import os
import tempfile
import time

from aura_engine.llama_wrapper.wrapper import LlamaWrapper, InferenceOutput, ProcessedOutput
from aura_engine.models import EngineConfig


class TestLlamaWrapper:
    """Test suite for LlamaWrapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EngineConfig(
            llama_cpp_path="llama.cpp",
            max_tokens=512,
            temperature=0.7
        )
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_validate_binary_success_in_path(self, mock_isfile, mock_which):
        """Test successful binary validation when llama.cpp is in PATH."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        
        wrapper = LlamaWrapper(self.config)
        
        mock_which.assert_called_once_with("llama.cpp")
        assert wrapper.config == self.config
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_validate_binary_success_direct_path(self, mock_isfile, mock_which):
        """Test successful binary validation with direct path."""
        mock_which.return_value = None
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        mock_which.assert_called_once_with("llama.cpp")
        mock_isfile.assert_called_once_with("llama.cpp")
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_validate_binary_not_found(self, mock_isfile, mock_which):
        """Test binary validation failure when llama.cpp is not found."""
        mock_which.return_value = None
        mock_isfile.return_value = False
        
        with pytest.raises(FileNotFoundError, match="llama.cpp binary not found"):
            LlamaWrapper(self.config)
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_build_command_basic(self, mock_isfile, mock_which):
        """Test basic command building."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True  # For model file check
        
        wrapper = LlamaWrapper(self.config)
        
        with patch('os.path.isfile', return_value=True):
            command = wrapper.build_command("model.gguf", "Hello world", 0)
        
        expected_command = [
            "llama.cpp",
            "-m", "model.gguf",
            "-p", "Hello world",
            "-n", "512",
            "--temp", "0.7",
            "-c", "2048",
            "--no-display-prompt",
            "-t", "8"
        ]
        
        assert command == expected_command
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_build_command_with_gpu_layers(self, mock_isfile, mock_which):
        """Test command building with GPU layers."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        with patch('os.path.isfile', return_value=True):
            command = wrapper.build_command("model.gguf", "Hello world", 25)
        
        # Should include GPU layer arguments
        assert "-ngl" in command
        assert "25" in command
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_build_command_model_not_found(self, mock_isfile, mock_which):
        """Test command building with missing model file."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        
        wrapper = LlamaWrapper(self.config)
        
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                wrapper.build_command("missing_model.gguf", "Hello world", 0)
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('subprocess.run')
    @patch('time.time')
    def test_execute_inference_success(self, mock_time, mock_run, mock_isfile, mock_which):
        """Test successful inference execution."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        # Mock time progression
        mock_time.side_effect = [100.0, 102.5]  # Start and end times
        
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.stdout = "Generated response text"
        mock_result.stderr = "llama.cpp: model loaded in 2.5s"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        wrapper = LlamaWrapper(self.config)
        command = ["llama.cpp", "-m", "model.gguf", "-p", "test"]
        
        result = wrapper.execute_inference(command)
        
        assert isinstance(result, InferenceOutput)
        assert result.stdout == "Generated response text"
        assert result.stderr == "llama.cpp: model loaded in 2.5s"
        assert result.returncode == 0
        assert result.execution_time == 2.5
        
        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace'
        )
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('subprocess.run')
    def test_execute_inference_timeout(self, mock_run, mock_isfile, mock_which):
        """Test inference execution timeout."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        mock_run.side_effect = subprocess.TimeoutExpired(["llama.cpp"], 300)
        
        wrapper = LlamaWrapper(self.config)
        command = ["llama.cpp", "-m", "model.gguf", "-p", "test"]
        
        with pytest.raises(subprocess.TimeoutExpired):
            wrapper.execute_inference(command)
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('subprocess.run')
    def test_execute_inference_subprocess_error(self, mock_run, mock_isfile, mock_which):
        """Test inference execution with subprocess error."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        mock_run.side_effect = Exception("Subprocess failed")
        
        wrapper = LlamaWrapper(self.config)
        command = ["llama.cpp", "-m", "model.gguf", "-p", "test"]
        
        with pytest.raises(subprocess.SubprocessError, match="Failed to execute llama.cpp"):
            wrapper.execute_inference(command)
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_parse_output_success(self, mock_isfile, mock_which):
        """Test successful output parsing."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        raw_output = InferenceOutput(
            stdout="This is the generated response from the model.",
            stderr="llama.cpp: model loaded in 2.5s\\nGeneration completed",
            returncode=0,
            execution_time=5.2
        )
        
        result = wrapper.parse_output(raw_output)
        
        assert isinstance(result, ProcessedOutput)
        assert result.response == "This is the generated response from the model."
        assert result.tokens_generated > 0
        assert result.model_load_time == 2.5
        assert result.inference_time == 5.2
        assert result.error_message is None
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_parse_output_error(self, mock_isfile, mock_which):
        """Test output parsing with error return code."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        raw_output = InferenceOutput(
            stdout="",
            stderr="Error: Model file not found",
            returncode=1,
            execution_time=0.1
        )
        
        result = wrapper.parse_output(raw_output)
        
        assert result.response == ""
        assert result.tokens_generated == 0
        assert result.error_message is not None
        assert "return code 1" in result.error_message
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_extract_response_clean(self, mock_isfile, mock_which):
        """Test response extraction with clean output."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        stdout = """llama_model_loader: loaded meta data
This is the actual response from the model.
It spans multiple lines.
llama_print_timings: load time = 2500.00 ms"""
        
        response = wrapper._extract_response(stdout)
        
        assert "This is the actual response from the model." in response
        assert "It spans multiple lines." in response
        assert "llama_model_loader" not in response
        assert "llama_print_timings" not in response
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_extract_model_load_time_ms(self, mock_isfile, mock_which):
        """Test model load time extraction from milliseconds."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        stderr = "llama_load_model_from_file: loaded meta data in 2500.00 ms"
        
        load_time = wrapper._extract_model_load_time(stderr)
        
        assert load_time == 2.5  # Converted to seconds
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_extract_model_load_time_seconds(self, mock_isfile, mock_which):
        """Test model load time extraction from seconds."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        stderr = "model loaded in 3.2 s"
        
        load_time = wrapper._extract_model_load_time(stderr)
        
        assert load_time == 3.2
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_extract_model_load_time_not_found(self, mock_isfile, mock_which):
        """Test model load time extraction when not found."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        stderr = "Some other log messages without timing info"
        
        load_time = wrapper._extract_model_load_time(stderr)
        
        assert load_time is None
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_count_tokens(self, mock_isfile, mock_which):
        """Test token counting estimation."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        wrapper = LlamaWrapper(self.config)
        
        # Test with various text lengths
        assert wrapper._count_tokens("") == 0
        assert wrapper._count_tokens("Hi") == 1  # Minimum 1 token
        assert wrapper._count_tokens("This is a test sentence.") > 1
        
        # Roughly 4 characters per token
        long_text = "A" * 100
        tokens = wrapper._count_tokens(long_text)
        assert 20 <= tokens <= 30  # Should be around 25 tokens
    
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch.object(LlamaWrapper, 'build_command')
    @patch.object(LlamaWrapper, 'execute_inference')
    @patch.object(LlamaWrapper, 'parse_output')
    def test_run_inference_complete_workflow(self, mock_parse, mock_execute, 
                                           mock_build, mock_isfile, mock_which):
        """Test complete inference workflow integration."""
        mock_which.return_value = "/usr/bin/llama.cpp"
        mock_isfile.return_value = True
        
        # Mock the workflow components
        mock_build.return_value = ["llama.cpp", "-m", "model.gguf"]
        mock_execute.return_value = InferenceOutput("response", "", 0, 2.5)
        mock_parse.return_value = ProcessedOutput("response", 10, 1.5, 2.5, None)
        
        wrapper = LlamaWrapper(self.config)
        
        result = wrapper.run_inference("model.gguf", "test prompt", 25)
        
        # Verify all components were called
        mock_build.assert_called_once_with("model.gguf", "test prompt", 25)
        mock_execute.assert_called_once()
        mock_parse.assert_called_once()
        
        # Verify result
        assert isinstance(result, ProcessedOutput)
        assert result.response == "response"
        assert result.tokens_generated == 10


if __name__ == "__main__":
    pytest.main([__file__])