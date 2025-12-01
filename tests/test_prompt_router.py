"""
Unit tests for prompt router functionality.
"""

import pytest
from aura_engine.orchestrator.router import PromptRouter, RoutingRule
from aura_engine.models import ModelType


class TestPromptRouter:
    """Test suite for PromptRouter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = PromptRouter()
    
    def test_initialization(self):
        """Test router initialization and rule setup."""
        assert len(self.router.routing_rules) == 3  # CODER, WRITER, GENERAL
        
        # Verify all model types are represented
        model_types = {rule.model_type for rule in self.router.routing_rules}
        assert ModelType.CODER in model_types
        assert ModelType.WRITER in model_types
        assert ModelType.GENERAL in model_types
    
    def test_coding_prompt_detection(self):
        """Test detection of coding-related prompts."""
        coding_prompts = [
            "Write a Python function to calculate fibonacci numbers",
            "How do I debug this JavaScript code?",
            "Implement a binary search algorithm in C++",
            "Create a REST API using Django",
            "Fix this SQL query that's not working",
            "def factorial(n): return n * factorial(n-1)",
            "import pandas as pd",
            "class MyClass:",
            "git commit -m 'initial commit'",
            "docker build -t myapp .",
            "Write unit tests for this function",
            "Refactor this code to use design patterns"
        ]
        
        for prompt in coding_prompts:
            result = self.router.analyze_prompt(prompt)
            assert result == ModelType.CODER, f"Failed to detect coding prompt: {prompt}"
    
    def test_writing_prompt_detection(self):
        """Test detection of writing-related prompts."""
        writing_prompts = [
            "Write an essay about climate change",
            "Create a blog post about artificial intelligence",
            "Draft a professional email to my manager",
            "Compose a short story about time travel",
            "Write a product review for this laptop",
            "Create marketing copy for our new product",
            "Draft a research proposal for my thesis",
            "Write a letter of recommendation",
            "Compose a poem about nature",
            "Create content for our company newsletter",
            "Write an analysis of Shakespeare's Hamlet",
            "Draft a business proposal for investors"
        ]
        
        for prompt in writing_prompts:
            result = self.router.analyze_prompt(prompt)
            assert result == ModelType.WRITER, f"Failed to detect writing prompt: {prompt}"
    
    def test_general_prompt_detection(self):
        """Test detection of general conversational prompts."""
        general_prompts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Can you help me understand quantum physics?",
            "What's the weather like?",
            "How do I cook pasta?",
            "What are the benefits of exercise?",
            "What's your opinion on renewable energy?",
            "Can you recommend a good restaurant?",
            "How does the internet work?",
            "What are some fun facts about space?"
        ]
        
        for prompt in general_prompts:
            result = self.router.analyze_prompt(prompt)
            assert result == ModelType.GENERAL, f"Failed to detect general prompt: {prompt}"
        
        # Some prompts might be ambiguous and could go to WRITER (like explanatory requests)
        ambiguous_prompts = [
            "Tell me about the history of Rome",  # Could be WRITER for narrative
            "Explain photosynthesis to me",  # Could be WRITER for explanation
        ]
        
        for prompt in ambiguous_prompts:
            result = self.router.analyze_prompt(prompt)
            # These could reasonably be GENERAL or WRITER
            assert result in [ModelType.GENERAL, ModelType.WRITER], f"Unexpected routing for ambiguous prompt: {prompt}"
    
    def test_mixed_content_prompts(self):
        """Test prompts with mixed content that should favor stronger signals."""
        
        # Strong coding signal should override general words
        coding_dominant = "Hello, can you help me write a Python function for sorting?"
        assert self.router.analyze_prompt(coding_dominant) == ModelType.CODER
        
        # Strong writing signal should override general words
        writing_dominant = "Hi there, I need help writing an essay about technology"
        assert self.router.analyze_prompt(writing_dominant) == ModelType.WRITER
        
        # Ambiguous prompt should default to general
        ambiguous = "Can you help me with something?"
        assert self.router.analyze_prompt(ambiguous) == ModelType.GENERAL
    
    def test_empty_and_invalid_prompts(self):
        """Test handling of empty and invalid prompts."""
        
        # Empty prompts
        assert self.router.analyze_prompt("") == ModelType.GENERAL
        assert self.router.analyze_prompt("   ") == ModelType.GENERAL
        assert self.router.analyze_prompt(None) == ModelType.GENERAL
        
        # Very short prompts
        assert self.router.analyze_prompt("hi") == ModelType.GENERAL
        assert self.router.analyze_prompt("?") == ModelType.GENERAL
    
    def test_case_insensitive_matching(self):
        """Test that routing is case-insensitive."""
        
        # Test different cases for the same content
        prompts = [
            "Write a PYTHON function",
            "write a python function", 
            "WRITE A PYTHON FUNCTION",
            "Write A Python Function"
        ]
        
        for prompt in prompts:
            result = self.router.analyze_prompt(prompt)
            assert result == ModelType.CODER
    
    def test_pattern_matching(self):
        """Test regex pattern matching functionality."""
        
        # Code patterns
        code_patterns = [
            "def my_function():",
            "import numpy as np",
            "console.log('hello');",
            "if (condition) { return true; }",
            "// This is a comment",
            "Check out this GitHub repo: https://github.com/user/repo"
        ]
        
        for prompt in code_patterns:
            result = self.router.analyze_prompt(prompt)
            assert result == ModelType.CODER, f"Pattern not detected: {prompt}"
    
    def test_get_routing_keywords(self):
        """Test retrieval of routing keywords."""
        keywords = self.router.get_routing_keywords()
        
        assert ModelType.CODER in keywords
        assert ModelType.WRITER in keywords
        assert ModelType.GENERAL in keywords
        
        # Verify some expected keywords are present
        assert "python" in keywords[ModelType.CODER]
        assert "function" in keywords[ModelType.CODER]
        assert "essay" in keywords[ModelType.WRITER]
        assert "write" in keywords[ModelType.WRITER]
        assert "help" in keywords[ModelType.GENERAL]
    
    def test_add_custom_rule(self):
        """Test adding custom routing rules."""
        
        # Add custom rule for CODER model
        custom_keywords = ["tensorflow", "pytorch", "machine learning"]
        custom_patterns = [r"\\bML\\b", r"\\bAI\\b"]
        
        initial_rules_count = len(self.router.routing_rules)
        self.router.add_custom_rule(
            ModelType.CODER, 
            custom_keywords, 
            custom_patterns, 
            weight=1.5
        )
        
        # Verify rule was added
        assert len(self.router.routing_rules) == initial_rules_count + 1
        
        # Test that custom rule works
        result = self.router.analyze_prompt("I want to learn about TensorFlow and ML")
        assert result == ModelType.CODER
    
    def test_explain_routing_decision(self):
        """Test detailed routing decision explanation."""
        
        prompt = "Write a Python function to process data"
        explanation = self.router.explain_routing_decision(prompt)
        
        # Verify explanation structure
        assert 'prompt' in explanation
        assert 'selected_model' in explanation
        assert 'rule_scores' in explanation
        assert 'matched_keywords' in explanation
        assert 'matched_patterns' in explanation
        
        # Verify content
        assert explanation['prompt'] == prompt
        assert explanation['selected_model'] == ModelType.CODER
        
        # Should have scores for all model types
        assert 'coder' in explanation['rule_scores']
        assert 'writer' in explanation['rule_scores']
        assert 'general' in explanation['rule_scores']
        
        # Should have matched keywords for coder
        coder_keywords = explanation['matched_keywords']['coder']
        assert 'python' in coder_keywords or 'function' in coder_keywords
    
    def test_calculate_rule_score(self):
        """Test rule score calculation."""
        
        # Create a test rule
        test_rule = RoutingRule(
            ModelType.CODER,
            keywords=["python", "function", "code"],
            patterns=[r"\\bdef\\b", r"\\bimport\\b"],
            weight=1.0
        )
        
        # Test prompt with matches
        prompt = "write a python function with def statement"
        score = self.router._calculate_rule_score(prompt, test_rule)
        
        # Should have positive score due to keyword and pattern matches
        assert score > 0
        
        # Test prompt with no matches
        prompt_no_match = "tell me about cooking recipes"
        score_no_match = self.router._calculate_rule_score(prompt_no_match, test_rule)
        
        # Should have zero or very low score
        assert score_no_match < score
    
    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        
        # Very long prompt
        long_prompt = "write a python function " * 100
        result = self.router.analyze_prompt(long_prompt)
        assert result == ModelType.CODER
        
        # Prompt with special characters
        special_chars = "Write a function with symbols: @#$%^&*()[]{}|\\:;\"'<>,.?/"
        result = self.router.analyze_prompt(special_chars)
        assert result == ModelType.CODER
        
        # Prompt with numbers
        numbers_prompt = "Create a function that processes 12345 items"
        result = self.router.analyze_prompt(numbers_prompt)
        assert result == ModelType.CODER
        
        # Prompt with URLs
        url_prompt = "Check this code at https://github.com/example/repo"
        result = self.router.analyze_prompt(url_prompt)
        assert result == ModelType.CODER
    
    def test_routing_consistency(self):
        """Test that routing decisions are consistent across multiple calls."""
        
        test_prompts = [
            "Write a Python function for data analysis",
            "Create a blog post about technology trends", 
            "What is the meaning of life?",
            "Debug this JavaScript error",
            "Compose a professional email"
        ]
        
        # Test each prompt multiple times
        for prompt in test_prompts:
            results = [self.router.analyze_prompt(prompt) for _ in range(5)]
            
            # All results should be the same
            assert len(set(results)) == 1, f"Inconsistent routing for: {prompt}"


if __name__ == "__main__":
    pytest.main([__file__])