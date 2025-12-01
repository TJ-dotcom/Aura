"""
Semantic prompt router using sentence transformers for intelligent model selection.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import ModelType


class SemanticRouter:
    """
    Advanced prompt router using sentence embeddings and semantic similarity
    for more accurate task categorization and model selection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._load_model()
        self._initialize_task_templates()
    
    def _load_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            # Use a lightweight, fast model suitable for classification
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            self.model = None
    
    def _initialize_task_templates(self):
        """Initialize task templates for semantic matching."""
        
        # Coding task examples
        self.coding_templates = [
            "Write a Python function to calculate factorial",
            "Implement a binary search algorithm in Java",
            "Create a REST API endpoint using Flask",
            "Debug this JavaScript code with syntax errors",
            "Write a SQL query to find duplicate records",
            "Implement a quicksort algorithm in C++",
            "Create a React component for user authentication",
            "Write unit tests for this Python class",
            "Refactor this code to improve performance",
            "Implement a data structure like linked list",
            "Create a web scraper using BeautifulSoup",
            "Write a function to parse JSON data",
            "Implement error handling in this code",
            "Create a database schema for an e-commerce app",
            "Write a script to automate file processing",
            "Implement OAuth authentication flow",
            "Create a machine learning model training script",
            "Write a function to validate email addresses",
            "Implement caching mechanism for API calls",
            "Create a command-line interface tool"
        ]
        
        # Mathematics task examples  
        self.math_templates = [
            "Solve this quadratic equation: x² + 5x - 6 = 0",
            "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1",
            "Find the integral of sin(x) from 0 to π",
            "What is the sum of first 100 natural numbers?",
            "Calculate the area of a circle with radius 5",
            "Solve this system of linear equations",
            "Find the probability of getting heads in 3 coin tosses",
            "Calculate the standard deviation of this dataset",
            "What is 15% of 240?",
            "Find the greatest common divisor of 48 and 18",
            "Calculate the compound interest for 5 years",
            "Solve for x: 2^x = 64",
            "Find the slope of line passing through (2,3) and (4,7)",
            "Calculate the volume of a sphere with radius 3",
            "What is the value of sin(30°)?",
            "Find the roots of x² - 7x + 12 = 0",
            "Calculate the factorial of 8",
            "What is the median of [5, 2, 8, 1, 9, 3]?",
            "Find the limit of (x²-1)/(x-1) as x approaches 1",
            "Calculate the permutation of 5 objects taken 3 at a time"
        ]
        
        # General/Writing task examples
        self.general_templates = [
            "Write an essay about climate change",
            "Explain the concept of artificial intelligence",
            "Tell me a story about a brave knight",
            "Summarize the main points of this article",
            "Write a professional email to a client",
            "Describe the process of photosynthesis",
            "What are the benefits of renewable energy?",
            "Write a poem about the ocean",
            "Explain how democracy works",
            "Create a marketing plan for a new product",
            "Write a book review for this novel",
            "Describe the culture of Japan",
            "What is the history of the internet?",
            "Write a cover letter for a job application",
            "Explain the theory of evolution",
            "Describe how to cook pasta perfectly",
            "What are the causes of global warming?",
            "Write a speech about leadership",
            "Explain the difference between psychology and psychiatry",
            "Describe the benefits of meditation"
        ]
        
        # Pre-compute embeddings for all templates
        if self.model:
            try:
                self.coding_embeddings = self.model.encode(self.coding_templates)
                self.math_embeddings = self.model.encode(self.math_templates) 
                self.general_embeddings = self.model.encode(self.general_templates)
                self.logger.info("Pre-computed embeddings for all task templates")
            except Exception as e:
                self.logger.error(f"Failed to compute template embeddings: {e}")
                self.coding_embeddings = None
                self.math_embeddings = None
                self.general_embeddings = None
    
    def analyze_prompt(self, prompt: str) -> ModelType:
        """
        Analyze prompt using semantic similarity to determine the best model type.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            ModelType: Recommended model type based on semantic analysis
        """
        if not prompt or not prompt.strip():
            self.logger.warning("Empty prompt provided, defaulting to GENERAL")
            return ModelType.GENERAL
            
        # Fallback to keyword-based if sentence transformer failed to load
        if not self.model or self.coding_embeddings is None:
            self.logger.warning("Sentence transformer not available, using fallback classification")
            return self._fallback_classification(prompt)
        
        try:
            # Encode the input prompt
            prompt_embedding = self.model.encode([prompt])
            
            # Calculate similarities with each task category
            coding_similarity = cosine_similarity(prompt_embedding, self.coding_embeddings).mean()
            math_similarity = cosine_similarity(prompt_embedding, self.math_embeddings).mean()
            general_similarity = cosine_similarity(prompt_embedding, self.general_embeddings).mean()
            
            # Find the highest similarity
            similarities = {
                ModelType.CODER: coding_similarity,
                ModelType.GENERAL: math_similarity,  # Using GENERAL for math tasks since no MATH type exists
                ModelType.WRITER: general_similarity
            }
            
            # Select the category with highest similarity
            best_match = max(similarities.items(), key=lambda x: x[1])
            selected_type = best_match[0]
            confidence = best_match[1]
            
            # Log the decision
            self.logger.debug(f"Semantic analysis: Coding={coding_similarity:.3f}, Math={math_similarity:.3f}, General={general_similarity:.3f}")
            self.logger.info(f"Selected {selected_type.value} with confidence {confidence:.3f}")
            
            # If math has highest similarity, we need to handle this specially
            # since we don't have a dedicated MATH ModelType
            if math_similarity > coding_similarity and math_similarity > general_similarity:
                # For now, use GENERAL but log that it's a math task
                self.logger.info("Detected MATH task, using GENERAL model (consider adding dedicated MATH ModelType)")
                return ModelType.GENERAL
            
            return selected_type
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            return self._fallback_classification(prompt)
    
    def _fallback_classification(self, prompt: str) -> ModelType:
        """Fallback keyword-based classification if semantic analysis fails."""
        prompt_lower = prompt.lower()
        
        # Strong coding indicators
        coding_keywords = [
            "function", "code", "python", "javascript", "java", "c++", "algorithm",
            "implement", "class", "method", "debug", "program", "programming",
            "script", "api", "database", "sql", "html", "css", "react", "flask"
        ]
        
        # Strong math indicators  
        math_keywords = [
            "calculate", "solve", "equation", "integral", "derivative", "probability",
            "statistics", "algebra", "geometry", "trigonometry", "factorial", 
            "percentage", "sum", "average", "median", "standard deviation"
        ]
        
        # Count matches
        coding_matches = sum(1 for word in coding_keywords if word in prompt_lower)
        math_matches = sum(1 for word in math_keywords if word in prompt_lower)
        
        if coding_matches > math_matches and coding_matches > 0:
            return ModelType.CODER
        elif math_matches > 0:
            return ModelType.GENERAL  # Use GENERAL for math
        else:
            return ModelType.WRITER

__all__ = ['SemanticRouter']
