"""
Enhanced prompt router using improved keyword matching and context analysis.
"""

import logging
import re
from typing import Dict, List, Set
from dataclasses import dataclass

from ..models import ModelType


@dataclass
class RoutingRule:
    """Configuration for a routing rule."""
    model_type: ModelType
    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0


class EnhancedRouter:
    """
    Enhanced prompt router with improved keyword matching, context analysis,
    and better task categorization for accurate model selection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_enhanced_rules()
    
    def _initialize_enhanced_rules(self) -> None:
        """Initialize enhanced routing rules with better categorization."""
        
        # CODING: Comprehensive coding keywords and patterns
        coding_keywords = [
            # Programming languages
            "python", "javascript", "java", "c++", "rust", "go", "php", "ruby",
            "typescript", "kotlin", "swift", "scala", "perl", "bash", "shell",
            
            # Programming concepts
            "function", "method", "class", "object", "variable", "array", "list",
            "dictionary", "hash", "map", "loop", "iteration", "recursion",
            "algorithm", "data structure", "binary tree", "linked list", "queue",
            "stack", "heap", "sorting", "searching", "binary search", "quicksort",
            
            # Development actions
            "code", "program", "programming", "script", "implement", "build",
            "create", "develop", "write code", "debug", "fix", "refactor",
            "optimize", "test", "unit test", "integration test",
            
            # Technical terms
            "api", "rest", "http", "json", "xml", "database", "sql", "nosql",
            "framework", "library", "module", "package", "import", "export",
            "compiler", "interpreter", "syntax", "semantics", "parsing",
            
            # Development tools
            "git", "github", "repository", "commit", "branch", "merge",
            "docker", "kubernetes", "deployment", "ci/cd", "devops",
            
            # Web development
            "html", "css", "react", "vue", "angular", "node", "express",
            "django", "flask", "laravel", "spring", "webapp", "frontend", "backend",
            
            # Keywords that clearly indicate coding
            "hello world", "print statement", "console output", "compile", "execute"
        ]
        
        coding_patterns = [
            # Function/method definitions
            r'\b(?:def\s+\w+|function\s+\w+|class\s+\w+|public\s+class)\b',
            r'\b(?:int|string|bool|float|double|char|void)\s+\w+',
            
            # Import/include statements
            r'\b(?:import\s+\w+|from\s+\w+\s+import|#include|require|using)\b',
            
            # Control structures
            r'\b(?:if\s*\(|else\s*\{|for\s*\(|while\s*\(|switch\s*\()\b',
            r'\breturn\s+\w+|\breturn\s*;',
            
            # Programming symbols
            r'[{}()\[\];]{2,}',  # Multiple programming symbols
            r'[=!<>]+',          # Comparison operators
            r'&&|\|\||==|!=|<=|>=',  # Logical operators
            
            # Comments
            r'(?://.*|/\*.*\*/|#.*|\"\"\"|\'\'\')',
            
            # File extensions
            r'\b\w+\.(?:py|js|java|cpp|c|h|php|rb|go|rs|kt|swift)\b',
            
            # URLs/repositories
            r'(?:https?://)?(?:github\.com|stackoverflow\.com|docs\.|api\.)',
            
            # Specific coding requests
            r'\bwrite\s+(?:a|an|the)?\s*(?:program|code|function|script|algorithm)\b',
            r'\bcreate\s+(?:a|an|the)?\s*(?:program|application|script|api)\b',
            r'\bimplement\s+(?:a|an|the)?\s*(?:algorithm|function|class|method)\b',
            r'\bdebug\s+(?:this|the|my)\s+code\b',
            r'\bfix\s+(?:this|the)\s+(?:code|bug|error)\b',
        ]
        
        # MATHEMATICS: Comprehensive math keywords and patterns  
        math_keywords = [
            # Basic operations
            "calculate", "compute", "solve", "find", "determine", "evaluate",
            "add", "subtract", "multiply", "divide", "sum", "difference", "product", "quotient",
            
            # Math concepts
            "equation", "formula", "expression", "function", "variable",
            "coefficient", "constant", "theorem", "proof", "lemma",
            
            # Algebra
            "algebra", "polynomial", "quadratic", "linear", "exponential",
            "logarithm", "root", "factor", "expand", "simplify",
            
            # Geometry
            "geometry", "triangle", "circle", "square", "rectangle", "polygon",
            "area", "perimeter", "volume", "surface area", "angle", "degree",
            "radius", "diameter", "circumference", "hypotenuse",
            
            # Calculus
            "calculus", "derivative", "integral", "differentiate", "integrate",
            "limit", "continuity", "optimization", "maximum", "minimum",
            
            # Statistics
            "statistics", "probability", "mean", "median", "mode", "average",
            "standard deviation", "variance", "correlation", "regression",
            "distribution", "normal", "binomial", "chi-square",
            
            # Numbers
            "number", "integer", "fraction", "decimal", "percentage", "percent",
            "ratio", "proportion", "prime", "composite", "factorial", "fibonacci",
            
            # Mathematical symbols (written out)
            "plus", "minus", "times", "divided by", "equals", "greater than",
            "less than", "infinity", "pi", "euler", "sine", "cosine", "tangent"
        ]
        
        math_patterns = [
            # Mathematical expressions
            r'\b\d+\s*[+\-*/=]\s*\d+',  # Simple arithmetic
            r'\b\w+\s*[+\-*/=]\s*\w+',  # Algebraic expressions
            r'\b[xy]\s*[=+\-*/]\s*\d+', # Variables with numbers
            
            # Equations
            r'\b\w+\s*=\s*\d+',        # x = 5
            r'\b\d+x\s*[+\-]\s*\d+',   # 2x + 3
            r'\bx\^?\d+',               # x^2, x2
            
            # Mathematical functions
            r'\b(?:sin|cos|tan|log|ln|exp|sqrt|abs)\s*\(',
            r'\bf\(x\)|\bg\(x\)|\by\s*=',
            
            # Common math phrases
            r'\bwhat\s+is\s+\d+\s*[+\-*/]\s*\d+',
            r'\bsolve\s+(?:for\s+\w+|the\s+equation)',
            r'\bcalculate\s+(?:the\s+)?(?:area|volume|perimeter)',
            r'\bfind\s+the\s+(?:derivative|integral|limit|root)',
            r'\b\d+\s*%\s*of\s*\d+',   # percentage calculations
        ]
        
        # WRITING/GENERAL: Text generation and general tasks
        writing_keywords = [
            "write", "compose", "create", "draft", "author", "pen",
            "essay", "article", "story", "blog", "post", "content",
            "paragraph", "sentence", "chapter", "section",
            "narrative", "character", "plot", "theme", "setting",
            "analysis", "review", "critique", "summary", "synopsis",
            "report", "proposal", "presentation", "document",
            "letter", "email", "memo", "notice", "announcement",
            "creative", "fiction", "non-fiction", "poetry", "prose",
            "grammar", "style", "tone", "voice", "audience",
            "edit", "revise", "proofread", "publish", "manuscript"
        ]
        
        writing_patterns = [
            r'\bwrite\s+(?:a|an|the)\s+(?:essay|story|article|blog|letter|email)\b',
            r'\bcompose\s+(?:a|an|the)\s+(?:poem|song|story|message)\b',
            r'\bcreate\s+(?:a|an|the)\s+(?:narrative|character|plot)\b',
            r'\btell\s+me\s+(?:a|about|how)',
            r'\bexplain\s+(?:the|how|why|what)',
            r'\bdescribe\s+(?:the|how|what)',
            r'\bwhat\s+(?:is|are|do|does|would|should)',
        ]
        
        # Store routing rules with adjusted weights
        self.routing_rules = [
            RoutingRule(ModelType.CODER, coding_keywords, coding_patterns, 1.5),      # High priority
            RoutingRule(ModelType.MATH, math_keywords, math_patterns, 1.3),         # Math uses MATH type
            RoutingRule(ModelType.WRITER, writing_keywords, writing_patterns, 1.0)   # Standard priority
        ]
        
        self.logger.info("Initialized enhanced routing rules with improved categorization")
    
    def analyze_prompt(self, prompt: str) -> ModelType:
        """
        Analyze prompt using enhanced keyword matching and context analysis.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            ModelType: Recommended model type based on analysis
        """
        if not prompt or not prompt.strip():
            self.logger.warning("Empty prompt provided, defaulting to WRITER")
            return ModelType.WRITER
        
        prompt_lower = prompt.lower().strip()
        scores = {}
        
        # Calculate scores for each model type
        for rule in self.routing_rules:
            score = 0.0
            
            # Keyword matching with context awareness
            for keyword in rule.keywords:
                if keyword in prompt_lower:
                    # Boost score for exact matches
                    if f" {keyword} " in f" {prompt_lower} ":
                        score += 2.0
                    else:
                        score += 1.0
            
            # Pattern matching
            for pattern in rule.patterns:
                matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
                score += len(matches) * 1.5
            
            # Apply rule weight
            scores[rule.model_type] = score * rule.weight
        
        # Special context-aware adjustments
        if any(word in prompt_lower for word in ["calculate", "solve", "equation", "math", "formula"]):
            scores[ModelType.MATH] = scores.get(ModelType.MATH, 0) + 3.0
            
        if any(word in prompt_lower for word in ["function", "code", "program", "implement", "algorithm"]):
            scores[ModelType.CODER] = scores.get(ModelType.CODER, 0) + 3.0
        
        # OVERRIDE: "Write a [language] function" patterns should ALWAYS be coding
        coding_override_patterns = [
            r'\bwrite\s+(?:a|an)?\s*(?:python|javascript|java|c\+\+|rust|go|php|ruby)\s+function\b',
            r'\bcreate\s+(?:a|an)?\s*(?:python|javascript|java|c\+\+|rust|go|php|ruby)\s+function\b',
            r'\bimplement\s+(?:a|an)?\s*(?:python|javascript|java|c\+\+|rust|go|php|ruby)\s+function\b'
        ]
        
        for pattern in coding_override_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                scores[ModelType.CODER] = scores.get(ModelType.CODER, 0) + 10.0  # STRONG override
                self.logger.info("OVERRIDE: Detected 'write [language] function' - forcing CODER selection")
        
        # Find the best match
        if not scores or max(scores.values()) == 0:
            self.logger.info("No clear category detected, defaulting to WRITER")
            return ModelType.WRITER
        
        best_type = max(scores.items(), key=lambda x: x[1])
        selected_type = best_type[0]
        confidence = best_type[1]
        
        self.logger.info(f"Enhanced analysis: {dict(scores)}")
        self.logger.info(f"Selected {selected_type.value} with score {confidence:.1f}")
        
        return selected_type

__all__ = ['EnhancedRouter']
