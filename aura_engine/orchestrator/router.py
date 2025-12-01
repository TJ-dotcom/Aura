"""
Prompt router for intelligent model selection based on content analysis.
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


class PromptRouter:
    """
    Intelligent prompt router that analyzes prompt content to determine
    the most appropriate specialized model for the task.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_routing_rules()
    
    def _initialize_routing_rules(self) -> None:
        """Initialize routing rules for different model types."""
        
        # Coding-related keywords and patterns
        coding_keywords = [
            "function", "code", "python", "javascript", "java", "c++", "rust",
            "algorithm", "debug", "implement", "class", "method", "variable",
            "loop", "if", "else", "return", "import", "library", "framework",
            "api", "database", "sql", "json", "xml", "html", "css", "react",
            "node", "django", "flask", "git", "github", "repository", "commit",
            "branch", "merge", "pull request", "bug", "error", "exception",
            "test", "unit test", "integration", "deployment", "docker", "kubernetes",
            "script", "automation", "regex", "parse", "serialize", "encrypt",
            "hash", "authentication", "authorization", "security", "performance",
            "optimization", "refactor", "clean code", "design pattern",
            "architecture", "microservice", "monolith", "scalability",
            # Essential coding words that were missing:
            "program", "programming", "software", "application", "app", "compile",
            "execute", "run", "syntax", "logic", "coding", "development", "dev",
            "hello world", "hello, world", "print statement", "console output"
        ]
        
        coding_patterns = [
            r'\b(?:def\s+\w+|function\s+\w+|class\s+\w+)\b',  # More specific function/class definitions
            r'\b(?:import\s+\w+|from\s+\w+\s+import)\b',  # Import statements
            r'\b(?:return\s+\w+|if\s+\w+|else:|for\s+\w+\s+in|while\s+\w+)\b',  # Control structures
            r'\b(?:print|console\.log|System\.out\.println)\s*\(',
            r'[{}()\[\];]{2,}',  # Multiple programming symbols together
            r'\b(?:var|let|const|int|string|bool|float|double)\s+\w+',  # Variable declarations
            r'(?://|#|/\*|\*/)',  # Comment patterns
            r'\b(?:\.py|\.js|\.java|\.cpp|\.rs|\.go|\.php)\b',  # File extensions
            r'(?:https?://)?(?:github\.com|stackoverflow\.com|docs\.python\.org)',
            # Context-aware patterns for coding requests:
            r'\bwrite\s+(?:a|an|the)?\s*(?:program|code|function|script|algorithm)\b',
            r'\bcreate\s+(?:a|an|the)?\s*(?:program|code|function|script|application)\b',
            r'\bhello\s+world\b',  # Classic programming example
            r'\bfix\s+(?:this|the)\s+code\b',
            r'\bdebug\s+(?:this|the)\b',
        ]
        
        # Writing-related keywords and patterns
        writing_keywords = [
            "write", "essay", "story", "article", "blog", "post", "content",
            "paragraph", "sentence", "grammar", "style", "tone", "voice",
            "narrative", "character", "plot", "theme", "analysis", "review",
            "summary", "report", "proposal", "letter", "email", "memo",
            "creative", "fiction", "non-fiction", "poetry", "prose", "draft",
            "edit", "revise", "proofread", "publish", "audience", "reader",
            "introduction", "conclusion", "thesis", "argument", "evidence",
            "research", "citation", "reference", "bibliography", "academic",
            "professional", "business", "marketing", "copywriting", "journalism",
            "compose", "poem", "copy", "manuscript", "document", "text"
        ]
        
        writing_patterns = [
            r'\b(?:write|compose|draft|create)\s+(?:a|an|the)\s+(?:essay|story|article|blog|letter|email|report)\b',
            r'\b(?:tell me about|explain|describe|summarize|analyze)\b',
            r'\b(?:in your opinion|what do you think|how would you)\b',
            r'\b(?:once upon a time|dear|sincerely|best regards)\b',
        ]
        
        # General/conversational keywords
        general_keywords = [
            "hello", "hi", "help", "question", "answer", "explain", "what",
            "how", "why", "when", "where", "who", "tell me", "can you",
            "please", "thank you", "advice", "suggestion", "recommendation",
            "opinion", "think", "believe", "feel", "understand", "know",
            "learn", "teach", "show", "demonstrate", "example", "instance"
        ]
        
        # Mathematics keywords and patterns
        math_keywords = [
            "calculate", "solve", "equation", "formula", "mathematics", "math",
            "algebra", "calculus", "geometry", "trigonometry", "statistics",
            "probability", "derivative", "integral", "matrix", "vector",
            "polynomial", "quadratic", "linear", "exponential", "logarithm",
            "sine", "cosine", "tangent", "theorem", "proof", "number",
            "addition", "subtraction", "multiplication", "division",
            "fraction", "decimal", "percentage", "ratio", "proportion",
            "sum", "product", "difference", "quotient", "square", "cube",
            "root", "power", "exponent", "factorial", "permutation",
            "combination", "mean", "median", "mode", "variance", "deviation"
        ]
        
        math_patterns = [
            r'\b(?:what is|calculate|solve|find)\s+\d+[\+\-\*/\^]\d+',  # Basic arithmetic
            r'\b\d+\s*[\+\-\*/\^=]\s*\d+',  # Math expressions
            r'\b(?:x|y|z)\s*[=+\-*/^]\s*\d+',  # Algebraic expressions
            r'\b(?:sin|cos|tan|log|ln|sqrt)\s*\(',  # Math functions
            r'\b(?:derivative|integral|limit|sum)\b',  # Calculus terms
            r'\b\d+x²\s*[+\-]\s*\d+x\s*[+\-]\s*\d+\s*=\s*0\b',  # Quadratic equations
            r'\b(?:theorem|proof|equation|formula)\b',
            r'∫|∑|∏|√|±|≤|≥|≠|∞',  # Math symbols
        ]

        general_patterns = [
            r'\b(?:what is|what are|how do|how can|why do|why is)\b',
            r'\b(?:can you|could you|would you|will you)\b',
            r'\?\s*$',  # Questions ending with ?
        ]
        
        # Store routing rules with proper weighting
        self.routing_rules = [
            RoutingRule(ModelType.MATH, math_keywords, math_patterns, 1.5),      # Highest priority for math
            RoutingRule(ModelType.CODER, coding_keywords, coding_patterns, 1.2),
            RoutingRule(ModelType.WRITER, writing_keywords, writing_patterns, 1.1),
            RoutingRule(ModelType.GENERAL, general_keywords, general_patterns, 0.3)  # Much lower weight for general
        ]
        
        self.logger.info("Initialized routing rules for model selection")
    
    def analyze_prompt(self, prompt: str) -> ModelType:
        """
        Analyze prompt content and determine the most appropriate model type.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            ModelType: Recommended model type for the prompt
        """
        if not prompt or not prompt.strip():
            self.logger.warning("Empty prompt provided, defaulting to GENERAL model")
            return ModelType.GENERAL
        
        prompt_lower = prompt.lower().strip()
        
        # Calculate scores for each model type
        scores = {}
        for rule in self.routing_rules:
            score = self._calculate_rule_score(prompt_lower, rule)
            scores[rule.model_type] = score
        
        # Find the model type with the highest score
        best_model = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_model]
        
        # If no significant match found, default to GENERAL
        if best_score == 0:
            self.logger.info("No specific patterns detected, using GENERAL model")
            return ModelType.GENERAL
        
        self.logger.info(f"Prompt analysis complete: {best_model.value} (score: {best_score:.2f})")
        self.logger.debug(f"All scores: {[(k.value, v) for k, v in scores.items()]}")
        
        return best_model
    
    def _calculate_rule_score(self, prompt: str, rule: RoutingRule) -> float:
        """
        Calculate the score for a specific routing rule against the prompt.
        
        Args:
            prompt: Lowercase prompt text
            rule: Routing rule to evaluate
            
        Returns:
            float: Score for this rule (higher = better match)
        """
        score = 0.0
        
        # Score based on keyword matches - count actual matches, not percentage
        keyword_matches = 0
        for keyword in rule.keywords:
            if keyword.lower() in prompt:
                keyword_matches += 1
        
        # Give higher weight to keyword matches
        keyword_score = keyword_matches * rule.weight
        score += keyword_score
        
        # Score based on pattern matches
        pattern_matches = 0
        for pattern in rule.patterns:
            try:
                if re.search(pattern, prompt, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                self.logger.warning(f"Invalid regex pattern: {pattern}")
                continue
        
        # Pattern matches get bonus points
        pattern_score = pattern_matches * rule.weight * 2.0  # Patterns are strong indicators
        score += pattern_score
        
        return score
    
    def get_routing_keywords(self) -> Dict[ModelType, List[str]]:
        """
        Get the routing keywords for each model type.
        
        Returns:
            Dict[ModelType, List[str]]: Keywords for each model type
        """
        keywords_by_type = {}
        for rule in self.routing_rules:
            keywords_by_type[rule.model_type] = rule.keywords.copy()
        
        return keywords_by_type
    
    def add_custom_rule(self, model_type: ModelType, keywords: List[str], 
                       patterns: List[str] = None, weight: float = 1.0) -> None:
        """
        Add a custom routing rule.
        
        Args:
            model_type: Target model type
            keywords: List of keywords to match
            patterns: List of regex patterns to match
            weight: Weight for this rule
        """
        patterns = patterns or []
        custom_rule = RoutingRule(model_type, keywords, patterns, weight)
        self.routing_rules.append(custom_rule)
        
        self.logger.info(f"Added custom routing rule for {model_type.value}")
    
    def explain_routing_decision(self, prompt: str) -> Dict:
        """
        Provide detailed explanation of routing decision for debugging.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Dict: Detailed analysis results
        """
        prompt_lower = prompt.lower().strip()
        
        analysis = {
            'prompt': prompt,
            'selected_model': self.analyze_prompt(prompt),
            'rule_scores': {},
            'matched_keywords': {},
            'matched_patterns': {}
        }
        
        for rule in self.routing_rules:
            score = self._calculate_rule_score(prompt_lower, rule)
            analysis['rule_scores'][rule.model_type.value] = score
            
            # Find matched keywords
            matched_keywords = [kw for kw in rule.keywords if kw.lower() in prompt_lower]
            analysis['matched_keywords'][rule.model_type.value] = matched_keywords
            
            # Find matched patterns
            matched_patterns = []
            for pattern in rule.patterns:
                try:
                    if re.search(pattern, prompt_lower, re.IGNORECASE):
                        matched_patterns.append(pattern)
                except re.error:
                    continue
            analysis['matched_patterns'][rule.model_type.value] = matched_patterns
        
        return analysis