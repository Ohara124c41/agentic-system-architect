"""
Agent implementations for the Architecture Governance System.

This module contains five specialized agents coordinated by the Orchestrator
(ArchitectureGovernanceSystem in src/agent.py):

1. InterceptorAgent - Request parsing and technology classification
2. WhyValidatorAgent - INCOSE Five Whys requirements extraction
3. EvaluatorAgent - Deterministic technology-requirement matching
4. RecommenderAgent - Alternative technology suggestions
5. IlitiesAnalystAgent - Quality attribute trade-off analysis (architect's second hat)

The orchestrator pattern coordinates these specialized subsystem agents,
managing potentially conflicting optimization priorities.
"""

from .interceptor import InterceptorAgent
from .why_validator import WhyValidatorAgent
from .evaluator import EvaluatorAgent
from .recommender import RecommenderAgent
from .ilities_analyst import IlitiesAnalystAgent

__all__ = [
    "InterceptorAgent",
    "WhyValidatorAgent",
    "EvaluatorAgent",
    "RecommenderAgent",
    "IlitiesAnalystAgent",
]
