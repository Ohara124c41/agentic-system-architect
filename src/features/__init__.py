"""
Advanced Features for Agentic Architecture Governance System

Implements IEEE Software paper innovations:
- Decision Traceability Graph
- Architecture Decision Records (ADR)
- Agent Thinking Layer with Technology Network
- Multi-Stakeholder Mode
- Cost-Benefit Analysis
- Subsystem Trade-off Analysis
"""

from .traceability import DecisionTraceabilityGraph
from .adr_generator import ADRGenerator
from .thinking_layer import AgentThinkingLayer, TechnologyNetwork
from .stakeholder_mode import MultiStakeholderMode, StakeholderRole
from .cost_benefit import CostBenefitAnalyzer
from .cogent_tradeoffs import SubsystemTradeoffAnalyzer

__all__ = [
    "DecisionTraceabilityGraph",
    "ADRGenerator",
    "AgentThinkingLayer",
    "TechnologyNetwork",
    "MultiStakeholderMode",
    "StakeholderRole",
    "CostBenefitAnalyzer",
    "SubsystemTradeoffAnalyzer",
]
