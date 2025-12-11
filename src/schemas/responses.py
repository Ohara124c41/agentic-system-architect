"""
Pydantic Response Models

Structured outputs for each agent, ensuring type safety and validation.
These models are used with LangChain's structured output feature.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class TechnologyClassification(BaseModel):
    """Output from the Request Interceptor Agent."""

    technology_name: str = Field(
        description="The specific technology mentioned (e.g., 'MongoDB', 'CNN', 'Kubernetes')"
    )
    category: Literal[
        "database", "ml_model", "architecture", "devops", "visualization", "data_pipeline", "unknown"
    ] = Field(
        description="The category of technology being requested"
    )
    context: str = Field(
        description="Additional context about what the user wants to achieve"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the classification (0.0 to 1.0)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the request is too vague to classify"
    )


class WhyResponse(BaseModel):
    """Output from analyzing a user's response to a 'why' question."""

    understood_reason: str = Field(
        description="The user's stated reason, paraphrased for clarity"
    )
    extracted_requirements: List[str] = Field(
        description="Technical requirements extracted from this response (e.g., 'acid_transactions', 'horizontal_scaling')"
    )
    implicit_needs: List[str] = Field(
        description="Needs implied but not explicitly stated"
    )
    follow_up_needed: bool = Field(
        description="Whether another 'why' question would be valuable"
    )
    suggested_follow_up: Optional[str] = Field(
        default=None,
        description="Suggested follow-up question if needed"
    )


class RequirementsProfile(BaseModel):
    """Complete requirements profile built from why analysis."""

    all_requirements: List[str] = Field(
        description="All extracted technical requirements"
    )
    primary_needs: List[str] = Field(
        description="The most important requirements (top 3-5)"
    )
    constraints: List[str] = Field(
        description="Constraints mentioned (team size, budget, timeline, etc.)"
    )
    stated_preferences: List[str] = Field(
        description="User's stated preferences (may not be requirements)"
    )
    data_characteristics: Optional[str] = Field(
        default=None,
        description="Description of the data type/structure if mentioned"
    )
    scale_requirements: Optional[str] = Field(
        default=None,
        description="Scale/performance requirements if mentioned"
    )


class EvaluationResult(BaseModel):
    """Output from the Technology Evaluator Agent."""

    technology: str = Field(
        description="The technology being evaluated"
    )
    match_score: float = Field(
        ge=0.0, le=1.0,
        description="How well the technology matches requirements (0.0 to 1.0)"
    )
    status: Literal["match", "partial", "mismatch"] = Field(
        description="Overall match status"
    )
    matching_requirements: List[str] = Field(
        description="Requirements that this technology handles well"
    )
    mismatched_requirements: List[str] = Field(
        description="Requirements that this technology handles poorly"
    )
    risk_factors: List[str] = Field(
        description="Potential risks of using this technology for the stated requirements"
    )
    explanation: str = Field(
        description="Human-readable explanation of the evaluation"
    )


class AlternativeRecommendation(BaseModel):
    """A single alternative technology recommendation."""

    technology: str = Field(
        description="The recommended alternative technology"
    )
    fit_score: float = Field(
        ge=0.0, le=1.0,
        description="How well this alternative fits the requirements"
    )
    rationale: str = Field(
        description="Why this alternative is better suited"
    )
    tradeoffs: List[str] = Field(
        description="Trade-offs to consider with this alternative"
    )
    migration_complexity: Literal["low", "medium", "high"] = Field(
        description="How complex it would be to use this instead"
    )


class GovernanceDecision(BaseModel):
    """Final output from the governance system."""

    original_request: str = Field(
        description="The user's original technology request"
    )
    original_technology: str = Field(
        description="The technology originally requested"
    )
    decision: Literal["approve", "recommend_alternative", "needs_discussion"] = Field(
        description="The governance decision"
    )
    match_score: float = Field(
        description="How well the original request matches requirements"
    )
    recommended_technology: Optional[str] = Field(
        default=None,
        description="The recommended technology (if different from original)"
    )
    alternatives: List[AlternativeRecommendation] = Field(
        default=[],
        description="List of alternative technologies considered"
    )
    requirements_summary: str = Field(
        description="Summary of extracted requirements"
    )
    rationale: str = Field(
        description="Full explanation of the governance decision"
    )
    conversation_summary: str = Field(
        description="Summary of the why-question conversation"
    )
    human_approval_required: bool = Field(
        description="Whether human approval is needed before proceeding"
    )


class InterceptorOutput(BaseModel):
    """Structured output for the interceptor agent."""

    technology: str = Field(description="Identified technology name")
    category: str = Field(description="Technology category")
    use_case_context: str = Field(description="What the user wants to accomplish")
    data_hints: List[str] = Field(
        default=[],
        description="Any hints about data type/structure from the request"
    )


class WhyQuestionOutput(BaseModel):
    """Structured output for generating why questions."""

    question: str = Field(description="The why question to ask")
    question_focus: str = Field(
        description="What aspect this question probes (e.g., 'data_structure', 'scale', 'team_expertise')"
    )
    expected_insight: str = Field(
        description="What insight we hope to gain from the answer"
    )


class IlitiesAnalysis(BaseModel):
    """Output from the Ilities Analyst Agent - quality attribute trade-off analysis."""

    technology: str = Field(
        description="The technology being analyzed"
    )
    strong_ilities: List[str] = Field(
        description="Quality attributes this technology excels at"
    )
    weak_ilities: List[str] = Field(
        description="Quality attributes this technology struggles with"
    )
    risks: List[str] = Field(
        description="Key risks to consider with this technology choice"
    )
    mitigations: List[str] = Field(
        description="Recommended actions to mitigate identified risks"
    )
    evolution_triggers: List[str] = Field(
        description="Conditions that should trigger reconsidering this choice"
    )
    monitoring_recommendations: List[str] = Field(
        description="Metrics to monitor for early warning signs"
    )
    context_specific_risks: List[str] = Field(
        default=[],
        description="Risks specific to the user's stated requirements"
    )
    architect_caveat: str = Field(
        description="The architect's advisory message with trade-offs and caveats"
    )
