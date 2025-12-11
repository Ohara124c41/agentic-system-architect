"""
Agent State Definitions

Defines the state that flows through the LangGraph workflow.
Based on patterns from Udacity LangGraph projects.
"""

from typing import TypedDict, List, Optional, Literal
from datetime import datetime


class ConversationTurn(TypedDict):
    """A single turn in the conversation."""
    role: Literal["user", "agent", "system"]
    content: str
    timestamp: str
    agent_name: Optional[str]


class AgentState(TypedDict):
    """
    Central state object that flows through the LangGraph workflow.

    This state accumulates information as the request passes through
    each agent in the governance pipeline.
    """
    # Original request
    user_request: str
    session_id: str

    # Interceptor outputs
    technology_requested: Optional[str]
    technology_category: Optional[str]  # database, ml_model, architecture, devops, visualization, data_pipeline
    request_context: Optional[str]  # Additional context extracted from request

    # Why Validator outputs
    why_questions: List[str]  # Questions asked
    why_responses: List[str]  # User's answers
    why_iteration: int  # Current why number (1-3+)
    awaiting_user_response: bool  # True when waiting for user input

    # Requirements Profile (built from why responses)
    extracted_requirements: List[str]  # e.g., ["acid_transactions", "complex_joins", "tabular_data"]
    stated_reasons: List[str]  # User's stated reasons for technology choice
    implicit_needs: List[str]  # Inferred needs from context

    # Evaluator outputs
    match_score: float  # 0.0 to 1.0
    match_status: Optional[Literal["match", "partial", "mismatch"]]
    mismatches: List[str]  # Requirements that don't fit the technology
    matches: List[str]  # Requirements that do fit

    # Recommender outputs
    recommended_technology: Optional[str]
    alternatives: List[dict]  # List of {technology, rationale, fit_score}
    recommendation_rationale: Optional[str]

    # Ilities Analyst outputs (architect's second hat)
    ilities_analysis: Optional[dict]  # Trade-off analysis with risks, mitigations, triggers

    # Human-in-the-Loop
    human_approved: Optional[bool]
    human_feedback: Optional[str]

    # Workflow control
    current_agent: str
    next_agent: Optional[str]
    workflow_complete: bool

    # Conversation history
    conversation_history: List[ConversationTurn]

    # Metadata
    created_at: str
    updated_at: str
    total_llm_calls: int

    # Error handling
    error: Optional[str]


def create_initial_state(user_request: str, session_id: Optional[str] = None) -> AgentState:
    """Create a fresh agent state for a new request."""
    now = datetime.now().isoformat()

    return AgentState(
        # Original request
        user_request=user_request,
        session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

        # Interceptor outputs
        technology_requested=None,
        technology_category=None,
        request_context=None,

        # Why Validator outputs
        why_questions=[],
        why_responses=[],
        why_iteration=0,
        awaiting_user_response=False,

        # Requirements Profile
        extracted_requirements=[],
        stated_reasons=[],
        implicit_needs=[],

        # Evaluator outputs
        match_score=0.0,
        match_status=None,
        mismatches=[],
        matches=[],

        # Recommender outputs
        recommended_technology=None,
        alternatives=[],
        recommendation_rationale=None,

        # Ilities Analyst outputs
        ilities_analysis=None,

        # Human-in-the-Loop
        human_approved=None,
        human_feedback=None,

        # Workflow control
        current_agent="interceptor",
        next_agent=None,
        workflow_complete=False,

        # Conversation history
        conversation_history=[
            ConversationTurn(
                role="user",
                content=user_request,
                timestamp=now,
                agent_name=None
            )
        ],

        # Metadata
        created_at=now,
        updated_at=now,
        total_llm_calls=0,

        # Error handling
        error=None,
    )
