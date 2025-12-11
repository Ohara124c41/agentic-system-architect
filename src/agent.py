"""
LangGraph Workflow Definition - Orchestrator Pattern

Main agent orchestration using LangGraph state machine.
Implements the architecture governance workflow.

The ArchitectureGovernanceSystem class serves as the ORCHESTRATOR that coordinates
five specialized agents:

    Orchestrator (this module)
        ├── InterceptorAgent      - Request parsing, technology classification
        ├── WhyValidatorAgent     - INCOSE Five Whys requirements extraction
        ├── EvaluatorAgent        - Deterministic matching (no LLM)
        ├── RecommenderAgent      - Alternative suggestions
        └── IlitiesAnalystAgent   - Trade-off analysis (architect's second hat)

The orchestrator handles:
- Agent initialization and configuration
- Workflow definition via LangGraph StateGraph
- Conditional routing between agents
- Session state management via MemorySaver
- Human-in-the-loop approval gateway
"""

import os
import uuid
from typing import Optional, Literal
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from .schemas.state import AgentState, create_initial_state, ConversationTurn
from .agents.interceptor import InterceptorAgent
from .agents.why_validator import WhyValidatorAgent
from .agents.evaluator import EvaluatorAgent
from .agents.recommender import RecommenderAgent
from .agents.ilities_analyst import IlitiesAnalystAgent

# Load environment variables
load_dotenv()


def generate_session_id() -> str:
    """Generate a unique session ID using UUID."""
    return f"session_{uuid.uuid4().hex[:12]}"


def create_llm() -> ChatOpenAI:
    """
    Create the LLM instance with appropriate configuration.

    Supports both Vocareum (for development) and standard OpenAI (for production).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    temperature = float(os.getenv("TEMPERATURE", "0.1"))

    # ==========================================================================
    # VOCAREUM CONFIGURATION (Active - for development with Udacity credits)
    # ==========================================================================
    if api_base and "vocareum" in api_base.lower():
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )

    # ==========================================================================
    # STANDARD OPENAI CONFIGURATION (for production/public release)
    # ==========================================================================
    # Uncomment below and comment out Vocareum section above for production
    # return ChatOpenAI(
    #     model=model_name,
    #     temperature=temperature,
    #     openai_api_key=api_key,
    #     # No api_base needed for standard OpenAI
    # )

    # Default: Standard OpenAI
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base,  # Will be None if not set
    )


class ArchitectureGovernanceSystem:
    """
    Orchestrator for the Architecture Governance System.

    This class implements the ORCHESTRATOR PATTERN, coordinating five specialized
    agents via a LangGraph state machine. A central coordinator manages subsystem
    agents with potentially conflicting optimization priorities.

    Orchestrator Responsibilities:
    1. Agent Initialization - Creates and configures all specialized agents
    2. Workflow Definition - Builds the LangGraph StateGraph with conditional edges
    3. Routing Logic - Determines agent sequence based on state
    4. Session Management - Handles start_session() and continue_session()
    5. State Persistence - Uses MemorySaver for stateful conversations

    The five specialized agents:
    - Interceptor: Request parsing and technology classification
    - Why Validator: INCOSE Five Whys requirements extraction
    - Evaluator: Deterministic technology-requirement matching
    - Recommender: Alternative technology suggestions
    - Ilities Analyst: Quality attribute trade-off analysis
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the governance system.

        Args:
            llm: Optional LangChain ChatOpenAI instance. If not provided,
                 creates one using environment configuration.
        """
        self.llm = llm or create_llm()

        # Initialize agents
        self.interceptor = InterceptorAgent(self.llm)
        self.why_validator = WhyValidatorAgent(self.llm, max_whys=5)  # Five Whys methodology
        self.evaluator = EvaluatorAgent(self.llm)
        self.recommender = RecommenderAgent(self.llm)
        self.ilities_analyst = IlitiesAnalystAgent(self.llm)  # Architect's second hat

        # Build the workflow
        self.workflow = self._build_workflow()

        # Memory for stateful conversations
        self.memory = MemorySaver()

        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes (agents)
        workflow.add_node("router", self._router_node)  # Entry router for session continuation
        workflow.add_node("intercept", self._intercept_node)
        workflow.add_node("clarify", self._clarification_node)  # Handle unclear requests
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("recommend", self._recommend_node)
        workflow.add_node("analyze_ilities", self._ilities_node)  # Architect's second hat
        workflow.add_node("approve", self._approval_node)

        # Set entry point to router (decides where to go based on state)
        workflow.set_entry_point("router")

        # Router decides where to start based on session state
        workflow.add_conditional_edges(
            "router",
            self._route_from_router,
            {
                "intercept": "intercept",   # New session
                "validate": "validate",      # Continue with validation
            }
        )

        # Conditional edge from intercept - route based on technology identification
        workflow.add_conditional_edges(
            "intercept",
            self._route_from_intercept,
            {
                "validate": "validate",    # Technology identified
                "clarify": "clarify",      # Need clarification
            }
        )

        # Clarification leads to await input
        workflow.add_edge("clarify", END)

        # Conditional edge from validate - may loop back for more whys
        workflow.add_conditional_edges(
            "validate",
            self._route_from_validate,
            {
                "validate": "validate",  # Loop for more whys
                "evaluate": "evaluate",   # Proceed to evaluation
                "await_input": END,       # Wait for user input
            }
        )

        # Conditional edge from evaluate
        workflow.add_conditional_edges(
            "evaluate",
            self._route_from_evaluate,
            {
                "recommend": "recommend",
                "analyze_ilities": "analyze_ilities",  # Good match still gets ilities analysis
            }
        )

        # After recommendation, analyze quality attribute trade-offs
        workflow.add_edge("recommend", "analyze_ilities")
        workflow.add_edge("analyze_ilities", "approve")
        workflow.add_edge("approve", END)

        return workflow

    def _router_node(self, state: AgentState) -> AgentState:
        """Router node - pass through, routing logic is in conditional edge."""
        return state

    def _route_from_router(self, state: AgentState) -> Literal["intercept", "validate"]:
        """Route from router based on session state."""
        # If we already have a technology identified and are in why iteration, continue validation
        if state.get("technology_requested") and state.get("why_iteration", 0) > 0:
            return "validate"
        # Otherwise, start fresh with interception
        return "intercept"

    def _intercept_node(self, state: AgentState) -> AgentState:
        """Interceptor node function."""
        return self.interceptor.process(state)

    def _clarification_node(self, state: AgentState) -> AgentState:
        """Handle cases where technology couldn't be identified."""
        now = datetime.now().isoformat()

        clarification_msg = """I'd like to help ensure you choose the right technology, but I need a bit more information.

Could you clarify:
1. What specific technology or type of solution are you considering?
2. What problem are you trying to solve?

For example: "I want to use MongoDB for storing user data" or "I need a machine learning model for image classification"."""

        state["conversation_history"].append(
            ConversationTurn(
                role="agent",
                content=clarification_msg,
                timestamp=now,
                agent_name="clarification"
            )
        )

        state["awaiting_user_response"] = True
        state["current_agent"] = "clarification"
        state["updated_at"] = now

        return state

    def _validate_node(self, state: AgentState) -> AgentState:
        """Why Validator node function."""
        if state.get("why_iteration", 0) == 0:
            return self.why_validator.process_initial(state)
        else:
            # Get latest user response
            history = state.get("conversation_history", [])
            for turn in reversed(history):
                if turn["role"] == "user":
                    return self.why_validator.process_response(state, turn["content"])
            return state

    def _evaluate_node(self, state: AgentState) -> AgentState:
        """Evaluator node function."""
        return self.evaluator.process(state)

    def _recommend_node(self, state: AgentState) -> AgentState:
        """Recommender node function."""
        return self.recommender.process(state)

    def _ilities_node(self, state: AgentState) -> AgentState:
        """Ilities Analyst node function - architect's second hat for trade-off analysis."""
        return self.ilities_analyst.process(state)

    def _approval_node(self, state: AgentState) -> AgentState:
        """Human approval node function."""
        now = datetime.now().isoformat()

        # Build approval summary
        summary = self._build_approval_summary(state)

        state["conversation_history"].append(
            ConversationTurn(
                role="agent",
                content=summary,
                timestamp=now,
                agent_name="approval"
            )
        )

        state["workflow_complete"] = True
        state["current_agent"] = "approval"
        state["updated_at"] = now

        return state

    def _route_from_intercept(self, state: AgentState) -> Literal["validate", "clarify"]:
        """Route from intercept node based on whether technology was identified."""
        if state.get("technology_requested"):
            return "validate"
        return "clarify"

    def _route_from_validate(self, state: AgentState) -> Literal["validate", "evaluate", "await_input"]:
        """Route from validate node."""
        if state.get("awaiting_user_response", False):
            return "await_input"
        if state.get("next_agent") == "evaluator":
            return "evaluate"
        return "validate"

    def _route_from_evaluate(self, state: AgentState) -> Literal["recommend", "analyze_ilities"]:
        """Route from evaluate node."""
        if state.get("match_status") in ["partial", "mismatch"]:
            return "recommend"
        # Even good matches go through ilities analysis for trade-off awareness
        return "analyze_ilities"

    def _build_approval_summary(self, state: AgentState) -> str:
        """Build the final approval summary."""
        original = state.get("technology_requested", "Unknown")
        recommended = state.get("recommended_technology", original)
        match_score = state.get("match_score", 0)
        status = state.get("match_status", "unknown")

        # Requirements summary
        requirements = state.get("extracted_requirements", [])
        req_summary = ", ".join(requirements[:5]) if requirements else "Not specified"

        # Build summary
        summary = f"""
════════════════════════════════════════════════════════════════════════════════
                    ARCHITECTURE GOVERNANCE SUMMARY
════════════════════════════════════════════════════════════════════════════════

📋 ORIGINAL REQUEST
   Technology: {original}
   Category: {state.get('technology_category', 'Unknown')}

📊 REQUIREMENTS IDENTIFIED (via Five Whys)
   {req_summary}

⚖️  EVALUATION
   Match Score: {match_score:.0%}
   Status: {status.upper()}
"""

        if original != recommended:
            summary += f"""
💡 RECOMMENDATION
   Suggested Alternative: {recommended}
   Rationale: {state.get('recommendation_rationale', 'Better alignment with requirements')}
"""

        mismatches = state.get("mismatches", [])
        if mismatches:
            summary += f"""
⚠️  CONCERNS
   {chr(10).join('   - ' + m for m in mismatches[:5])}
"""

        summary += """
════════════════════════════════════════════════════════════════════════════════
📝 DECISION REQUIRED

Would you like to:
  [1] Proceed with """ + recommended + """
  [2] Proceed with original choice (""" + original + """)
  [3] Discuss further

════════════════════════════════════════════════════════════════════════════════
"""

        return summary

    def start_session(self, user_request: str, session_id: Optional[str] = None) -> AgentState:
        """
        Start a new governance session with a user request.

        Args:
            user_request: The user's technology request
            session_id: Optional session identifier

        Returns:
            Initial agent state after interceptor processing
        """
        initial_state = create_initial_state(user_request, session_id)

        # Run the workflow until it needs user input
        config = {"configurable": {"thread_id": initial_state["session_id"]}}

        result = self.app.invoke(initial_state, config)

        return result

    def continue_session(self, state: AgentState, user_response: str) -> AgentState:
        """
        Continue a session with user's response to a why question.

        Args:
            state: Current session state
            user_response: User's response to the why question

        Returns:
            Updated agent state
        """
        now = datetime.now().isoformat()

        # Add user response to conversation
        state["conversation_history"].append(
            ConversationTurn(
                role="user",
                content=user_response,
                timestamp=now,
                agent_name=None
            )
        )
        state["awaiting_user_response"] = False

        # Continue the workflow
        config = {"configurable": {"thread_id": state["session_id"]}}

        result = self.app.invoke(state, config)

        return result

    def get_current_question(self, state: AgentState) -> Optional[str]:
        """Get the current question awaiting response."""
        if not state.get("awaiting_user_response", False):
            return None

        # Get last agent message
        for turn in reversed(state.get("conversation_history", [])):
            if turn["role"] == "agent":
                return turn["content"]

        return None

    def get_conversation_history(self, state: AgentState) -> str:
        """Get formatted conversation history."""
        history = state.get("conversation_history", [])
        formatted = []

        for turn in history:
            role = turn["role"].upper()
            agent = f" ({turn.get('agent_name', '')})" if turn.get("agent_name") else ""
            formatted.append(f"[{role}{agent}]: {turn['content']}")

        return "\n\n".join(formatted)


def create_governance_system(llm: Optional[ChatOpenAI] = None) -> ArchitectureGovernanceSystem:
    """
    Factory function to create the governance system.

    Usage:
        system = create_governance_system()
        state = system.start_session("I want to use MongoDB for my tabular sales data")
    """
    return ArchitectureGovernanceSystem(llm)
