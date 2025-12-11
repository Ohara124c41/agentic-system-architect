"""
Technology Evaluator Agent

Evaluates how well the requested technology matches the extracted requirements.
Uses deterministic matching logic - minimal LLM calls needed.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI

from ..schemas.state import AgentState, ConversationTurn
from ..schemas.responses import EvaluationResult
from ..tools.matching import evaluate_technology_match, find_requirement_conflicts
from ..tools.knowledge_base import get_technology_profile


class EvaluatorAgent:
    """
    Technology Evaluator Agent

    Responsibilities:
    1. Compare extracted requirements against technology profile
    2. Calculate match score
    3. Identify specific mismatches and risks
    4. Route to recommender (mismatch) or approval (match)
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the evaluator agent.

        Args:
            llm: LangChain ChatOpenAI instance (rarely needed - mostly deterministic)
        """
        self.llm = llm
        self.name = "evaluator"
        self.match_threshold = 0.7
        self.mismatch_threshold = 0.4

    def process(self, state: AgentState) -> AgentState:
        """
        Evaluate the technology against extracted requirements.

        This is mostly deterministic - uses the technology profiles
        knowledge base for matching.
        """
        technology = state.get("technology_requested")
        requirements = state.get("extracted_requirements", [])
        constraints = self._extract_constraints(state)

        if not technology:
            # No technology to evaluate - this shouldn't happen if workflow is correct
            # But handle gracefully by going to approval with a warning
            state["error"] = "No technology identified to evaluate"
            state["match_score"] = 0.0
            state["match_status"] = "unknown"
            state["next_agent"] = "approval"  # Go to approval with warning
            state["recommended_technology"] = None
            return state

        # Perform deterministic evaluation (NO API CALL)
        evaluation = evaluate_technology_match(
            technology=technology,
            requirements=requirements,
            constraints=constraints
        )

        # Update state with evaluation results
        state["match_score"] = evaluation["match_score"]
        state["match_status"] = evaluation["status"]
        state["matches"] = evaluation["matches"]
        state["mismatches"] = evaluation["mismatches"]

        # Determine next action based on match status
        now = datetime.now().isoformat()

        if evaluation["status"] == "match":
            # Good match - go to approval
            state["next_agent"] = "approval"
            state["recommended_technology"] = technology

            message = self._build_match_message(technology, evaluation)

        elif evaluation["status"] == "partial":
            # Partial match - recommend alternatives but note it could work
            state["next_agent"] = "recommender"

            message = self._build_partial_message(technology, evaluation)

        else:  # mismatch
            # Poor match - definitely recommend alternatives
            state["next_agent"] = "recommender"

            message = self._build_mismatch_message(technology, evaluation)

        # Add to conversation
        state["conversation_history"].append(
            ConversationTurn(
                role="agent",
                content=message,
                timestamp=now,
                agent_name=self.name
            )
        )

        state["current_agent"] = self.name
        state["updated_at"] = now

        return state

    def _extract_constraints(self, state: AgentState) -> list:
        """Extract constraints from state context."""
        constraints = []
        context = state.get("request_context", "") or ""

        if "small" in context.lower() or "solo" in context.lower():
            constraints.append("small_team")
        if "urgent" in context.lower() or "deadline" in context.lower():
            constraints.append("urgent_timeline")
        if "scale:large" in context:
            constraints.append("large_scale")
        if "scale:small" in context:
            constraints.append("small_scale")

        return constraints

    def _build_match_message(self, technology: str, evaluation: Dict[str, Any]) -> str:
        """Build message for a good technology match."""
        matches = evaluation.get("matches", [])
        match_str = ", ".join(matches) if matches else "your general requirements"

        return f"""**Evaluation Complete**

✅ **{technology}** appears to be a good fit for your requirements.

**Match Score:** {evaluation['match_score']:.0%}

**Why it works:**
{technology} aligns well with your needs for {match_str}.

{evaluation.get('explanation', '')}

I'll prepare this for your approval."""

    def _build_partial_message(self, technology: str, evaluation: Dict[str, Any]) -> str:
        """Build message for a partial technology match."""
        matches = evaluation.get("matches", [])
        mismatches = evaluation.get("mismatches", [])

        matches_str = ", ".join(matches) if matches else "some of your requirements"
        mismatches_str = ", ".join(mismatches) if mismatches else "some areas"

        return f"""**Evaluation Complete**

⚠️ **{technology}** partially matches your requirements.

**Match Score:** {evaluation['match_score']:.0%}

**What works:** {matches_str}
**Potential concerns:** {mismatches_str}

{evaluation.get('explanation', '')}

Let me suggest some alternatives that might be a better fit..."""

    def _build_mismatch_message(self, technology: str, evaluation: Dict[str, Any]) -> str:
        """Build message for a poor technology match."""
        mismatches = evaluation.get("mismatches", [])
        risks = evaluation.get("risk_factors", [])

        mismatches_str = ", ".join(mismatches) if mismatches else "your core requirements"

        risk_section = ""
        if risks:
            risk_section = "\n**Risk Factors:**\n" + "\n".join(f"- {r}" for r in risks)

        return f"""**Evaluation Complete**

❌ **{technology}** may not be the best fit for your requirements.

**Match Score:** {evaluation['match_score']:.0%}

**Key Concerns:**
Your requirements for {mismatches_str} don't align well with {technology}'s strengths.

{evaluation.get('explanation', '')}
{risk_section}

Let me recommend some better-suited alternatives..."""

    def get_evaluation_summary(self, state: AgentState) -> EvaluationResult:
        """Get structured evaluation result."""
        return EvaluationResult(
            technology=state.get("technology_requested", "unknown"),
            match_score=state.get("match_score", 0.0),
            status=state.get("match_status", "unknown"),
            matching_requirements=state.get("matches", []),
            mismatched_requirements=state.get("mismatches", []),
            risk_factors=[],  # Could extract from evaluation
            explanation=f"Match score: {state.get('match_score', 0):.0%}"
        )


def create_evaluator_node(llm: Optional[ChatOpenAI] = None):
    """
    Factory function to create evaluator node for LangGraph.

    Usage in LangGraph:
        workflow.add_node("evaluate", create_evaluator_node(llm))
    """
    agent = EvaluatorAgent(llm)

    def node_function(state: AgentState) -> AgentState:
        return agent.process(state)

    return node_function
