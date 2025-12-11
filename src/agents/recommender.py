"""
Alternative Recommender Agent

Suggests better-fitting technologies when there's a mismatch
between requested technology and actual requirements.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.state import AgentState, ConversationTurn
from ..schemas.responses import AlternativeRecommendation
from ..tools.matching import score_alternatives
from ..tools.knowledge_base import (
    get_technology_profile,
    get_alternatives_for_category,
    get_best_alternative,
)
from ..prompts import RECOMMENDER_SYSTEM_PROMPT, RECOMMEND_ALTERNATIVE_PROMPT


class RecommenderAgent:
    """
    Alternative Recommender Agent

    Responsibilities:
    1. Find better-fitting technologies for the requirements
    2. Explain why alternatives are better suited
    3. Present trade-offs honestly
    4. Prepare recommendation for human approval
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, max_alternatives: int = 3):
        """
        Initialize the recommender agent.

        Args:
            llm: LangChain ChatOpenAI instance (for generating explanations)
            max_alternatives: Maximum number of alternatives to suggest
        """
        self.llm = llm
        self.name = "recommender"
        self.max_alternatives = max_alternatives

    def process(self, state: AgentState) -> AgentState:
        """
        Find and recommend better alternatives.

        Process:
        1. Get alternatives for the technology category
        2. Score each alternative against requirements
        3. Generate recommendations with rationale
        4. Update state and route to approval
        """
        technology = state.get("technology_requested")
        category = state.get("technology_category")
        requirements = state.get("extracted_requirements", [])
        mismatches = state.get("mismatches", [])

        # Get the technology profile for alternatives list
        profile = get_technology_profile(technology) if technology else None

        # Get potential alternatives
        if profile and profile.get("typical_alternatives"):
            alternatives = profile["typical_alternatives"]
        else:
            alternatives = get_alternatives_for_category(category) if category else []

        # Score alternatives against requirements (DETERMINISTIC - no API)
        scored_alternatives = score_alternatives(
            requirements=requirements,
            alternatives=alternatives,
            original_technology=technology or ""
        )

        # Take top N alternatives
        top_alternatives = scored_alternatives[:self.max_alternatives]

        # Build recommendation
        if top_alternatives and top_alternatives[0]["match_score"] > state.get("match_score", 0):
            best_alt = top_alternatives[0]
            state["recommended_technology"] = best_alt["technology"]

            # Generate detailed recommendations (may use LLM)
            detailed_recommendations = self._build_recommendations(
                original=technology,
                alternatives=top_alternatives,
                requirements=requirements,
                mismatches=mismatches,
                state=state
            )
            state["alternatives"] = detailed_recommendations
            state["recommendation_rationale"] = self._build_rationale(
                original=technology,
                recommended=best_alt["technology"],
                requirements=requirements,
                mismatches=mismatches
            )
        else:
            # No better alternative found - proceed with original + warning
            state["recommended_technology"] = technology
            state["alternatives"] = []
            state["recommendation_rationale"] = f"No clearly better alternative found. {technology} may work despite some concerns."

        # Build conversation message
        now = datetime.now().isoformat()
        message = self._build_recommendation_message(state)

        state["conversation_history"].append(
            ConversationTurn(
                role="agent",
                content=message,
                timestamp=now,
                agent_name=self.name
            )
        )

        # Route to approval
        state["current_agent"] = self.name
        state["next_agent"] = "approval"
        state["updated_at"] = now

        return state

    def _build_recommendations(
        self,
        original: str,
        alternatives: List[Dict],
        requirements: List[str],
        mismatches: List[str],
        state: AgentState
    ) -> List[Dict[str, Any]]:
        """Build detailed recommendation objects for each alternative."""
        recommendations = []

        for alt in alternatives:
            rec = {
                "technology": alt["technology"],
                "fit_score": alt["match_score"],
                "improvement": alt.get("improvement_over_original", 0),
                "matches": alt.get("matches", []),
                "rationale": self._generate_rationale(
                    original=original,
                    alternative=alt["technology"],
                    requirements=requirements,
                    mismatches=mismatches,
                    alt_matches=alt.get("matches", [])
                ),
                "tradeoffs": self._identify_tradeoffs(original, alt["technology"]),
                "migration_complexity": self._assess_migration_complexity(original, alt["technology"]),
            }
            recommendations.append(rec)

        return recommendations

    def _generate_rationale(
        self,
        original: str,
        alternative: str,
        requirements: List[str],
        mismatches: List[str],
        alt_matches: List[str]
    ) -> str:
        """Generate rationale for why alternative is better."""
        # Get profiles
        original_profile = get_technology_profile(original)
        alt_profile = get_technology_profile(alternative)

        rationale_parts = []

        # Address mismatches
        if mismatches:
            mismatch_str = ", ".join(mismatches[:3])
            rationale_parts.append(
                f"{alternative} better handles your requirements for {mismatch_str}"
            )

        # Highlight alternative strengths
        if alt_profile:
            best_for = alt_profile.get("best_for", [])
            matching_strengths = [r for r in requirements if r in best_for]
            if matching_strengths:
                strengths_str = ", ".join(matching_strengths[:3])
                rationale_parts.append(
                    f"It's specifically designed for {strengths_str}"
                )

        # Add complexity comparison
        if original_profile and alt_profile:
            orig_complexity = original_profile.get("complexity", "medium")
            alt_complexity = alt_profile.get("complexity", "medium")
            if alt_complexity == "low" and orig_complexity in ["medium", "high"]:
                rationale_parts.append("It's simpler to implement and maintain")
            elif alt_complexity == "medium" and orig_complexity == "high":
                rationale_parts.append("It offers a better balance of power and simplicity")

        return ". ".join(rationale_parts) + "." if rationale_parts else f"{alternative} may be a better fit."

    def _identify_tradeoffs(self, original: str, alternative: str) -> List[str]:
        """Identify trade-offs when switching to the alternative."""
        tradeoffs = []

        original_profile = get_technology_profile(original)
        alt_profile = get_technology_profile(alternative)

        if original_profile and alt_profile:
            # What does original do better?
            orig_best = set(original_profile.get("best_for", []))
            alt_best = set(alt_profile.get("best_for", []))

            lost_capabilities = orig_best - alt_best
            if lost_capabilities:
                tradeoffs.append(
                    f"May have reduced capabilities for: {', '.join(list(lost_capabilities)[:3])}"
                )

            # Complexity change
            orig_complexity = original_profile.get("complexity", "medium")
            alt_complexity = alt_profile.get("complexity", "medium")

            complexity_order = {"low": 1, "medium": 2, "high": 3}
            if complexity_order.get(alt_complexity, 2) > complexity_order.get(orig_complexity, 2):
                tradeoffs.append("Higher complexity than original choice")

            # Learning curve
            if alt_profile.get("learning_curve") == "steep":
                tradeoffs.append("Steeper learning curve")

        if not tradeoffs:
            tradeoffs.append("Minimal trade-offs for your use case")

        return tradeoffs

    def _assess_migration_complexity(self, original: str, alternative: str) -> str:
        """Assess how complex it would be to use the alternative."""
        # Same category = usually low complexity
        orig_profile = get_technology_profile(original)
        alt_profile = get_technology_profile(alternative)

        if orig_profile and alt_profile:
            if orig_profile.get("_category") == alt_profile.get("_category"):
                # Same category - check if they're related
                orig_alts = orig_profile.get("typical_alternatives", [])
                if alternative.lower() in [a.lower() for a in orig_alts]:
                    return "low"
                return "medium"

        return "medium"

    def _build_rationale(
        self,
        original: str,
        recommended: str,
        requirements: List[str],
        mismatches: List[str]
    ) -> str:
        """Build overall recommendation rationale."""
        if original == recommended:
            return f"Despite some concerns, {original} appears to be a reasonable choice."

        parts = []
        if mismatches:
            parts.append(f"Your requirements for {', '.join(mismatches[:3])} are better served by {recommended}")

        req_str = ", ".join(requirements[:3]) if requirements else "your stated needs"
        parts.append(f"{recommended} aligns better with {req_str}")

        return ". ".join(parts) + "."

    def _build_recommendation_message(self, state: AgentState) -> str:
        """Build the conversation message with recommendations."""
        original = state.get("technology_requested")
        recommended = state.get("recommended_technology")
        alternatives = state.get("alternatives", [])
        match_score = state.get("match_score", 0)

        if original == recommended or not alternatives:
            # No change recommended
            return f"""**Recommendation**

After analyzing your requirements, I recommend proceeding with **{original}** as originally planned.

While there were some potential concerns (match score: {match_score:.0%}), no clearly superior alternative was identified for your specific requirements.

**Proceed with caution on:**
{chr(10).join('- ' + m for m in state.get('mismatches', [])[:3]) or '- Monitor performance as you scale'}

Ready for your approval to proceed."""

        # Alternative recommended
        message = f"""**Recommendation**

Based on your requirements, I recommend **{recommended}** instead of {original}.

**Why {recommended}?**
{state.get('recommendation_rationale', 'Better alignment with your requirements.')}

**Original Choice ({original}):** {match_score:.0%} match
**Recommended ({recommended}):** {alternatives[0]['fit_score']:.0%} match
"""

        if len(alternatives) > 1:
            message += "\n**Other Alternatives Considered:**\n"
            for alt in alternatives[1:]:
                message += f"- {alt['technology']} ({alt['fit_score']:.0%} match)\n"

        if alternatives[0].get("tradeoffs"):
            message += f"\n**Trade-offs to Consider:**\n"
            for tradeoff in alternatives[0]["tradeoffs"][:3]:
                message += f"- {tradeoff}\n"

        message += "\nReady for your decision."

        return message


def create_recommender_node(llm: Optional[ChatOpenAI] = None, max_alternatives: int = 3):
    """
    Factory function to create recommender node for LangGraph.

    Usage in LangGraph:
        workflow.add_node("recommend", create_recommender_node(llm))
    """
    agent = RecommenderAgent(llm, max_alternatives)

    def node_function(state: AgentState) -> AgentState:
        return agent.process(state)

    return node_function
