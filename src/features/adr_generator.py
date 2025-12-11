"""
Architecture Decision Record (ADR) Generator

Auto-generates ADR documents from governance conversations.
Follows the standard ADR format (Michael Nygard style).
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ADRDocument:
    """Structured ADR document."""
    number: int
    title: str
    date: str
    status: str  # proposed, accepted, deprecated, superseded
    context: str
    decision: str
    consequences: List[str]
    alternatives_considered: List[Dict[str, str]]
    requirements_addressed: List[str]
    stakeholders: List[str]
    related_adrs: List[int]


class ADRGenerator:
    """
    Generates Architecture Decision Records from governance sessions.

    ADR Format:
    - Title
    - Status
    - Context
    - Decision
    - Consequences
    - Alternatives Considered
    """

    def __init__(self, adr_number: int = 1):
        self.adr_number = adr_number

    def generate_from_state(self, state: Dict[str, Any]) -> ADRDocument:
        """Generate an ADR from governance session state."""
        original_tech = state.get("technology_requested", "Unknown")
        recommended_tech = state.get("recommended_technology", original_tech)
        requirements = state.get("extracted_requirements", [])
        mismatches = state.get("mismatches", [])
        alternatives = state.get("alternatives", [])
        rationale = state.get("recommendation_rationale", "")
        match_score = state.get("match_score", 0)

        # Determine decision
        if original_tech == recommended_tech:
            decision_text = f"We will use **{original_tech}** as originally proposed."
            status = "accepted"
        else:
            decision_text = (
                f"We will use **{recommended_tech}** instead of the originally "
                f"proposed {original_tech}."
            )
            status = "proposed"

        # Build context from conversation
        context = self._build_context(state)

        # Build consequences
        consequences = self._build_consequences(
            original_tech, recommended_tech, mismatches, match_score
        )

        # Build alternatives list
        alt_list = self._build_alternatives(original_tech, alternatives)

        return ADRDocument(
            number=self.adr_number,
            title=f"Use {recommended_tech} for {self._infer_purpose(state)}",
            date=datetime.now().strftime("%Y-%m-%d"),
            status=status,
            context=context,
            decision=decision_text,
            consequences=consequences,
            alternatives_considered=alt_list,
            requirements_addressed=requirements,
            stakeholders=["Architecture Team", "Development Team"],
            related_adrs=[]
        )

    def _build_context(self, state: Dict[str, Any]) -> str:
        """Build context section from conversation history."""
        parts = []

        user_request = state.get("user_request", "")
        if user_request:
            parts.append(f"The team proposed: \"{user_request}\"")

        requirements = state.get("extracted_requirements", [])
        if requirements:
            req_text = ", ".join(requirements[:5])
            parts.append(f"Through requirements analysis, we identified: {req_text}.")

        stated_reasons = state.get("stated_reasons", [])
        if stated_reasons:
            reasons_text = " ".join(stated_reasons[:3])
            parts.append(f"The team's rationale included: {reasons_text}")

        return " ".join(parts)

    def _build_consequences(self, original: str, recommended: str,
                            mismatches: List[str], match_score: float) -> List[str]:
        """Build consequences section."""
        consequences = []

        if original != recommended:
            consequences.append(
                f"✅ Better alignment with requirements (recommended: {match_score:.0%} match)"
            )
            if mismatches:
                mismatch_text = ", ".join(mismatches[:3])
                consequences.append(
                    f"✅ Avoids issues with: {mismatch_text}"
                )
            consequences.append(
                f"⚠️ Team may need to learn {recommended} if unfamiliar"
            )
            consequences.append(
                f"⚠️ Migration from {original} design patterns may be needed"
            )
        else:
            consequences.append(
                f"✅ Team can proceed with familiar technology ({original})"
            )
            if mismatches:
                mismatch_text = ", ".join(mismatches[:3])
                consequences.append(
                    f"⚠️ Potential challenges with: {mismatch_text}"
                )
            consequences.append(
                "ℹ️ Decision aligns with original proposal"
            )

        return consequences

    def _build_alternatives(self, original: str,
                            alternatives: List[Dict]) -> List[Dict[str, str]]:
        """Build alternatives considered section."""
        alt_list = []

        # Original is always an alternative
        alt_list.append({
            "technology": original,
            "reason_not_chosen": "Originally proposed; evaluated during governance"
        })

        # Add other alternatives from recommendations
        for alt in alternatives[:3]:
            if isinstance(alt, dict):
                alt_list.append({
                    "technology": alt.get("technology", "Unknown"),
                    "reason_not_chosen": alt.get("rationale", "Lower match score")
                })

        return alt_list

    def _infer_purpose(self, state: Dict[str, Any]) -> str:
        """Infer the purpose from requirements and context."""
        requirements = state.get("extracted_requirements", [])
        category = state.get("technology_category", "")

        if "tabular_data" in requirements:
            return "structured data storage"
        elif "graph_data" in requirements or "relationship" in str(requirements):
            return "relationship data management"
        elif "image_data" in requirements:
            return "image processing"
        elif category == "ml_model":
            return "predictive modeling"
        elif category == "architecture":
            return "system architecture"
        elif category == "devops":
            return "deployment infrastructure"
        elif category == "visualization":
            return "data visualization"
        else:
            return "the proposed use case"

    def to_markdown(self, adr: ADRDocument) -> str:
        """Convert ADR to markdown format."""
        lines = []

        # Header
        lines.append(f"# ADR-{adr.number:04d}: {adr.title}")
        lines.append("")
        lines.append(f"**Date:** {adr.date}")
        lines.append(f"**Status:** {adr.status.upper()}")
        lines.append("")

        # Context
        lines.append("## Context")
        lines.append("")
        lines.append(adr.context)
        lines.append("")

        # Decision
        lines.append("## Decision")
        lines.append("")
        lines.append(adr.decision)
        lines.append("")

        # Requirements Addressed
        if adr.requirements_addressed:
            lines.append("## Requirements Addressed")
            lines.append("")
            for req in adr.requirements_addressed:
                lines.append(f"- {req.replace('_', ' ').title()}")
            lines.append("")

        # Consequences
        lines.append("## Consequences")
        lines.append("")
        for consequence in adr.consequences:
            lines.append(f"- {consequence}")
        lines.append("")

        # Alternatives Considered
        lines.append("## Alternatives Considered")
        lines.append("")
        for alt in adr.alternatives_considered:
            lines.append(f"### {alt['technology']}")
            lines.append(f"_{alt['reason_not_chosen']}_")
            lines.append("")

        # Stakeholders
        if adr.stakeholders:
            lines.append("## Stakeholders")
            lines.append("")
            for stakeholder in adr.stakeholders:
                lines.append(f"- {stakeholder}")
            lines.append("")

        # Related ADRs
        if adr.related_adrs:
            lines.append("## Related ADRs")
            lines.append("")
            for related in adr.related_adrs:
                lines.append(f"- ADR-{related:04d}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by Agentic Architecture Governance System*")

        return "\n".join(lines)


def generate_adr(state: Dict[str, Any], adr_number: int = 1) -> str:
    """Generate an ADR markdown document from session state."""
    generator = ADRGenerator(adr_number)
    adr = generator.generate_from_state(state)
    return generator.to_markdown(adr)
