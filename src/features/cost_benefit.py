"""
Cost-Benefit Analysis for Technical Debt Prevention

Quantifies:
- Technical debt prevented by governance
- Time spent in governance process
- ROI of making the right choice upfront

Provides measurable impact for IEEE paper.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class DebtCategory(Enum):
    """Categories of technical debt."""
    ARCHITECTURE = "architecture"
    TECHNOLOGY = "technology"
    PERFORMANCE = "performance"
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    SCALABILITY = "scalability"


@dataclass
class TechnicalDebtItem:
    """A potential technical debt item."""
    category: DebtCategory
    description: str
    estimated_hours_to_fix: float
    probability_of_occurrence: float  # 0-1
    severity: str  # low, medium, high, critical
    expected_cost: float = 0.0  # hours * probability

    def __post_init__(self):
        self.expected_cost = self.estimated_hours_to_fix * self.probability_of_occurrence


@dataclass
class GovernanceCost:
    """Cost of the governance process."""
    why_questions_asked: int
    total_session_minutes: float
    llm_calls_made: int
    estimated_hourly_rate: float = 100.0  # Developer hourly rate

    @property
    def total_cost(self) -> float:
        return (self.total_session_minutes / 60) * self.estimated_hourly_rate

    @property
    def llm_cost(self) -> float:
        # Estimate: ~$0.05 per LLM call average
        return self.llm_calls_made * 0.05


@dataclass
class CostBenefitResult:
    """Result of cost-benefit analysis."""
    mediation_cost: GovernanceCost  # GovernanceCost renamed to match mediation terminology
    debt_prevented: List[TechnicalDebtItem]
    total_debt_hours_prevented: float
    total_debt_cost_prevented: float
    roi_ratio: float
    payback_description: str


# Technical debt estimates by mismatch type
DEBT_ESTIMATES: Dict[str, TechnicalDebtItem] = {
    "tabular_data": TechnicalDebtItem(
        category=DebtCategory.TECHNOLOGY,
        description="Using document store for tabular data - migration needed",
        estimated_hours_to_fix=80.0,
        probability_of_occurrence=0.7,
        severity="high"
    ),
    "complex_joins": TechnicalDebtItem(
        category=DebtCategory.PERFORMANCE,
        description="Manual joins in application code - performance issues",
        estimated_hours_to_fix=40.0,
        probability_of_occurrence=0.8,
        severity="high"
    ),
    "acid_transactions": TechnicalDebtItem(
        category=DebtCategory.ARCHITECTURE,
        description="Implementing transaction handling at app level",
        estimated_hours_to_fix=60.0,
        probability_of_occurrence=0.6,
        severity="critical"
    ),
    "interpretability": TechnicalDebtItem(
        category=DebtCategory.TECHNOLOGY,
        description="Adding explainability layer to black-box model",
        estimated_hours_to_fix=40.0,
        probability_of_occurrence=0.5,
        severity="medium"
    ),
    "small_team": TechnicalDebtItem(
        category=DebtCategory.ARCHITECTURE,
        description="Microservices operational overhead for small team",
        estimated_hours_to_fix=100.0,
        probability_of_occurrence=0.8,
        severity="high"
    ),
    "mvp_stage": TechnicalDebtItem(
        category=DebtCategory.ARCHITECTURE,
        description="Over-engineering slowing MVP delivery",
        estimated_hours_to_fix=60.0,
        probability_of_occurrence=0.7,
        severity="medium"
    ),
    "limited_ops_experience": TechnicalDebtItem(
        category=DebtCategory.MAINTENANCE,
        description="Kubernetes learning curve and operational issues",
        estimated_hours_to_fix=120.0,
        probability_of_occurrence=0.6,
        severity="high"
    ),
    "custom_visualizations": TechnicalDebtItem(
        category=DebtCategory.TECHNOLOGY,
        description="Workarounds for limited customization",
        estimated_hours_to_fix=30.0,
        probability_of_occurrence=0.5,
        severity="medium"
    ),
    "graph_data": TechnicalDebtItem(
        category=DebtCategory.PERFORMANCE,
        description="Graph traversals in non-graph database",
        estimated_hours_to_fix=80.0,
        probability_of_occurrence=0.7,
        severity="high"
    ),
    "horizontal_scaling": TechnicalDebtItem(
        category=DebtCategory.SCALABILITY,
        description="Re-architecting for scale after initial build",
        estimated_hours_to_fix=150.0,
        probability_of_occurrence=0.4,
        severity="high"
    ),
}


class CostBenefitAnalyzer:
    """
    Analyzes cost-benefit of governance decisions.

    Quantifies:
    - Time spent in governance
    - Technical debt prevented
    - ROI of making right choice
    """

    def __init__(self, hourly_rate: float = 100.0):
        self.hourly_rate = hourly_rate

    def analyze(self, state: Dict[str, Any],
                session_start: Optional[datetime] = None) -> CostBenefitResult:
        """Analyze cost-benefit from governance session state."""

        # Calculate governance cost
        why_questions = len(state.get("why_questions", []))
        llm_calls = state.get("total_llm_calls", 0)

        # Estimate session time: ~2 min per why question + 1 min base
        session_minutes = 1.0 + (why_questions * 2.0)

        mediation_cost = GovernanceCost(
            why_questions_asked=why_questions,
            total_session_minutes=session_minutes,
            llm_calls_made=llm_calls,
            estimated_hourly_rate=self.hourly_rate
        )

        # Identify prevented debt from mismatches
        mismatches = state.get("mismatches", [])
        match_status = state.get("match_status", "match")
        recommended = state.get("recommended_technology")
        original = state.get("technology_requested")

        debt_prevented = []

        # If recommendation was different and user follows it, debt is prevented
        if recommended and recommended != original and match_status in ["partial", "mismatch"]:
            for mismatch in mismatches:
                mismatch_key = mismatch.lower().replace(" ", "_")
                if mismatch_key in DEBT_ESTIMATES:
                    debt_prevented.append(DEBT_ESTIMATES[mismatch_key])

            # Also add general category-based debt
            requirements = state.get("extracted_requirements", [])
            for req in requirements:
                req_key = req.lower().replace(" ", "_")
                if req_key in DEBT_ESTIMATES and DEBT_ESTIMATES[req_key] not in debt_prevented:
                    # Only add if it was a mismatch
                    if req_key in [m.lower().replace(" ", "_") for m in mismatches]:
                        debt_prevented.append(DEBT_ESTIMATES[req_key])

        # Calculate totals
        total_debt_hours = sum(d.expected_cost for d in debt_prevented)
        total_debt_cost = total_debt_hours * self.hourly_rate

        # Calculate ROI
        if mediation_cost.total_cost > 0:
            roi_ratio = total_debt_cost / mediation_cost.total_cost
        else:
            roi_ratio = float('inf') if total_debt_cost > 0 else 1.0

        # Generate payback description
        payback_desc = self._generate_payback_description(
            mediation_cost, total_debt_cost, roi_ratio, debt_prevented
        )

        return CostBenefitResult(
            mediation_cost=mediation_cost,
            debt_prevented=debt_prevented,
            total_debt_hours_prevented=total_debt_hours,
            total_debt_cost_prevented=total_debt_cost,
            roi_ratio=roi_ratio,
            payback_description=payback_desc
        )

    def _generate_payback_description(self, cost: GovernanceCost,
                                       debt_prevented: float,
                                       roi: float,
                                       items: List[TechnicalDebtItem]) -> str:
        """Generate human-readable payback description."""
        if not items:
            return "No significant technical debt identified. Mediation validated the choice."

        if roi >= 10:
            return (
                f"Excellent ROI: {roi:.0f}x return on mediation investment. "
                f"${debt_prevented:,.0f} in debt prevented for ${cost.total_cost:,.0f} in mediation time."
            )
        elif roi >= 5:
            return (
                f"Strong ROI: {roi:.0f}x return. "
                f"Every hour of mediation prevents ~{roi:.0f} hours of future rework."
            )
        elif roi >= 2:
            return (
                f"Positive ROI: {roi:.1f}x return. "
                f"Governance time well spent avoiding {len(items)} debt items."
            )
        elif roi >= 1:
            return (
                f"Break-even ROI: {roi:.1f}x. "
                f"Governance cost roughly equals debt prevented."
            )
        else:
            return (
                f"Limited ROI for this session: {roi:.1f}x. "
                f"Original choice may have been reasonable."
            )

    def format_report(self, result: CostBenefitResult) -> str:
        """Format cost-benefit analysis as readable report."""
        lines = []
        lines.append("💰 COST-BENEFIT ANALYSIS")
        lines.append("=" * 60)
        lines.append("")

        # Mediation cost section
        lines.append("📊 MEDIATION COST")
        lines.append("-" * 40)
        lines.append(f"  Why Questions Asked: {result.mediation_cost.why_questions_asked}")
        lines.append(f"  Session Time: {result.mediation_cost.total_session_minutes:.1f} minutes")
        lines.append(f"  LLM API Calls: {result.mediation_cost.llm_calls_made}")
        lines.append(f"  Estimated Cost: ${result.mediation_cost.total_cost:,.2f}")
        lines.append(f"  LLM API Cost: ${result.mediation_cost.llm_cost:,.2f}")
        lines.append("")

        # Debt prevented section
        lines.append("🛡️ TECHNICAL DEBT PREVENTED")
        lines.append("-" * 40)

        if result.debt_prevented:
            for debt in result.debt_prevented:
                severity_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(
                    debt.severity, "⚪"
                )
                lines.append(f"  {severity_icon} {debt.description}")
                lines.append(f"     Est. fix time: {debt.estimated_hours_to_fix:.0f}h "
                           f"(prob: {debt.probability_of_occurrence:.0%})")
        else:
            lines.append("  No significant debt items identified.")

        lines.append("")
        lines.append(f"  Total Debt Hours Prevented: {result.total_debt_hours_prevented:.0f}h")
        lines.append(f"  Total Debt Cost Prevented: ${result.total_debt_cost_prevented:,.2f}")
        lines.append("")

        # ROI section
        lines.append("📈 RETURN ON INVESTMENT")
        lines.append("-" * 40)

        roi_bar = "█" * min(int(result.roi_ratio), 20)
        lines.append(f"  ROI Ratio: {roi_bar} {result.roi_ratio:.1f}x")
        lines.append("")
        lines.append(f"  {result.payback_description}")
        lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_aggregate_stats(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get aggregate statistics across multiple sessions."""
        total_mediation_cost = 0.0
        total_debt_prevented = 0.0
        total_why_questions = 0
        total_sessions = len(sessions)

        for session in sessions:
            result = self.analyze(session)
            total_mediation_cost += result.mediation_cost.total_cost
            total_debt_prevented += result.total_debt_cost_prevented
            total_why_questions += result.mediation_cost.why_questions_asked

        return {
            "total_sessions": total_sessions,
            "total_mediation_cost": total_mediation_cost,
            "total_debt_prevented": total_debt_prevented,
            "aggregate_roi": total_debt_prevented / total_mediation_cost if total_mediation_cost > 0 else 0,
            "average_questions_per_session": total_why_questions / total_sessions if total_sessions > 0 else 0,
            "average_debt_per_session": total_debt_prevented / total_sessions if total_sessions > 0 else 0
        }
