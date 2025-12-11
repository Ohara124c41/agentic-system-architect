"""
Multi-Stakeholder Mode

Adapts "why" questions and evaluation criteria based on stakeholder role:
- Developer: Focus on implementation complexity, learning curve
- Architect: Focus on scalability, maintainability, patterns
- PM: Focus on timeline, risk, team capacity

Based on INCOSE stakeholder analysis principles.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class StakeholderRole(Enum):
    """Stakeholder roles with different priorities."""
    DEVELOPER = "developer"
    ARCHITECT = "architect"
    PM = "project_manager"
    DEVOPS = "devops_engineer"
    DATA_SCIENTIST = "data_scientist"
    PRODUCT_OWNER = "product_owner"


@dataclass
class StakeholderPerspective:
    """A stakeholder's unique perspective on technology choices."""
    role: StakeholderRole
    priorities: List[str]
    concerns: List[str]
    evaluation_weights: Dict[str, float]
    why_question_focus: List[str]


# Stakeholder profiles with priorities and concerns
STAKEHOLDER_PROFILES: Dict[StakeholderRole, StakeholderPerspective] = {
    StakeholderRole.DEVELOPER: StakeholderPerspective(
        role=StakeholderRole.DEVELOPER,
        priorities=[
            "learning_curve",
            "documentation_quality",
            "debugging_ease",
            "local_development",
            "tooling_support"
        ],
        concerns=[
            "complexity_overhead",
            "unfamiliar_paradigms",
            "lack_of_examples",
            "poor_error_messages"
        ],
        evaluation_weights={
            "complexity": 0.3,
            "learning_curve": 0.25,
            "documentation": 0.2,
            "community_support": 0.15,
            "performance": 0.1
        },
        why_question_focus=[
            "How comfortable is the team with {technology}?",
            "What's the debugging experience like with {technology}?",
            "How well-documented is {technology} for your use case?",
            "What does local development look like with {technology}?"
        ]
    ),

    StakeholderRole.ARCHITECT: StakeholderPerspective(
        role=StakeholderRole.ARCHITECT,
        priorities=[
            "scalability",
            "maintainability",
            "pattern_alignment",
            "future_proofing",
            "integration_capability"
        ],
        concerns=[
            "architectural_mismatch",
            "vendor_lock_in",
            "evolution_difficulty",
            "data_migration_complexity"
        ],
        evaluation_weights={
            "scalability": 0.25,
            "maintainability": 0.25,
            "integration": 0.2,
            "flexibility": 0.15,
            "maturity": 0.15
        },
        why_question_focus=[
            "How does {technology} fit with your existing architecture?",
            "What's the evolution path if requirements change?",
            "How will {technology} integrate with your other systems?",
            "What's the data migration strategy if you need to switch?"
        ]
    ),

    StakeholderRole.PM: StakeholderPerspective(
        role=StakeholderRole.PM,
        priorities=[
            "timeline_impact",
            "risk_level",
            "team_capacity",
            "delivery_predictability",
            "stakeholder_confidence"
        ],
        concerns=[
            "schedule_risk",
            "unknown_unknowns",
            "team_availability",
            "dependencies"
        ],
        evaluation_weights={
            "timeline": 0.3,
            "risk": 0.25,
            "team_readiness": 0.2,
            "predictability": 0.15,
            "cost": 0.1
        },
        why_question_focus=[
            "How does {technology} impact your delivery timeline?",
            "What risks does {technology} introduce to the project?",
            "Does the team have capacity to learn {technology}?",
            "What's the backup plan if {technology} doesn't work out?"
        ]
    ),

    StakeholderRole.DEVOPS: StakeholderPerspective(
        role=StakeholderRole.DEVOPS,
        priorities=[
            "operational_complexity",
            "monitoring_observability",
            "deployment_ease",
            "security_posture",
            "resource_efficiency"
        ],
        concerns=[
            "operational_burden",
            "alert_fatigue",
            "security_vulnerabilities",
            "resource_waste"
        ],
        evaluation_weights={
            "operability": 0.3,
            "observability": 0.2,
            "security": 0.2,
            "efficiency": 0.15,
            "automation": 0.15
        },
        why_question_focus=[
            "What's the operational overhead of {technology}?",
            "How will you monitor and observe {technology}?",
            "What's the security model for {technology}?",
            "How does {technology} fit your deployment pipeline?"
        ]
    ),

    StakeholderRole.DATA_SCIENTIST: StakeholderPerspective(
        role=StakeholderRole.DATA_SCIENTIST,
        priorities=[
            "model_performance",
            "experimentation_speed",
            "interpretability",
            "data_access",
            "reproducibility"
        ],
        concerns=[
            "black_box_models",
            "data_quality_issues",
            "experiment_tracking",
            "deployment_friction"
        ],
        evaluation_weights={
            "accuracy": 0.25,
            "interpretability": 0.2,
            "experimentation": 0.2,
            "data_integration": 0.2,
            "deployment": 0.15
        },
        why_question_focus=[
            "How interpretable is {technology} for stakeholders?",
            "How quickly can you iterate experiments with {technology}?",
            "How does {technology} handle your data format?",
            "What's the path from experiment to production with {technology}?"
        ]
    ),

    StakeholderRole.PRODUCT_OWNER: StakeholderPerspective(
        role=StakeholderRole.PRODUCT_OWNER,
        priorities=[
            "user_value_delivery",
            "feature_velocity",
            "market_fit",
            "competitive_advantage",
            "customer_feedback_loop"
        ],
        concerns=[
            "slow_time_to_market",
            "missed_opportunities",
            "user_experience_impact",
            "feature_limitations"
        ],
        evaluation_weights={
            "value_delivery": 0.3,
            "velocity": 0.25,
            "flexibility": 0.2,
            "user_experience": 0.15,
            "innovation": 0.1
        },
        why_question_focus=[
            "How does {technology} enable faster feature delivery?",
            "What user experience capabilities does {technology} unlock?",
            "How does {technology} affect our competitive position?",
            "Can we iterate on features quickly with {technology}?"
        ]
    ),
}


class MultiStakeholderMode:
    """
    Adapts governance questions and evaluation based on stakeholder role.

    Uses stakeholder-specific:
    - Why questions
    - Evaluation weights
    - Risk assessments
    - Recommendation framing
    """

    def __init__(self, primary_role: StakeholderRole = StakeholderRole.DEVELOPER):
        self.primary_role = primary_role
        self.additional_roles: List[StakeholderRole] = []

    def set_primary_role(self, role: StakeholderRole):
        """Set the primary stakeholder role."""
        self.primary_role = role

    def add_role(self, role: StakeholderRole):
        """Add an additional stakeholder perspective."""
        if role not in self.additional_roles and role != self.primary_role:
            self.additional_roles.append(role)

    def get_perspective(self, role: Optional[StakeholderRole] = None) -> StakeholderPerspective:
        """Get the perspective for a role."""
        role = role or self.primary_role
        return STAKEHOLDER_PROFILES.get(role, STAKEHOLDER_PROFILES[StakeholderRole.DEVELOPER])

    def get_why_questions(self, technology: str,
                          role: Optional[StakeholderRole] = None) -> List[str]:
        """Get role-specific why questions."""
        perspective = self.get_perspective(role)
        return [q.format(technology=technology) for q in perspective.why_question_focus]

    def evaluate_for_role(self, technology_profile: Dict[str, Any],
                          requirements: List[str],
                          role: Optional[StakeholderRole] = None) -> Dict[str, Any]:
        """Evaluate a technology from a specific role's perspective."""
        perspective = self.get_perspective(role)

        # Calculate weighted score based on role priorities
        score = 0.0
        priority_matches = []
        concern_flags = []

        for priority in perspective.priorities:
            if priority in requirements or priority in technology_profile.get("best_for", []):
                score += 0.15
                priority_matches.append(priority)

        for concern in perspective.concerns:
            if concern in technology_profile.get("not_ideal_for", []):
                score -= 0.1
                concern_flags.append(concern)

        # Clamp score
        score = max(0.0, min(1.0, 0.5 + score))

        return {
            "role": perspective.role.value,
            "score": score,
            "priority_matches": priority_matches,
            "concerns_flagged": concern_flags,
            "recommendation": self._generate_role_recommendation(
                score, priority_matches, concern_flags, perspective
            )
        }

    def _generate_role_recommendation(self, score: float,
                                       matches: List[str],
                                       concerns: List[str],
                                       perspective: StakeholderPerspective) -> str:
        """Generate role-specific recommendation text."""
        role_name = perspective.role.value.replace("_", " ").title()

        if score >= 0.7:
            if matches:
                return f"From a {role_name} perspective, this aligns well with your priorities: {', '.join(matches[:3])}."
            return f"From a {role_name} perspective, this appears to be a reasonable choice."
        elif score >= 0.4:
            msg = f"From a {role_name} perspective, consider these trade-offs."
            if concerns:
                msg += f" Watch out for: {', '.join(concerns[:2])}."
            return msg
        else:
            msg = f"From a {role_name} perspective, there are significant concerns."
            if concerns:
                msg += f" Major issues: {', '.join(concerns[:2])}."
            return msg

    def get_multi_perspective_evaluation(self, technology_profile: Dict[str, Any],
                                          requirements: List[str]) -> Dict[str, Any]:
        """Get evaluations from all configured roles."""
        evaluations = {}

        # Primary role
        evaluations[self.primary_role.value] = self.evaluate_for_role(
            technology_profile, requirements, self.primary_role
        )

        # Additional roles
        for role in self.additional_roles:
            evaluations[role.value] = self.evaluate_for_role(
                technology_profile, requirements, role
            )

        # Compute aggregate recommendation
        scores = [e["score"] for e in evaluations.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.5

        all_concerns = []
        for e in evaluations.values():
            all_concerns.extend(e["concerns_flagged"])

        return {
            "perspectives": evaluations,
            "aggregate_score": avg_score,
            "consensus": avg_score >= 0.6,
            "shared_concerns": list(set(all_concerns)),
            "recommendation": self._generate_aggregate_recommendation(
                evaluations, avg_score
            )
        }

    def _generate_aggregate_recommendation(self, evaluations: Dict[str, Dict],
                                            avg_score: float) -> str:
        """Generate aggregate recommendation across all perspectives."""
        if avg_score >= 0.7:
            return "✅ All stakeholder perspectives support this choice."
        elif avg_score >= 0.5:
            # Find dissenting perspectives
            low_scores = [
                role for role, e in evaluations.items()
                if e["score"] < 0.5
            ]
            if low_scores:
                return f"⚠️ Mixed perspectives. Concerns from: {', '.join(low_scores)}"
            return "⚠️ Moderate support across perspectives. Discuss trade-offs."
        else:
            return "❌ Multiple stakeholder concerns. Consider alternatives."

    def format_multi_perspective_report(self, evaluation: Dict[str, Any]) -> str:
        """Format a readable multi-perspective report."""
        lines = []
        lines.append("👥 MULTI-STAKEHOLDER EVALUATION")
        lines.append("=" * 60)
        lines.append("")

        for role, eval_data in evaluation.get("perspectives", {}).items():
            role_display = role.replace("_", " ").title()
            score = eval_data["score"]
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))

            lines.append(f"📋 {role_display}")
            lines.append(f"   Score: {bar} {score:.0%}")
            lines.append(f"   {eval_data['recommendation']}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"Aggregate Score: {evaluation['aggregate_score']:.0%}")
        lines.append(f"Consensus: {'Yes ✓' if evaluation['consensus'] else 'No ✗'}")
        lines.append("")
        lines.append(evaluation["recommendation"])

        return "\n".join(lines)
