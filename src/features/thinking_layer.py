"""
Agent Thinking Layer with Technology Network Graph

Visualizes domain knowledge and reasoning:
- Technology relationship network
- Trade-off analysis visualization
- Decision confidence mapping
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class TechnologyNode:
    """A technology in the network."""
    name: str
    category: str
    complexity: str  # low, medium, high
    best_for: List[str] = field(default_factory=list)
    not_ideal_for: List[str] = field(default_factory=list)


@dataclass
class TechnologyRelation:
    """A relationship between technologies."""
    source: str
    target: str
    relation_type: str  # alternative_to, upgrades_to, pairs_with, conflicts_with
    strength: float = 0.5  # 0-1 strength of relationship
    context: str = ""


class TechnologyNetwork:
    """
    Graph of technology relationships and trade-offs.

    Captures:
    - Alternative technologies
    - Upgrade paths
    - Complementary pairings
    - Known conflicts
    """

    def __init__(self):
        self.technologies: Dict[str, TechnologyNode] = {}
        self.relations: List[TechnologyRelation] = []
        self._build_default_network()

    def _build_default_network(self):
        """Build the default technology network from knowledge base."""
        # Database technologies
        self._add_tech("PostgreSQL", "database", "medium",
                       ["tabular_data", "acid_transactions", "complex_joins"],
                       ["horizontal_scaling", "document_storage"])
        self._add_tech("MongoDB", "database", "medium",
                       ["document_storage", "flexible_schema", "horizontal_scaling"],
                       ["complex_joins", "acid_transactions", "tabular_data"])
        self._add_tech("Neo4j", "database", "medium",
                       ["graph_data", "relationship_queries", "path_finding"],
                       ["tabular_data", "simple_crud"])
        self._add_tech("Redis", "database", "low",
                       ["caching", "session_storage", "real_time"],
                       ["primary_storage", "complex_queries"])

        # ML technologies
        self._add_tech("XGBoost", "ml_model", "medium",
                       ["tabular_data", "interpretability", "feature_importance"],
                       ["image_data", "sequential_data"])
        self._add_tech("CNN", "ml_model", "high",
                       ["image_data", "spatial_patterns"],
                       ["tabular_data", "interpretability"])
        self._add_tech("Linear Regression", "ml_model", "low",
                       ["tabular_data", "interpretability", "baseline"],
                       ["complex_patterns", "image_data"])

        # Architecture patterns
        self._add_tech("Monolith", "architecture", "low",
                       ["small_team", "mvp", "simple_deployment"],
                       ["large_teams", "independent_scaling"])
        self._add_tech("Microservices", "architecture", "high",
                       ["large_teams", "independent_scaling"],
                       ["small_team", "mvp", "tight_timeline"])
        self._add_tech("Modular Monolith", "architecture", "medium",
                       ["medium_teams", "domain_boundaries"],
                       ["polyglot_persistence"])

        # DevOps
        self._add_tech("Kubernetes", "devops", "high",
                       ["large_scale", "auto_scaling", "multi_service"],
                       ["small_team", "single_service", "mvp"])
        self._add_tech("Docker Compose", "devops", "low",
                       ["small_deployment", "development", "simple_orchestration"],
                       ["production_at_scale", "auto_scaling"])

        # Add relationships
        self._add_relation("PostgreSQL", "MongoDB", "alternative_to", 0.7,
                           "Different paradigm for data storage")
        self._add_relation("PostgreSQL", "Neo4j", "pairs_with", 0.6,
                           "Can complement for hybrid workloads")
        self._add_relation("MongoDB", "Neo4j", "alternative_to", 0.8,
                           "Both handle non-relational data")
        self._add_relation("XGBoost", "CNN", "alternative_to", 0.6,
                           "Different approaches for different data types")
        self._add_relation("Linear Regression", "XGBoost", "upgrades_to", 0.8,
                           "Natural progression for complexity")
        self._add_relation("Monolith", "Modular Monolith", "upgrades_to", 0.9,
                           "Structured evolution path")
        self._add_relation("Modular Monolith", "Microservices", "upgrades_to", 0.7,
                           "Further decomposition when needed")
        self._add_relation("Docker Compose", "Kubernetes", "upgrades_to", 0.8,
                           "Scale-out evolution")

    def _add_tech(self, name: str, category: str, complexity: str,
                  best_for: List[str], not_ideal_for: List[str]):
        """Add a technology to the network."""
        self.technologies[name] = TechnologyNode(
            name=name,
            category=category,
            complexity=complexity,
            best_for=best_for,
            not_ideal_for=not_ideal_for
        )

    def _add_relation(self, source: str, target: str, relation_type: str,
                      strength: float, context: str):
        """Add a relationship between technologies."""
        self.relations.append(TechnologyRelation(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=strength,
            context=context
        ))

    def get_alternatives(self, tech_name: str) -> List[Tuple[str, float, str]]:
        """Get alternative technologies with relationship strength."""
        alternatives = []
        for rel in self.relations:
            if rel.source == tech_name and rel.relation_type == "alternative_to":
                alternatives.append((rel.target, rel.strength, rel.context))
            elif rel.target == tech_name and rel.relation_type == "alternative_to":
                alternatives.append((rel.source, rel.strength, rel.context))
        return alternatives

    def get_upgrade_path(self, tech_name: str) -> List[Tuple[str, str]]:
        """Get upgrade path from a technology."""
        path = []
        for rel in self.relations:
            if rel.source == tech_name and rel.relation_type == "upgrades_to":
                path.append((rel.target, rel.context))
        return path

    def get_complementary(self, tech_name: str) -> List[Tuple[str, str]]:
        """Get technologies that pair well."""
        pairs = []
        for rel in self.relations:
            if rel.relation_type == "pairs_with":
                if rel.source == tech_name:
                    pairs.append((rel.target, rel.context))
                elif rel.target == tech_name:
                    pairs.append((rel.source, rel.context))
        return pairs

    def to_mermaid(self) -> str:
        """Export network as Mermaid diagram."""
        lines = ["graph LR"]

        # Add nodes by category
        categories = {}
        for name, tech in self.technologies.items():
            if tech.category not in categories:
                categories[tech.category] = []
            categories[tech.category].append(name)

        # Subgraphs by category
        for category, techs in categories.items():
            lines.append(f"    subgraph {category.upper()}")
            for tech in techs:
                safe_name = tech.replace(" ", "_")
                lines.append(f'        {safe_name}["{tech}"]')
            lines.append("    end")

        # Add edges
        edge_styles = {
            "alternative_to": "-.->|alternative|",
            "upgrades_to": "==>|upgrades to|",
            "pairs_with": "-->|pairs with|",
            "conflicts_with": "x-->|conflicts|"
        }

        for rel in self.relations:
            source = rel.source.replace(" ", "_")
            target = rel.target.replace(" ", "_")
            arrow = edge_styles.get(rel.relation_type, "-->")
            lines.append(f"    {source} {arrow} {target}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export network as JSON."""
        return json.dumps({
            "technologies": [
                {
                    "name": t.name,
                    "category": t.category,
                    "complexity": t.complexity,
                    "best_for": t.best_for,
                    "not_ideal_for": t.not_ideal_for
                }
                for t in self.technologies.values()
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.relation_type,
                    "strength": r.strength,
                    "context": r.context
                }
                for r in self.relations
            ]
        }, indent=2)


@dataclass
class ThinkingStep:
    """A step in the agent's reasoning process."""
    step_number: int
    agent_name: str
    action: str
    reasoning: str
    confidence: float
    evidence: List[str] = field(default_factory=list)


class AgentThinkingLayer:
    """
    Captures and visualizes agent reasoning process.

    Shows:
    - What each agent considered
    - Confidence at each step
    - Evidence used for decisions
    - Domain knowledge applied
    """

    def __init__(self):
        self.thinking_steps: List[ThinkingStep] = []
        self.network = TechnologyNetwork()
        self.confidence_trace: List[Tuple[str, float]] = []

    def add_thinking_step(self, agent_name: str, action: str,
                          reasoning: str, confidence: float,
                          evidence: Optional[List[str]] = None):
        """Record a thinking step."""
        step = ThinkingStep(
            step_number=len(self.thinking_steps) + 1,
            agent_name=agent_name,
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            evidence=evidence or []
        )
        self.thinking_steps.append(step)
        self.confidence_trace.append((agent_name, confidence))

    def build_from_state(self, state: Dict[str, Any]) -> "AgentThinkingLayer":
        """Build thinking trace from governance state."""
        # Interceptor thinking
        tech = state.get("technology_requested", "Unknown")
        category = state.get("technology_category", "unknown")
        self.add_thinking_step(
            agent_name="Interceptor",
            action=f"Identified technology: {tech}",
            reasoning=f"Classified as {category} based on request keywords",
            confidence=0.9 if tech else 0.3,
            evidence=[state.get("user_request", "")]
        )

        # Why Validator thinking
        requirements = state.get("extracted_requirements", [])
        why_questions = state.get("why_questions", [])
        why_responses = state.get("why_responses", [])

        for i, (q, r) in enumerate(zip(why_questions, why_responses)):
            reqs_found = requirements[i:i+2] if i < len(requirements) else []
            self.add_thinking_step(
                agent_name="Why Validator",
                action=f"Why #{i+1}: Probed requirements",
                reasoning=f"Asked about specific need to uncover true requirements",
                confidence=0.6 + (0.1 * len(reqs_found)),
                evidence=[f"Q: {q[:50]}...", f"A: {r[:50]}..."]
            )

        # Evaluator thinking
        match_score = state.get("match_score", 0)
        mismatches = state.get("mismatches", [])
        self.add_thinking_step(
            agent_name="Evaluator",
            action=f"Evaluated {tech} against requirements",
            reasoning=f"Matched {len(requirements)} requirements, found {len(mismatches)} conflicts",
            confidence=match_score,
            evidence=[f"Match score: {match_score:.0%}"] + mismatches[:3]
        )

        # Recommender thinking (if alternatives suggested)
        recommended = state.get("recommended_technology")
        if recommended and recommended != tech:
            alternatives = self.network.get_alternatives(tech)
            self.add_thinking_step(
                agent_name="Recommender",
                action=f"Recommended {recommended}",
                reasoning=state.get("recommendation_rationale", "Better fit for requirements"),
                confidence=0.85,
                evidence=[f"Original: {tech}", f"Alternative: {recommended}"]
            )

        return self

    def get_thinking_trace(self) -> str:
        """Get formatted thinking trace."""
        lines = []
        lines.append("🧠 AGENT THINKING TRACE")
        lines.append("=" * 60)

        for step in self.thinking_steps:
            lines.append("")
            lines.append(f"Step {step.step_number}: [{step.agent_name}]")
            lines.append(f"  Action: {step.action}")
            lines.append(f"  Reasoning: {step.reasoning}")
            lines.append(f"  Confidence: {'█' * int(step.confidence * 10)}{'░' * (10 - int(step.confidence * 10))} {step.confidence:.0%}")
            if step.evidence:
                lines.append("  Evidence:")
                for e in step.evidence[:3]:
                    lines.append(f"    - {e}")

        return "\n".join(lines)

    def get_confidence_chart(self) -> str:
        """Get ASCII confidence chart."""
        lines = []
        lines.append("📊 CONFIDENCE TRACE")
        lines.append("-" * 40)

        for agent, conf in self.confidence_trace:
            bar_len = int(conf * 30)
            bar = "▓" * bar_len + "░" * (30 - bar_len)
            lines.append(f"{agent:15} |{bar}| {conf:.0%}")

        return "\n".join(lines)

    def get_domain_knowledge_applied(self, tech_name: str) -> str:
        """Show domain knowledge applied for a technology."""
        lines = []
        lines.append(f"📚 DOMAIN KNOWLEDGE: {tech_name}")
        lines.append("-" * 40)

        if tech_name in [t.name for t in self.network.technologies.values()]:
            # Alternatives
            alts = self.network.get_alternatives(tech_name)
            if alts:
                lines.append("Alternatives considered:")
                for alt, strength, ctx in alts:
                    lines.append(f"  → {alt} ({strength:.0%} relevance): {ctx}")

            # Upgrade path
            upgrades = self.network.get_upgrade_path(tech_name)
            if upgrades:
                lines.append("Upgrade path:")
                for up, ctx in upgrades:
                    lines.append(f"  ⬆ {up}: {ctx}")

            # Complementary
            pairs = self.network.get_complementary(tech_name)
            if pairs:
                lines.append("Pairs well with:")
                for pair, ctx in pairs:
                    lines.append(f"  + {pair}: {ctx}")
        else:
            lines.append("  (Technology not in knowledge network)")

        return "\n".join(lines)
