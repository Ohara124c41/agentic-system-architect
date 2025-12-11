"""
Ilities Analyst Agent

Implements the "architect's second hat" - after a recommendation is made,
this agent analyzes quality attribute (-ilities) trade-offs and surfaces
risks that the development team should consider.

Recommending a simpler solution (e.g., monolith) without discussing future
extensibility, maintainability, and scalability risks provides incomplete
architectural advice.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.state import AgentState, ConversationTurn
from ..schemas.responses import IlitiesAnalysis
from ..tools.knowledge_base import get_technology_profile


# Quality attributes and their relationships/tensions
ILITY_RELATIONSHIPS = {
    "scalability": {
        "tensions_with": ["simplicity", "cost_efficiency", "time_to_market"],
        "supports": ["performance", "availability"],
        "description": "Ability to handle growth in users, data, or transactions",
    },
    "maintainability": {
        "tensions_with": ["time_to_market", "performance"],
        "supports": ["extensibility", "testability"],
        "description": "Ease of fixing bugs and making changes over time",
    },
    "extensibility": {
        "tensions_with": ["simplicity", "time_to_market"],
        "supports": ["maintainability", "flexibility"],
        "description": "Ability to add new features without major rewrites",
    },
    "reliability": {
        "tensions_with": ["cost_efficiency", "time_to_market"],
        "supports": ["availability", "security"],
        "description": "System performs correctly under stated conditions",
    },
    "availability": {
        "tensions_with": ["cost_efficiency", "simplicity"],
        "supports": ["reliability", "scalability"],
        "description": "System is operational when needed",
    },
    "performance": {
        "tensions_with": ["maintainability", "portability", "cost_efficiency"],
        "supports": ["scalability", "usability"],
        "description": "Response time, throughput, resource utilization",
    },
    "security": {
        "tensions_with": ["usability", "performance", "time_to_market"],
        "supports": ["reliability", "compliance"],
        "description": "Protection against unauthorized access and attacks",
    },
    "testability": {
        "tensions_with": ["time_to_market", "simplicity"],
        "supports": ["maintainability", "reliability"],
        "description": "Ease of validating system behavior",
    },
    "portability": {
        "tensions_with": ["performance", "simplicity"],
        "supports": ["flexibility", "vendor_independence"],
        "description": "Ability to run in different environments",
    },
    "usability": {
        "tensions_with": ["security", "complexity"],
        "supports": ["adoption", "productivity"],
        "description": "Ease of use for end users and developers",
    },
}

# Technology-specific ility profiles
TECHNOLOGY_ILITY_PROFILES = {
    # Architecture patterns
    "monolith": {
        "strong": ["simplicity", "time_to_market", "testability", "debuggability"],
        "weak": ["scalability", "extensibility", "team_autonomy", "fault_isolation"],
        "risks": [
            "Technical debt accumulates as codebase grows",
            "Scaling requires scaling entire application, not just bottlenecks",
            "Large team coordination becomes difficult over time",
            "Single point of failure - one bug can bring down entire system",
            "Technology lock-in - hard to adopt new frameworks incrementally",
        ],
        "mitigations": [
            "Establish clear module boundaries from day one",
            "Plan migration path to modular monolith at specific triggers (team size, complexity)",
            "Implement comprehensive test coverage to enable future refactoring",
            "Document architectural decisions for future team members",
        ],
    },
    "modular_monolith": {
        "strong": ["maintainability", "extensibility", "testability", "simplicity"],
        "weak": ["independent_scaling", "technology_diversity", "team_autonomy"],
        "risks": [
            "Module boundaries may erode under delivery pressure",
            "Still requires coordinated deployments",
            "Database schema changes affect entire application",
            "May delay necessary microservices migration",
        ],
        "mitigations": [
            "Enforce module boundaries with architectural fitness functions",
            "Use separate schemas/tables per module where possible",
            "Define clear API contracts between modules",
            "Establish metrics to trigger microservices extraction",
        ],
    },
    "microservices": {
        "strong": ["scalability", "extensibility", "fault_isolation", "team_autonomy"],
        "weak": ["simplicity", "debuggability", "consistency", "operational_overhead"],
        "risks": [
            "Distributed system complexity (network failures, latency)",
            "Data consistency challenges across services",
            "Operational overhead (monitoring, deployment, debugging)",
            "Service coordination and versioning complexity",
        ],
        "mitigations": [
            "Invest in observability (distributed tracing, centralized logging)",
            "Implement circuit breakers and retry policies",
            "Establish clear service ownership and API versioning",
            "Use event sourcing or saga patterns for distributed transactions",
        ],
    },
    # Databases
    "postgresql": {
        "strong": ["data_integrity", "consistency", "query_flexibility", "reliability"],
        "weak": ["horizontal_scaling", "schema_flexibility", "write_throughput"],
        "risks": [
            "Vertical scaling has limits - may need sharding strategy later",
            "Schema migrations can be painful at scale",
            "Connection pooling becomes critical under load",
        ],
        "mitigations": [
            "Design schema with future partitioning in mind",
            "Use connection pooling (PgBouncer) from the start",
            "Implement read replicas for read-heavy workloads",
            "Plan for logical replication if multi-region needed",
        ],
    },
    "mongodb": {
        "strong": ["schema_flexibility", "horizontal_scaling", "developer_velocity"],
        "weak": ["data_integrity", "complex_queries", "consistency"],
        "risks": [
            "Schema drift without governance leads to data quality issues",
            "Aggregation pipeline complexity for reporting",
            "Transaction support is limited compared to RDBMS",
        ],
        "mitigations": [
            "Implement schema validation rules",
            "Use ODM with schema definitions (Mongoose, etc.)",
            "Design for eventual consistency from the start",
            "Consider PostgreSQL for financial/transactional data",
        ],
    },
    "neo4j": {
        "strong": ["relationship_queries", "graph_traversal", "pattern_matching"],
        "weak": ["bulk_operations", "aggregations", "operational_simplicity"],
        "risks": [
            "Operational expertise is harder to find",
            "Not suitable for non-graph workloads (may need polyglot persistence)",
            "Memory requirements for large graphs",
            "Limited ecosystem compared to relational databases",
        ],
        "mitigations": [
            "Use Neo4j for graph-specific queries, keep other data in PostgreSQL",
            "Plan for graph data modeling expertise or training",
            "Implement caching for frequently accessed subgraphs",
            "Consider managed Neo4j (Aura) to reduce ops burden",
        ],
    },
    # DevOps
    "docker_compose": {
        "strong": ["simplicity", "local_development", "learning_curve", "cost"],
        "weak": ["production_readiness", "high_availability", "auto_scaling"],
        "risks": [
            "Not suitable for production at scale without orchestration",
            "No built-in health checks, rolling updates, or self-healing",
            "Single host limitation - no multi-node clustering",
            "Team may outgrow it quickly if successful",
        ],
        "mitigations": [
            "Plan migration path to Kubernetes/ECS at specific scale triggers",
            "Implement external health monitoring",
            "Use Docker Swarm as intermediate step if needed",
            "Document deployment procedures for future orchestration",
        ],
    },
    "kubernetes": {
        "strong": ["scalability", "self_healing", "declarative_config", "ecosystem"],
        "weak": ["complexity", "learning_curve", "operational_overhead", "cost"],
        "risks": [
            "Significant learning curve for team",
            "Over-engineering for small applications",
            "Operational complexity requires dedicated expertise",
            "Cost of managed K8s or self-managed clusters",
        ],
        "mitigations": [
            "Start with managed Kubernetes (EKS, GKE, AKS)",
            "Use GitOps for deployment automation",
            "Invest in team training before adoption",
            "Consider simpler alternatives until scale justifies complexity",
        ],
    },
    # ML Models
    "random_forest": {
        "strong": ["interpretability", "robustness", "handling_missing_data"],
        "weak": ["real_time_inference", "memory_footprint", "extrapolation"],
        "risks": [
            "May not capture complex non-linear patterns",
            "Ensemble size affects inference latency",
            "Not suitable for online learning",
        ],
        "mitigations": [
            "Benchmark against gradient boosting (XGBoost/LightGBM)",
            "Use feature importance for model explanations",
            "Consider model compression for production deployment",
        ],
    },
    "linear_regression": {
        "strong": ["interpretability", "simplicity", "speed", "statistical_rigor"],
        "weak": ["non_linear_patterns", "complex_interactions", "outlier_sensitivity"],
        "risks": [
            "May underfit complex relationships",
            "Assumptions (linearity, normality) may not hold",
            "Feature engineering burden shifts to data scientists",
        ],
        "mitigations": [
            "Use as baseline before trying complex models",
            "Implement regularization (Ridge/Lasso) for stability",
            "Monitor residuals for pattern violations",
            "Document when to escalate to more complex models",
        ],
    },
    # Visualization
    "custom_dashboard": {
        "strong": ["flexibility", "branding", "integration", "customization"],
        "weak": ["development_time", "maintenance_burden", "feature_parity"],
        "risks": [
            "Significant development and maintenance investment",
            "Need to build features that come free with off-the-shelf tools",
            "Requires frontend expertise on team",
            "Security and accessibility compliance burden",
        ],
        "mitigations": [
            "Use established charting libraries (D3, Chart.js, Plotly)",
            "Plan for ongoing maintenance budget",
            "Implement comprehensive testing for visualizations",
            "Consider hybrid: embedded BI tool with custom wrapper",
        ],
    },
}


class IlitiesAnalystAgent:
    """
    Ilities Analyst Agent - The Architect's Second Hat

    After a recommendation is made, this agent:
    1. Identifies quality attribute trade-offs
    2. Surfaces risks specific to the recommended technology
    3. Provides mitigations to address those risks
    4. Considers future evolution scenarios

    Recommendations include appropriate caveats, matching how a senior
    architect would communicate decisions.
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the Ilities Analyst agent.

        Args:
            llm: Optional LLM for generating contextual risk analysis
        """
        self.llm = llm
        self.name = "ilities_analyst"

    def analyze(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze quality attribute trade-offs for the recommendation.

        This is called after the Recommender agent has made its suggestion.
        """
        recommended_tech = state.get("recommended_technology") or state.get("technology_requested")
        original_tech = state.get("technology_requested", "")
        requirements = state.get("extracted_requirements", [])
        category = state.get("technology_category", "")

        # Get ility profile for recommended technology
        tech_key = recommended_tech.lower().replace(" ", "_")
        ility_profile = TECHNOLOGY_ILITY_PROFILES.get(tech_key, {})

        # Analyze trade-offs
        analysis = {
            "technology": recommended_tech,
            "strong_ilities": ility_profile.get("strong", []),
            "weak_ilities": ility_profile.get("weak", []),
            "risks": ility_profile.get("risks", []),
            "mitigations": ility_profile.get("mitigations", []),
            "evolution_triggers": self._generate_evolution_triggers(tech_key, category),
            "monitoring_recommendations": self._generate_monitoring_recommendations(tech_key),
        }

        # Add context-specific risks based on extracted requirements
        context_risks = self._analyze_context_risks(tech_key, requirements)
        if context_risks:
            analysis["context_specific_risks"] = context_risks

        # Generate the architect's caveat message
        analysis["architect_caveat"] = self._generate_caveat_message(
            recommended_tech, original_tech, analysis
        )

        return analysis

    def _generate_evolution_triggers(self, tech_key: str, category: str) -> List[str]:
        """Generate triggers that indicate when to reconsider this technology choice."""
        triggers = {
            "monolith": [
                "Team size exceeds 8-10 developers",
                "Deployment frequency needs exceed weekly",
                "Different components need independent scaling",
                "Time to onboard new developers exceeds 2 weeks",
            ],
            "modular_monolith": [
                "Team size exceeds 15-20 developers",
                "Need for polyglot persistence or technology diversity",
                "Module boundaries create deployment bottlenecks",
                "Independent scaling becomes critical",
            ],
            "docker_compose": [
                "Need for zero-downtime deployments",
                "Traffic exceeds single-host capacity",
                "Need for auto-scaling based on load",
                "Multiple environment management becomes complex",
            ],
            "postgresql": [
                "Write throughput exceeds vertical scaling limits",
                "Need for multi-region active-active deployment",
                "Schema evolution becomes blocking for releases",
            ],
            "neo4j": [
                "Non-graph query patterns dominate workload",
                "Operational complexity exceeds team capacity",
                "Cost becomes prohibitive at scale",
            ],
            "random_forest": [
                "Inference latency SLA cannot be met",
                "Model size exceeds deployment constraints",
                "Need for online learning / real-time updates",
            ],
            "custom_dashboard": [
                "Feature requests exceed development capacity",
                "Maintenance burden impacts core product development",
                "Off-the-shelf tool would cover 80%+ of needs",
            ],
        }
        return triggers.get(tech_key, [
            "Requirements significantly change from original assessment",
            "Team or scale grows beyond current solution's sweet spot",
            "Operational burden exceeds acceptable threshold",
        ])

    def _generate_monitoring_recommendations(self, tech_key: str) -> List[str]:
        """Generate what to monitor to detect when evolution is needed."""
        monitoring = {
            "monolith": [
                "Build and deployment times",
                "Time to implement new features",
                "Bug density and resolution time",
                "Developer onboarding time",
            ],
            "modular_monolith": [
                "Cross-module coupling metrics",
                "Module deployment independence",
                "Test suite execution time",
                "Circular dependency detection",
            ],
            "docker_compose": [
                "Container resource utilization",
                "Deployment success rate",
                "Recovery time from failures",
                "Host capacity metrics",
            ],
            "postgresql": [
                "Query performance (p95, p99 latency)",
                "Connection pool utilization",
                "Disk I/O and storage growth",
                "Replication lag (if using replicas)",
            ],
            "neo4j": [
                "Graph query performance",
                "Memory utilization",
                "Cache hit rates",
                "Query complexity trends",
            ],
        }
        return monitoring.get(tech_key, [
            "System performance metrics",
            "Error rates and types",
            "Resource utilization trends",
            "User satisfaction indicators",
        ])

    def _analyze_context_risks(self, tech_key: str, requirements: List[str]) -> List[str]:
        """Analyze risks specific to the context/requirements."""
        context_risks = []

        # Check for potential future needs not covered
        if tech_key == "monolith":
            if "scalability" in requirements or "horizontal_scaling" in requirements:
                context_risks.append(
                    "You mentioned scalability as a requirement - monolith will require "
                    "significant refactoring to scale horizontally. Plan your migration path early."
                )
            if "extensibility" in requirements:
                context_risks.append(
                    "Extensibility was identified as a need - establish strong module boundaries "
                    "now to avoid a 'big ball of mud' that's hard to extend later."
                )

        if tech_key == "docker_compose":
            if "reliability" in requirements or "availability" in requirements:
                context_risks.append(
                    "High availability was mentioned - Docker Compose lacks built-in HA features. "
                    "Consider external health monitoring and plan for orchestration upgrade."
                )

        if tech_key in ["postgresql", "mongodb", "neo4j"]:
            if "horizontal_scaling" in requirements:
                context_risks.append(
                    "Horizontal scaling was mentioned - design your data model with future "
                    "sharding/partitioning in mind from the start."
                )

        return context_risks

    def _generate_caveat_message(
        self,
        recommended_tech: str,
        original_tech: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Generate the architect's caveat message."""
        risks = analysis.get("risks", [])[:3]  # Top 3 risks
        mitigations = analysis.get("mitigations", [])[:3]  # Top 3 mitigations
        triggers = analysis.get("evolution_triggers", [])[:2]  # Top 2 triggers

        message = f"""
🏗️ **ARCHITECT'S ADVISORY**

While **{recommended_tech}** is recommended for your current context, consider these trade-offs:

**Key Risks:**
"""
        for risk in risks:
            message += f"  ⚠️ {risk}\n"

        message += "\n**Recommended Mitigations:**\n"
        for mitigation in mitigations:
            message += f"  ✓ {mitigation}\n"

        message += "\n**Revisit This Decision When:**\n"
        for trigger in triggers:
            message += f"  📊 {trigger}\n"

        # Add context-specific risks if any
        context_risks = analysis.get("context_specific_risks", [])
        if context_risks:
            message += "\n**Context-Specific Concerns:**\n"
            for risk in context_risks:
                message += f"  💡 {risk}\n"

        message += f"""
**Bottom Line:** {recommended_tech} is the right choice for now, but architecture is not a one-time decision.
Document this decision (see ADR in outputs/) and schedule a review when you hit the triggers above.
"""
        return message

    def process(self, state: AgentState) -> AgentState:
        """
        Process the state and add ilities analysis.

        Called after recommendation is made, before final approval.
        """
        now = datetime.now().isoformat()

        # Perform analysis
        analysis = self.analyze(state)

        # Store analysis in state
        state["ilities_analysis"] = analysis

        # Add caveat message to conversation
        caveat = analysis.get("architect_caveat", "")
        if caveat:
            state["conversation_history"].append(
                ConversationTurn(
                    role="agent",
                    content=caveat,
                    timestamp=now,
                    agent_name=self.name
                )
            )

        state["current_agent"] = self.name
        state["updated_at"] = now

        return state


def create_ilities_analyst_node(llm: Optional[ChatOpenAI] = None):
    """
    Factory function to create ilities_analyst node for LangGraph.

    Usage in LangGraph:
        workflow.add_node("analyze_ilities", create_ilities_analyst_node(llm))
    """
    agent = IlitiesAnalystAgent(llm)

    def node_function(state: AgentState) -> AgentState:
        return agent.process(state)

    return node_function
