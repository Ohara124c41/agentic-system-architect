"""
Why Validator Agent

Implements INCOSE's "Five Whys" methodology to uncover true requirements
before allowing solution commitment.

This is the core innovation of the architecture governance system.
"""

from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import random

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.state import AgentState, ConversationTurn
from ..schemas.responses import WhyResponse, RequirementsProfile
from ..prompts import (
    WHY_VALIDATOR_SYSTEM_PROMPT,
    WHY_QUESTION_TEMPLATES,
    ANALYZE_WHY_RESPONSE_PROMPT,
)
from ..tools.knowledge_base import get_technology_profile


class WhyValidatorAgent:
    """
    Why Validator Agent - Implements INCOSE Five Whys

    Responsibilities:
    1. Generate contextually appropriate "why" questions
    2. Analyze user responses to extract requirements
    3. Build a requirements profile from the conversation
    4. Determine when enough information has been gathered
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, max_whys: int = 5):
        """
        Initialize the Why Validator agent.

        Args:
            llm: LangChain ChatOpenAI instance (for analyzing responses)
            max_whys: Maximum number of why questions to ask (default: 5)
        """
        self.llm = llm
        self.name = "why_validator"
        self.max_whys = max_whys
        # Track used question templates to avoid repetition
        self._used_templates: Set[str] = set()

    def generate_why_question(self, state: AgentState) -> str:
        """
        Generate the next "why" question based on current state.

        Uses templates first (no API cost), falls back to LLM if needed.
        Ensures variety by tracking previously asked questions.
        """
        technology = state.get("technology_requested", "this technology")
        category = state.get("technology_category", "database")
        iteration = state.get("why_iteration", 0) + 1
        previous_responses = state.get("why_responses", [])
        previous_questions = state.get("why_questions", [])

        # Get templates for this category
        templates = WHY_QUESTION_TEMPLATES.get(category, WHY_QUESTION_TEMPLATES["database"])

        # Select appropriate why level (1-5)
        why_key = f"why_{min(iteration, 5)}"
        template_list = templates.get(why_key, templates.get("why_1", []))

        # Get alternative technology for questions that need it
        profile = get_technology_profile(technology) if technology else None
        alternatives = profile.get("typical_alternatives", ["a simpler solution"]) if profile else ["a simpler solution"]

        # Rotate through alternatives to add variety
        alt_index = (iteration - 1) % len(alternatives) if alternatives else 0
        alternative = alternatives[alt_index] if alternatives else "a simpler solution"

        # Get stated reason from previous response
        stated_reason = previous_responses[-1][:100] if previous_responses else "your choice"

        # Find a template that hasn't been used yet
        available_templates = [t for t in template_list if t not in self._used_templates]

        # If all templates used, try templates from adjacent why levels for variety
        if not available_templates:
            # Try to get fresh templates from other why levels
            for fallback_key in [f"why_{i}" for i in range(1, 6) if i != iteration]:
                fallback_list = templates.get(fallback_key, [])
                available_templates = [t for t in fallback_list if t not in self._used_templates]
                if available_templates:
                    break

        # Last resort: reuse templates but pick different from last question
        if not available_templates:
            available_templates = [t for t in template_list if t not in previous_questions[-1:]]
            if not available_templates:
                available_templates = template_list

        # Select template
        template = random.choice(available_templates) if available_templates else template_list[0]
        self._used_templates.add(template)

        # Format the question
        try:
            question = template.format(
                technology=technology,
                stated_reason=stated_reason,
                alternative=alternative,
                alternative_benefits="simplicity and lower overhead"
            )
        except KeyError:
            # Handle templates that don't need all variables
            question = template.replace("{technology}", technology)
            question = question.replace("{stated_reason}", stated_reason)
            question = question.replace("{alternative}", alternative)
            question = question.replace("{alternative_benefits}", "simplicity and lower overhead")

        return question

    def process_initial(self, state: AgentState) -> AgentState:
        """
        Start the why questioning process.

        Called when entering the why_validator from interceptor.
        """
        # Reset used templates for new session
        self._used_templates.clear()

        # Generate first why question
        question = self.generate_why_question(state)

        # Update state
        state["why_questions"].append(question)
        state["why_iteration"] = 1
        state["awaiting_user_response"] = True

        # Add to conversation
        now = datetime.now().isoformat()
        state["conversation_history"].append(
            ConversationTurn(
                role="agent",
                content=question,
                timestamp=now,
                agent_name=self.name
            )
        )

        state["current_agent"] = self.name
        state["updated_at"] = now

        return state

    def process_response(self, state: AgentState, user_response: str) -> AgentState:
        """
        Process user's response to a why question.

        Analyzes the response, extracts requirements, and decides
        whether to ask another why or proceed to evaluation.
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
        state["why_responses"].append(user_response)
        state["awaiting_user_response"] = False

        # Analyze the response
        analysis = self._analyze_response(state, user_response)

        if analysis:
            # Extract requirements from analysis - avoid duplicates
            existing_reqs = set(state.get("extracted_requirements", []))
            new_reqs = [r for r in analysis.extracted_requirements if r not in existing_reqs]
            state["extracted_requirements"].extend(new_reqs)

            existing_implicit = set(state.get("implicit_needs", []))
            new_implicit = [n for n in analysis.implicit_needs if n not in existing_implicit]
            state["implicit_needs"].extend(new_implicit)

            state["stated_reasons"].append(analysis.understood_reason)
            state["total_llm_calls"] = state.get("total_llm_calls", 0) + 1

        # Decide next action
        current_iteration = state.get("why_iteration", 1)
        total_requirements = len(state.get("extracted_requirements", []))

        # Determine if we should continue asking or proceed to evaluation
        should_continue = self._should_continue_asking(
            current_iteration=current_iteration,
            total_requirements=total_requirements,
            analysis=analysis,
            user_response=user_response
        )

        if not should_continue or current_iteration >= self.max_whys:
            # Proceed to evaluation
            state["next_agent"] = "evaluator"
            state["awaiting_user_response"] = False

            # Add transition message
            transition_msg = self._get_transition_message(total_requirements)
            state["conversation_history"].append(
                ConversationTurn(
                    role="agent",
                    content=transition_msg,
                    timestamp=now,
                    agent_name=self.name
                )
            )
        else:
            # Ask another why
            state["why_iteration"] = current_iteration + 1
            question = self.generate_why_question(state)
            state["why_questions"].append(question)
            state["awaiting_user_response"] = True

            state["conversation_history"].append(
                ConversationTurn(
                    role="agent",
                    content=question,
                    timestamp=now,
                    agent_name=self.name
                )
            )

        state["current_agent"] = self.name
        state["updated_at"] = now

        return state

    def _should_continue_asking(
        self,
        current_iteration: int,
        total_requirements: int,
        analysis: Optional[WhyResponse],
        user_response: str
    ) -> bool:
        """
        Determine if we should ask another why question.

        Considers:
        - Number of requirements gathered
        - Whether user seems satisfied/convinced
        - Analysis recommendation
        """
        # If user indicates agreement/understanding, stop asking
        agreement_signals = [
            "i see", "makes sense", "you're right", "that's true",
            "i understand", "good point", "fair enough", "i agree",
            "what would you recommend", "what do you suggest"
        ]
        response_lower = user_response.lower()
        if any(signal in response_lower for signal in agreement_signals):
            return False

        # If enough requirements exist (3+), evaluation can proceed
        if total_requirements >= 3 and current_iteration >= 2:
            return False

        # If analysis says no follow-up needed
        if analysis and not analysis.follow_up_needed:
            return False

        # Minimum 2 questions before evaluating
        if current_iteration < 2:
            return True

        # Default: continue if under max
        return current_iteration < self.max_whys

    def _get_transition_message(self, total_requirements: int) -> str:
        """Get appropriate transition message based on context."""
        messages = [
            "Thank you for those details. Let me evaluate how well your chosen technology matches these requirements...",
            "I have a good understanding of your requirements now. Let me assess the technology fit...",
            "Based on our conversation, I can now evaluate whether this technology aligns with your needs...",
            "Great, I've gathered enough context. Let me analyze the match between your requirements and the technology...",
        ]
        return random.choice(messages)

    def _analyze_response(self, state: AgentState, user_response: str) -> Optional[WhyResponse]:
        """
        Analyze user's response to extract requirements.

        Uses LLM to understand semantic meaning and extract requirements.
        """
        if not self.llm:
            # Fallback: Simple keyword extraction
            return self._simple_extraction(user_response)

        # Build context from previous conversation
        previous_context = "\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in zip(state.get("why_questions", []), state.get("why_responses", []))
        ])

        # Create structured output chain
        structured_llm = self.llm.with_structured_output(WhyResponse)

        prompt = ChatPromptTemplate.from_messages([
            ("system", WHY_VALIDATOR_SYSTEM_PROMPT),
            ("human", ANALYZE_WHY_RESPONSE_PROMPT),
        ])

        chain = prompt | structured_llm

        try:
            result = chain.invoke({
                "technology": state.get("technology_requested", "unknown"),
                "category": state.get("technology_category", "unknown"),
                "why_question": state.get("why_questions", [""])[-1],
                "user_response": user_response,
                "previous_context": previous_context,
            })
            return result
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._simple_extraction(user_response)

    def _simple_extraction(self, response: str) -> WhyResponse:
        """
        Simple keyword-based requirement extraction.

        Fallback when LLM is not available.
        """
        response_lower = response.lower()

        # Keyword to requirement mapping - expanded
        keyword_requirements = {
            # Database requirements - aligned with PostgreSQL best_for
            "transaction": "acid_transactions",
            "transactions": "acid_transactions",
            "acid": "acid_transactions",
            "join": "complex_queries",
            "joins": "complex_queries",
            "complex join": "complex_queries",
            "relational": "relational_data",
            "foreign key": "relational_data",
            "normalize": "relational_data",
            "report": "reporting_systems",
            "reports": "reporting_systems",
            "reporting": "reporting_systems",
            "monthly report": "reporting_systems",
            "financial": "financial_data",
            "finance": "financial_data",
            "accuracy": "data_integrity",
            "accurate": "data_integrity",
            "integrity": "data_integrity",

            # Scale requirements
            "scale": "horizontal_scaling",
            "scalab": "horizontal_scaling",
            "million": "large_scale",
            "billion": "large_scale",

            # Schema requirements
            "flexible": "flexible_schema",
            "schema change": "flexible_schema",
            "document": "document_storage",

            # Data type requirements
            "tabular": "tabular_data",
            "spreadsheet": "tabular_data",
            "csv": "tabular_data",
            "rows and columns": "tabular_data",
            "structured": "tabular_data",
            "column": "tabular_data",
            "image": "image_data",
            "picture": "image_data",
            "photo": "image_data",
            "visual": "image_data",

            # ML requirements
            "interpret": "interpretability",
            "explain": "interpretability",
            "business": "interpretability",
            "feature importance": "interpretability",

            # Operational requirements
            "simple": "simplicity",
            "easy": "simplicity",
            "fast": "quick_deployment",
            "quick": "quick_deployment",
            "deadline": "tight_timeline",
            "weeks": "tight_timeline",
            "mvp": "mvp_stage",
            "prototype": "mvp_stage",

            # Team requirements
            "team": "team_context",
            "developer": "team_context",
            "just me": "small_team",
            "solo": "small_team",
            "small team": "small_team",
            "no experience": "low_expertise",

            # Graph requirements - aligned with Neo4j best_for
            "graph": "graph_data",
            "network": "network_analysis",
            "social network": "social_networks",
            "relationship": "relationship_heavy_queries",
            "relationships": "relationship_heavy_queries",
            "connection": "connected_data",
            "connections": "connected_data",
            "edge": "edges_as_first_class",
            "edges": "edges_as_first_class",
            "node": "graph_data",
            "nodes": "graph_data",
            "embedding": "graph_data",
            "path": "path_finding",
            "paths": "path_finding",
            "pathfinding": "path_finding",
            "traversal": "graph_traversal",
            "cluster": "network_analysis",
            "recommend": "recommendation_engines",

            # Architecture requirements
            "crud": "simple_crud_operations",
            "inventory": "inventory_tracking",
            "report": "reporting_needs",
            "dashboard": "dashboard_needs",
            "embed": "embedded_analytics",
            "white-label": "white_label_needs",
            "saas": "saas_product",

            # Infrastructure requirements - aligned with Docker Compose/K8s best_for
            "container": "small_deployments",
            "containers": "small_deployments",
            "docker": "small_deployments",
            "2-3 containers": "small_deployments",
            "few containers": "small_deployments",
            "service": "single_host",
            "one service": "single_host",
            "single service": "single_host",
            "deploy": "simple_orchestration",
            "quick deploy": "simple_orchestration",
            "ship": "simple_orchestration",
            "2 weeks": "simple_orchestration",
            "no experience": "learning_containers",
            "no k8s experience": "learning_containers",
            "small team": "small_deployments",
        }

        extracted = []
        for keyword, requirement in keyword_requirements.items():
            if keyword in response_lower:
                extracted.append(requirement)

        # Determine if follow-up is needed
        # More requirements extracted = less need for follow-up
        follow_up_needed = len(set(extracted)) < 2

        return WhyResponse(
            understood_reason=response[:200],
            extracted_requirements=list(set(extracted)),
            implicit_needs=[],
            follow_up_needed=follow_up_needed,
            suggested_follow_up=None
        )

    def build_requirements_profile(self, state: AgentState) -> RequirementsProfile:
        """
        Build a complete requirements profile from all gathered information.
        """
        all_requirements = list(set(state.get("extracted_requirements", [])))
        constraints = []

        # Extract constraints from context
        context = state.get("request_context", "")
        if context:
            if "small" in context.lower() or "solo" in context.lower():
                constraints.append("small_team")
            if "urgent" in context.lower():
                constraints.append("tight_timeline")
            if "scale:large" in context:
                constraints.append("large_scale")

        return RequirementsProfile(
            all_requirements=all_requirements,
            primary_needs=all_requirements[:5],  # Top 5
            constraints=constraints,
            stated_preferences=state.get("stated_reasons", []),
            data_characteristics=self._extract_data_type(all_requirements),
            scale_requirements=self._extract_scale(constraints),
        )

    def _extract_data_type(self, requirements: List[str]) -> Optional[str]:
        """Extract data type from requirements."""
        data_types = {
            "tabular_data": "Tabular/structured data",
            "image_data": "Image/visual data",
            "document_data": "Document/JSON data",
            "relational_data": "Relational data with joins",
            "time_series": "Time series data",
            "graph_data": "Graph/network data",
        }
        for req in requirements:
            if req in data_types:
                return data_types[req]
        return None

    def _extract_scale(self, constraints: List[str]) -> Optional[str]:
        """Extract scale requirements from constraints."""
        if "large_scale" in constraints:
            return "Large scale / high volume"
        if "small_team" in constraints:
            return "Small scale / MVP"
        return None


def create_why_validator_node(llm: Optional[ChatOpenAI] = None, max_whys: int = 5):
    """
    Factory function to create why_validator node for LangGraph.

    Usage in LangGraph:
        workflow.add_node("validate", create_why_validator_node(llm))
    """
    agent = WhyValidatorAgent(llm, max_whys)

    def node_function(state: AgentState) -> AgentState:
        # Check if this is initial entry or processing a response
        if state.get("why_iteration", 0) == 0:
            return agent.process_initial(state)
        else:
            # Get the latest user response (should be last item in history if it's from user)
            history = state.get("conversation_history", [])
            if history and history[-1]["role"] == "user":
                return agent.process_response(state, history[-1]["content"])
            return state

    return node_function
