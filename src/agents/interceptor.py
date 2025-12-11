"""
Request Interceptor Agent

First agent in the pipeline. Parses incoming requests to identify
the technology being requested and its category.

Uses rule-based classification first, falls back to LLM for ambiguous requests.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.state import AgentState, ConversationTurn
from ..schemas.responses import TechnologyClassification, InterceptorOutput
from ..tools.classification import classify_technology, extract_context_hints
from ..prompts import INTERCEPTOR_SYSTEM_PROMPT, INTERCEPTOR_USER_PROMPT


class InterceptorAgent:
    """
    Request Interceptor Agent

    Responsibilities:
    1. Parse user request to identify technology
    2. Classify technology into category
    3. Extract any contextual hints about requirements
    4. Route to Why Validator agent
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the interceptor agent.

        Args:
            llm: LangChain ChatOpenAI instance (only used for ambiguous requests)
        """
        self.llm = llm
        self.name = "interceptor"

    def process(self, state: AgentState) -> AgentState:
        """
        Process the incoming request and classify the technology.

        This method:
        1. Attempts rule-based classification first (no API cost)
        2. Falls back to LLM only if classification is uncertain
        3. Updates state with technology info and routes to next agent

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        user_request = state["user_request"]

        # Try rule-based classification first (FREE - no API call)
        technology, category, confidence = classify_technology(user_request)

        # Extract contextual hints regardless of classification method
        context_hints = extract_context_hints(user_request)

        # If confidence is low, use LLM for better classification
        if confidence < 0.5 and self.llm:
            llm_result = self._classify_with_llm(user_request)
            if llm_result:
                technology = llm_result.technology_name
                category = llm_result.category
                confidence = llm_result.confidence
                state["total_llm_calls"] = state.get("total_llm_calls", 0) + 1

        # Update state
        state["technology_requested"] = technology
        state["technology_category"] = category
        state["request_context"] = ", ".join(context_hints) if context_hints else None

        # Add to conversation history
        now = datetime.now().isoformat()
        if technology:
            response_msg = f"I see you're interested in using {technology}. Let me ask a few questions to ensure this is the best fit for your needs."
        else:
            response_msg = "I'd like to understand more about what technology you're considering. Could you be more specific about what you'd like to use?"

        state["conversation_history"].append(
            ConversationTurn(
                role="agent",
                content=response_msg,
                timestamp=now,
                agent_name=self.name
            )
        )

        # Set routing
        state["current_agent"] = self.name
        state["next_agent"] = "why_validator" if technology else "clarification"
        state["updated_at"] = now

        return state

    def _classify_with_llm(self, user_request: str) -> Optional[TechnologyClassification]:
        """
        Use LLM to classify ambiguous requests.

        Only called when rule-based classification has low confidence.
        """
        if not self.llm:
            return None

        # Create structured output chain
        structured_llm = self.llm.with_structured_output(TechnologyClassification)

        prompt = ChatPromptTemplate.from_messages([
            ("system", INTERCEPTOR_SYSTEM_PROMPT),
            ("human", INTERCEPTOR_USER_PROMPT),
        ])

        chain = prompt | structured_llm

        try:
            result = chain.invoke({"user_request": user_request})
            return result
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return None

    def get_agent_summary(self, state: AgentState) -> Dict[str, Any]:
        """Get a summary of what this agent determined."""
        return {
            "agent": self.name,
            "technology_identified": state.get("technology_requested"),
            "category": state.get("technology_category"),
            "context_hints": state.get("request_context"),
            "next_step": state.get("next_agent"),
        }


def create_interceptor_node(llm: Optional[ChatOpenAI] = None):
    """
    Factory function to create interceptor node for LangGraph.

    Usage in LangGraph:
        workflow.add_node("intercept", create_interceptor_node(llm))
    """
    agent = InterceptorAgent(llm)

    def node_function(state: AgentState) -> AgentState:
        return agent.process(state)

    return node_function
