#!/usr/bin/env python3
"""
Agentic Architecture Governance System - CLI Entry Point

A multi-agent system that prevents solution-jumping by enforcing requirements
validation through INCOSE's Five Whys methodology.

Usage:
    python main.py                    # Interactive mode
    python main.py --demo             # Run demonstration scenarios
    python main.py --request "..."    # Single request mode

Target Publication: IEEE Software Special Issue - Engineering Agentic Systems
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import rich for better output, fall back to basic print
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_header():
    """Print the application header."""
    header = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║          AGENTIC ARCHITECTURE GOVERNANCE SYSTEM                               ║
║          Implementing INCOSE Five Whys Methodology                            ║
║                                                                               ║
║          Preventing Solution-Jumping in LLM-Based System Design               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console.print(header, style="bold blue")
    else:
        print(header)


def print_message(role: str, content: str, agent_name: str = None):
    """Print a message with appropriate formatting."""
    if role == "user":
        prefix = "👤 YOU"
        style = "green"
    elif role == "agent":
        prefix = f"🤖 AGENT" + (f" ({agent_name})" if agent_name else "")
        style = "cyan"
    else:
        prefix = "ℹ️  SYSTEM"
        style = "yellow"

    if RICH_AVAILABLE:
        console.print(f"\n[bold {style}]{prefix}[/bold {style}]")
        console.print(content)
    else:
        print(f"\n{prefix}")
        print(content)


def save_transcript(state: dict, output_dir: str = "outputs"):
    """Save the conversation transcript to a file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Prepare transcript
    transcript = {
        "session_id": state.get("session_id"),
        "timestamp": timestamp,
        "original_request": state.get("user_request"),
        "technology_requested": state.get("technology_requested"),
        "technology_category": state.get("technology_category"),
        "requirements_extracted": state.get("extracted_requirements", []),
        "match_score": state.get("match_score"),
        "match_status": state.get("match_status"),
        "recommended_technology": state.get("recommended_technology"),
        "recommendation_rationale": state.get("recommendation_rationale"),
        "conversation_history": state.get("conversation_history", []),
        "total_llm_calls": state.get("total_llm_calls", 0),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, default=str)

    return filepath


def generate_enhanced_outputs(state: dict, output_dir: str = "outputs"):
    """Generate all enhanced outputs: ADR, traceability graph, cost-benefit, etc."""
    from src.features.traceability import DecisionTraceabilityGraph, generate_traceability_report
    from src.features.adr_generator import generate_adr
    from src.features.thinking_layer import AgentThinkingLayer
    from src.features.cost_benefit import CostBenefitAnalyzer
    from src.features.cogent_tradeoffs import SubsystemTradeoffAnalyzer
    from src.tools.knowledge_base import get_technology_profile

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {}

    # 1. Decision Traceability Graph
    try:
        graph = DecisionTraceabilityGraph().build_from_state(state)
        mermaid_path = os.path.join(output_dir, f"traceability_{timestamp}.md")
        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write("# Decision Traceability Graph\n\n")
            f.write("```mermaid\n")
            f.write(graph.to_mermaid())
            f.write("\n```\n\n")
            f.write(generate_traceability_report(state))
        outputs["traceability"] = mermaid_path
    except Exception as e:
        print(f"Warning: Could not generate traceability graph: {e}")

    # 2. Architecture Decision Record
    try:
        adr_content = generate_adr(state)
        adr_path = os.path.join(output_dir, f"ADR_{timestamp}.md")
        with open(adr_path, "w", encoding="utf-8") as f:
            f.write(adr_content)
        outputs["adr"] = adr_path
    except Exception as e:
        print(f"Warning: Could not generate ADR: {e}")

    # 3. Agent Thinking Trace
    try:
        thinking = AgentThinkingLayer().build_from_state(state)
        thinking_path = os.path.join(output_dir, f"thinking_{timestamp}.txt")
        with open(thinking_path, "w", encoding="utf-8") as f:
            f.write(thinking.get_thinking_trace())
            f.write("\n\n")
            f.write(thinking.get_confidence_chart())
            tech = state.get("technology_requested", "")
            if tech:
                f.write("\n\n")
                f.write(thinking.get_domain_knowledge_applied(tech))
        outputs["thinking"] = thinking_path
    except Exception as e:
        print(f"Warning: Could not generate thinking trace: {e}")

    # 4. Cost-Benefit Analysis
    try:
        analyzer = CostBenefitAnalyzer()
        result = analyzer.analyze(state)
        cost_path = os.path.join(output_dir, f"cost_benefit_{timestamp}.txt")
        with open(cost_path, "w", encoding="utf-8") as f:
            f.write(analyzer.format_report(result))
        outputs["cost_benefit"] = cost_path
    except Exception as e:
        print(f"Warning: Could not generate cost-benefit analysis: {e}")

    # 5. Subsystem Trade-off Analysis
    try:
        tech = state.get("technology_requested", "")
        profile = get_technology_profile(tech) if tech else {}
        requirements = state.get("extracted_requirements", [])

        if profile:
            tradeoff_analyzer = SubsystemTradeoffAnalyzer()
            subsystem_analysis = tradeoff_analyzer.analyze_technology_for_subsystems(
                tech, profile, requirements
            )
            conflicts = tradeoff_analyzer.identify_conflicts()

            tradeoff_path = os.path.join(output_dir, f"tradeoffs_{timestamp}.txt")
            with open(tradeoff_path, "w", encoding="utf-8") as f:
                f.write(tradeoff_analyzer.format_tradeoff_report(tech, subsystem_analysis, conflicts))
            outputs["tradeoffs"] = tradeoff_path
    except Exception as e:
        print(f"Warning: Could not generate trade-off analysis: {e}")

    return outputs


def run_interactive_session():
    """Run an interactive governance session."""
    from src.agent import create_governance_system

    print_header()

    if RICH_AVAILABLE:
        console.print("\n[bold]Welcome to the Architecture Governance System![/bold]")
        console.print("I'll help ensure your technology choices match your actual requirements.")
        console.print("Type 'quit' or 'exit' to end the session.\n")
    else:
        print("\nWelcome to the Architecture Governance System!")
        print("I'll help ensure your technology choices match your actual requirements.")
        print("Type 'quit' or 'exit' to end the session.\n")

    # Create the governance system
    try:
        system = create_governance_system()
    except Exception as e:
        print(f"\nError initializing system: {e}")
        print("Make sure you have set OPENAI_API_KEY in your .env file")
        return

    # Get initial request
    if RICH_AVAILABLE:
        user_request = Prompt.ask("\n[bold green]What technology would you like to use?[/bold green]")
    else:
        user_request = input("\nWhat technology would you like to use? ")

    if user_request.lower() in ["quit", "exit"]:
        print("\nGoodbye!")
        return

    # Start the session
    print_message("user", user_request)

    try:
        state = system.start_session(user_request)
    except Exception as e:
        print(f"\nError starting session: {e}")
        return

    # Track messages already shown to avoid duplicates
    shown_message_count = 0

    # Main conversation loop
    while not state.get("workflow_complete", False):
        # Show only NEW agent messages since last iteration
        history = state.get("conversation_history", [])
        for turn in history[shown_message_count:]:
            if turn["role"] == "agent":
                print_message("agent", turn["content"], turn.get("agent_name"))
        shown_message_count = len(history)

        # Check if awaiting user response
        if state.get("awaiting_user_response", False):
            if RICH_AVAILABLE:
                user_response = Prompt.ask("\n[bold green]Your response[/bold green]")
            else:
                user_response = input("\nYour response: ")

            if user_response.lower() in ["quit", "exit"]:
                print("\nSession ended by user.")
                break

            print_message("user", user_response)

            try:
                state = system.continue_session(state, user_response)
            except Exception as e:
                print(f"\nError continuing session: {e}")
                break
        else:
            # Workflow continues automatically
            break

    # Show final summary if workflow completed (only new messages)
    if state.get("workflow_complete", False):
        history = state.get("conversation_history", [])
        for turn in history[shown_message_count:]:
            if turn["role"] == "agent":
                print_message("agent", turn["content"], turn.get("agent_name"))

    # Save transcript
    transcript_path = save_transcript(state)
    if RICH_AVAILABLE:
        console.print(f"\n[dim]Transcript saved to: {transcript_path}[/dim]")
    else:
        print(f"\nTranscript saved to: {transcript_path}")

    # Generate enhanced outputs
    if state.get("workflow_complete", False):
        try:
            outputs = generate_enhanced_outputs(state)
            if outputs:
                if RICH_AVAILABLE:
                    console.print("\n[bold]Enhanced outputs generated:[/bold]")
                    for output_type, path in outputs.items():
                        console.print(f"  [dim]• {output_type}: {path}[/dim]")
                else:
                    print("\nEnhanced outputs generated:")
                    for output_type, path in outputs.items():
                        print(f"  • {output_type}: {path}")
        except Exception as e:
            print(f"\nNote: Could not generate enhanced outputs: {e}")

    # Ask for final decision
    if RICH_AVAILABLE:
        decision = Prompt.ask("\n[bold]Enter your decision (1/2/3 or press Enter to end)[/bold]")
    else:
        decision = input("\nEnter your decision (1/2/3 or press Enter to end): ")

    print("\nThank you for using the Architecture Governance System!")


def run_demo_scenarios():
    """Run demonstration scenarios."""
    from src.agent import create_governance_system

    print_header()

    if RICH_AVAILABLE:
        console.print("\n[bold yellow]Running Demonstration Scenarios[/bold yellow]\n")
    else:
        print("\nRunning Demonstration Scenarios\n")

    # Demo scenarios
    scenarios = [
        {
            "name": "MongoDB for Tabular Sales Data",
            "request": "I want to use MongoDB for storing our tabular sales data with complex reporting needs",
            "responses": [
                "We have sales records with customer info, product details, and transaction amounts. It's all structured in rows and columns.",
                "We need to generate monthly reports with joins across customers, products, and transactions. Accuracy is critical for finance.",
                "I thought MongoDB was more modern and scalable, but I see your point about joins and transactions.",
            ],
        },
        {
            "name": "CNN for Customer Churn Prediction",
            "request": "I need a CNN to predict customer churn from our CRM data",
            "responses": [
                "We have customer data in a spreadsheet - age, tenure, monthly charges, contract type, about 50 columns total.",
                "We need to explain to the business team why customers are churning, so interpretability is important.",
                "I see - the data is tabular and we need explainability. What would you recommend instead?",
            ],
        },
        {
            "name": "Microservices for MVP",
            "request": "We should build our new MVP with microservices architecture",
            "responses": [
                "It's just me and one other developer. We want to launch in 3 months.",
                "We expect maybe 100 users initially. It's a simple CRUD app for tracking inventory.",
                "That makes sense. We can always split into services later if we need to scale.",
            ],
        },
        {
            "name": "Kubernetes for Small Deployment",
            "request": "I want to deploy our application on Kubernetes",
            "responses": [
                "We have one Node.js backend service and a React frontend. Maybe 2-3 containers total.",
                "Our team is 3 people, none with K8s experience. We need to ship in 2 weeks.",
                "You're right, Docker Compose would be much simpler for our needs.",
            ],
        },
        {
            "name": "Grafana for Embedded Analytics (with HDF5)",
            "request": "I want to use Grafana for our data visualization dashboard",
            "responses": [
                "Our data pipeline outputs compressed HDF5 files with scientific data. We need custom visualizations.",
                "The dashboard needs to be embedded in our SaaS product for end-users, white-labeled with our branding.",
                "I understand - we need a custom solution that can handle HDF5 and embed properly.",
            ],
        },
        {
            "name": "Graph Database for Relationship-Heavy Data",
            "request": "I want to use MongoDB for our social network data with user connections",
            "responses": [
                "We're building a professional network - users connect with each other, endorse skills, share content. Relationships are key.",
                "We need to find paths between users, recommend connections, and analyze network clusters. The edges matter as much as the nodes.",
                "The relationships and graph traversal are the core of our product. I see why a graph database makes more sense.",
            ],
        },
    ]

    system = create_governance_system()

    for i, scenario in enumerate(scenarios, 1):
        if RICH_AVAILABLE:
            console.print(f"\n[bold]{'='*60}[/bold]")
            console.print(f"[bold cyan]Scenario {i}: {scenario['name']}[/bold cyan]")
            console.print(f"[bold]{'='*60}[/bold]")
        else:
            print(f"\n{'='*60}")
            print(f"Scenario {i}: {scenario['name']}")
            print(f"{'='*60}")

        # Run the scenario
        state = system.start_session(scenario["request"])

        print_message("user", scenario["request"])

        # Track messages already shown to avoid duplicates
        shown_message_count = 0

        response_idx = 0
        while not state.get("workflow_complete", False) and response_idx < len(scenario["responses"]):
            # Show only NEW agent messages since last iteration
            history = state.get("conversation_history", [])
            for turn in history[shown_message_count:]:
                if turn["role"] == "agent":
                    print_message("agent", turn["content"], turn.get("agent_name"))
            shown_message_count = len(history)

            if state.get("awaiting_user_response", False):
                user_response = scenario["responses"][response_idx]
                print_message("user", user_response)
                state = system.continue_session(state, user_response)
                response_idx += 1
            else:
                break

        # Show final result (only new messages)
        if state.get("workflow_complete", False):
            history = state.get("conversation_history", [])
            for turn in history[shown_message_count:]:
                if turn["role"] == "agent":
                    print_message("agent", turn["content"], turn.get("agent_name"))

        # Save transcript and generate enhanced outputs
        save_transcript(state)
        if state.get("workflow_complete", False):
            try:
                outputs = generate_enhanced_outputs(state)
                if outputs and RICH_AVAILABLE:
                    console.print("\n[dim]Enhanced outputs saved to outputs/ folder[/dim]")
            except Exception:
                pass

        if RICH_AVAILABLE:
            console.print("\n[dim]Press Enter to continue to next scenario...[/dim]")
        input()

    print("\nDemonstration complete!")
    print("Check the outputs/ folder for ADRs, traceability graphs, and analysis reports.")


def run_single_request(request: str):
    """Run a single request through the system."""
    from src.agent import create_governance_system

    print_header()

    system = create_governance_system()
    state = system.start_session(request)

    print_message("user", request)

    # For single request mode, just show the initial analysis
    history = state.get("conversation_history", [])
    for turn in history:
        if turn["role"] == "agent":
            print_message("agent", turn["content"], turn.get("agent_name"))

    if state.get("awaiting_user_response", False):
        print("\n[Session requires interactive input. Run without --request for interactive mode.]")

    save_transcript(state)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Architecture Governance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive mode
  python main.py --demo                       # Run demo scenarios
  python main.py --request "Use MongoDB"      # Single request

Environment Variables:
  OPENAI_API_KEY    - Your OpenAI API key
  OPENAI_API_BASE   - API base URL (for Vocareum or custom endpoints)
  MODEL_NAME        - Model to use (default: gpt-4o)
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration scenarios"
    )
    parser.add_argument(
        "--request",
        type=str,
        help="Single technology request to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output transcripts"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it:")
        print("  export OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    if args.demo:
        run_demo_scenarios()
    elif args.request:
        run_single_request(args.request)
    else:
        run_interactive_session()


if __name__ == "__main__":
    main()
