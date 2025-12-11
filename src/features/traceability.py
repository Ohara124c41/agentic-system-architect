"""
Decision Traceability Graph

Generates a visual trace of the decision path:
Requirements → Evaluation → Recommendation

Provides full audit trail for governance decisions.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DecisionNode:
    """A node in the decision traceability graph."""
    node_id: str
    node_type: str  # requirement, technology, evaluation, recommendation
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DecisionEdge:
    """An edge connecting nodes in the decision graph."""
    source_id: str
    target_id: str
    edge_type: str  # extracted_from, evaluated_against, leads_to, conflicts_with
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionTraceabilityGraph:
    """
    Builds and exports a decision traceability graph.

    This graph captures:
    - User requirements as nodes
    - Technology options as nodes
    - Evaluation relationships as edges
    - Final recommendation path

    Can be exported for visualization (Mermaid, GraphViz, JSON).
    """

    def __init__(self):
        self.nodes: Dict[str, DecisionNode] = {}
        self.edges: List[DecisionEdge] = []
        self.session_id: Optional[str] = None

    def build_from_state(self, state: Dict[str, Any]) -> "DecisionTraceabilityGraph":
        """Build the graph from a governance session state."""
        self.session_id = state.get("session_id", "unknown")

        # Add user request node
        self._add_node(
            node_id="user_request",
            node_type="request",
            label=f"Request: {state.get('technology_requested', 'Unknown')}",
            metadata={"original_request": state.get("user_request", "")}
        )

        # Add requirement nodes from Five Whys
        requirements = state.get("extracted_requirements", [])
        for i, req in enumerate(requirements):
            req_id = f"req_{i}"
            self._add_node(
                node_id=req_id,
                node_type="requirement",
                label=req.replace("_", " ").title(),
                metadata={"extracted_from": "five_whys", "iteration": i}
            )
            # Link requirement to user request
            self._add_edge("user_request", req_id, "extracted_from")

        # Add technology evaluation node
        tech = state.get("technology_requested", "Unknown")
        match_score = state.get("match_score", 0)
        match_status = state.get("match_status", "unknown")

        self._add_node(
            node_id="original_tech",
            node_type="technology",
            label=f"{tech} ({match_score:.0%} match)",
            metadata={
                "match_score": match_score,
                "match_status": match_status,
                "mismatches": state.get("mismatches", [])
            }
        )

        # Link requirements to technology evaluation
        for i, req in enumerate(requirements):
            req_id = f"req_{i}"
            # Determine if this requirement matches or conflicts
            if req in state.get("mismatches", []):
                self._add_edge(req_id, "original_tech", "conflicts_with", weight=0.0)
            else:
                self._add_edge(req_id, "original_tech", "evaluated_against", weight=1.0)

        # Add recommendation if different from original
        recommended = state.get("recommended_technology")
        if recommended and recommended != tech:
            self._add_node(
                node_id="recommended_tech",
                node_type="recommendation",
                label=f"Recommended: {recommended}",
                metadata={
                    "rationale": state.get("recommendation_rationale", ""),
                    "alternatives": state.get("alternatives", [])
                }
            )
            self._add_edge("original_tech", "recommended_tech", "leads_to")

            # Link requirements that are better served
            for i, req in enumerate(requirements):
                req_id = f"req_{i}"
                if req in state.get("mismatches", []):
                    self._add_edge(req_id, "recommended_tech", "better_served_by")

        # Add decision node
        self._add_node(
            node_id="decision",
            node_type="decision",
            label="Awaiting Decision",
            metadata={"workflow_complete": state.get("workflow_complete", False)}
        )

        if recommended and recommended != tech:
            self._add_edge("recommended_tech", "decision", "leads_to")
        else:
            self._add_edge("original_tech", "decision", "leads_to")

        return self

    def _add_node(self, node_id: str, node_type: str, label: str,
                  metadata: Optional[Dict] = None):
        """Add a node to the graph."""
        self.nodes[node_id] = DecisionNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            metadata=metadata or {}
        )

    def _add_edge(self, source_id: str, target_id: str, edge_type: str,
                  weight: float = 1.0, metadata: Optional[Dict] = None):
        """Add an edge to the graph."""
        self.edges.append(DecisionEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {}
        ))

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram syntax."""
        lines = ["graph TD"]

        # Define node styles
        style_map = {
            "request": ":::request",
            "requirement": ":::requirement",
            "technology": ":::technology",
            "recommendation": ":::recommendation",
            "decision": ":::decision"
        }

        # Add nodes
        for node_id, node in self.nodes.items():
            safe_label = node.label.replace('"', "'")
            style = style_map.get(node.node_type, "")

            if node.node_type == "request":
                lines.append(f'    {node_id}["{safe_label}"]{style}')
            elif node.node_type == "requirement":
                lines.append(f'    {node_id}(("{safe_label}"))')
            elif node.node_type == "technology":
                lines.append(f'    {node_id}[["{safe_label}"]]')
            elif node.node_type == "recommendation":
                lines.append(f'    {node_id}{{"{safe_label}"}}')
            else:
                lines.append(f'    {node_id}["{safe_label}"]')

        # Add edges
        edge_arrows = {
            "extracted_from": "-->|extracted|",
            "evaluated_against": "-->|matches|",
            "conflicts_with": "-.->|conflicts|",
            "leads_to": "==>",
            "better_served_by": "-->|better fit|"
        }

        for edge in self.edges:
            arrow = edge_arrows.get(edge.edge_type, "-->")
            lines.append(f'    {edge.source_id} {arrow} {edge.target_id}')

        # Add styles
        lines.extend([
            "",
            "    classDef request fill:#e1f5fe,stroke:#01579b",
            "    classDef requirement fill:#fff3e0,stroke:#e65100",
            "    classDef technology fill:#fce4ec,stroke:#880e4f",
            "    classDef recommendation fill:#e8f5e9,stroke:#1b5e20",
            "    classDef decision fill:#f3e5f5,stroke:#4a148c",
        ])

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export graph as JSON for external tools."""
        return json.dumps({
            "session_id": self.session_id,
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type,
                    "label": n.label,
                    "metadata": n.metadata,
                    "timestamp": n.timestamp
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type,
                    "weight": e.weight,
                    "metadata": e.metadata
                }
                for e in self.edges
            ]
        }, indent=2)

    def to_graphviz(self) -> str:
        """Export graph as GraphViz DOT format."""
        lines = [
            "digraph DecisionTrace {",
            '    rankdir=TB;',
            '    node [fontname="Arial"];',
            ""
        ]

        # Node shapes by type
        shape_map = {
            "request": "box",
            "requirement": "ellipse",
            "technology": "parallelogram",
            "recommendation": "hexagon",
            "decision": "diamond"
        }

        color_map = {
            "request": "#01579b",
            "requirement": "#e65100",
            "technology": "#880e4f",
            "recommendation": "#1b5e20",
            "decision": "#4a148c"
        }

        # Add nodes
        for node_id, node in self.nodes.items():
            safe_label = node.label.replace('"', '\\"')
            shape = shape_map.get(node.node_type, "box")
            color = color_map.get(node.node_type, "#000000")
            lines.append(
                f'    {node_id} [label="{safe_label}" shape={shape} '
                f'color="{color}" style=filled fillcolor="{color}20"];'
            )

        lines.append("")

        # Add edges
        for edge in self.edges:
            style = "dashed" if edge.edge_type == "conflicts_with" else "solid"
            color = "red" if edge.edge_type == "conflicts_with" else "black"
            lines.append(
                f'    {edge.source_id} -> {edge.target_id} '
                f'[label="{edge.edge_type}" style={style} color="{color}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    def get_decision_path(self) -> List[str]:
        """Get the main decision path as a list of node labels."""
        path = []
        current = "user_request"
        visited = set()

        while current and current not in visited:
            visited.add(current)
            if current in self.nodes:
                path.append(self.nodes[current].label)

            # Find next node via "leads_to" edge
            next_node = None
            for edge in self.edges:
                if edge.source_id == current and edge.edge_type == "leads_to":
                    next_node = edge.target_id
                    break
            current = next_node

        return path

    def get_conflict_summary(self) -> List[Dict[str, str]]:
        """Get summary of all conflicts identified."""
        conflicts = []
        for edge in self.edges:
            if edge.edge_type == "conflicts_with":
                source_node = self.nodes.get(edge.source_id)
                target_node = self.nodes.get(edge.target_id)
                if source_node and target_node:
                    conflicts.append({
                        "requirement": source_node.label,
                        "technology": target_node.label,
                        "description": f"{source_node.label} conflicts with {target_node.label}"
                    })
        return conflicts


def generate_traceability_report(state: Dict[str, Any]) -> str:
    """Generate a full traceability report from session state."""
    graph = DecisionTraceabilityGraph().build_from_state(state)

    report = []
    report.append("=" * 80)
    report.append("DECISION TRACEABILITY REPORT")
    report.append("=" * 80)
    report.append("")

    # Decision path
    report.append("📍 DECISION PATH")
    report.append("-" * 40)
    path = graph.get_decision_path()
    for i, step in enumerate(path, 1):
        report.append(f"  {i}. {step}")
    report.append("")

    # Conflicts
    conflicts = graph.get_conflict_summary()
    if conflicts:
        report.append("⚠️  CONFLICTS IDENTIFIED")
        report.append("-" * 40)
        for conflict in conflicts:
            report.append(f"  • {conflict['description']}")
        report.append("")

    # Mermaid diagram
    report.append("📊 MERMAID DIAGRAM")
    report.append("-" * 40)
    report.append("```mermaid")
    report.append(graph.to_mermaid())
    report.append("```")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)
