"""
Technology-Requirement Matching Tools

Deterministic matching logic to evaluate how well a technology
fits extracted requirements. Minimizes LLM API calls.
"""

from typing import List, Dict, Tuple, Optional, Any
from .knowledge_base import get_technology_profile


# Canonical requirement mapping - normalizes LLM-extracted terms to profile terms
# This bridges the gap between LLM's natural language extraction and exact profile matching
REQUIREMENT_SYNONYMS = {
    # Graph/relationship requirements -> Neo4j terms
    "relationship_management": "relationship_heavy_queries",
    "relationship_data": "relationship_heavy_queries",
    "relationship_analysis": "relationship_heavy_queries",
    "relationships": "relationship_heavy_queries",
    "graph_operations": "graph_data",
    "graph_queries": "graph_data",
    "graph_structure": "graph_data",
    "network_data": "network_analysis",
    "social_network": "social_networks",
    "social_networks": "social_networks",
    "path_queries": "path_finding",
    "pathfinding": "path_finding",
    "shortest_path": "path_finding",
    "graph_traversal": "graph_traversal",
    "traversals": "graph_traversal",
    "connected_data": "connected_data",
    "connections": "connected_data",
    "edges": "edges_as_first_class",
    "edge_data": "edges_as_first_class",
    "nodes_and_edges": "edges_as_first_class",
    "recommendation": "recommendation_engines",
    "recommendations": "recommendation_engines",
    "knowledge_graph": "knowledge_graphs",
    "fraud": "fraud_detection",

    # Database/SQL requirements -> PostgreSQL terms
    "complex_joins": "complex_queries",
    "join_operations": "complex_queries",
    "sql_joins": "complex_queries",
    "report_generation": "reporting_systems",
    "monthly_reports": "reporting_systems",
    "reporting": "reporting_systems",
    "accurate_reporting": "reporting_systems",
    "financial_accuracy": "financial_data",
    "financial_reporting": "financial_data",
    "finance": "financial_data",
    "data_accuracy": "data_integrity",
    "high_accuracy": "data_integrity",
    "accuracy": "data_integrity",
    "transactions": "acid_transactions",
    "transaction_support": "acid_transactions",
    "acid_compliance": "acid_transactions",
    "relational": "relational_data",
    "structured_data": "tabular_data",
    "rows_and_columns": "tabular_data",
    "spreadsheet_data": "tabular_data",

    # ML requirements -> profile terms
    "explainability": "interpretability",
    "model_interpretability": "interpretability",
    "explain_to_business": "interpretability",
    "feature_analysis": "feature_importance",

    # Architecture requirements -> profile terms
    "simple_crud_operations": "simple_crud",
    "crud_operations": "simple_crud",
    "mvp": "mvp_stage",
    "prototype": "mvp_stage",
    "rapid_development": "rapid_prototyping",
    "small_team": "small_teams",
    "solo_developer": "solo_developer",
    "2_developers": "small_team",
    "tight_timeline": "tight_deadlines",
    "launch_quickly": "tight_deadlines",
    "inventory": "inventory_tracking",
    "initial_user_base_100": "small_applications",

    # DevOps requirements -> profile terms
    "container_management": "container_orchestration",
    "service_orchestration": "container_orchestration",
    "quick_deployment": "rapid_deployment",
    "ease_of_use": "managed_infrastructure",
    "minimal_configuration": "managed_infrastructure",
    "few_containers": "small_deployments",
    "2_3_containers": "small_deployments",
    "no_k8s_experience": "limited_ops_experience",
    "no_ops_experience": "limited_ops_experience",
    "learning_containers": "learning_containers",
    # Cross-category terms that apply to DevOps
    "mvp_stage": "mvp_stage",  # Keep for architecture, but also works for Heroku
    "small_teams": "small_teams",  # Keep for architecture, but also works for Heroku
    "tight_deadlines": "tight_deadlines",  # Keep for architecture

    # Visualization requirements -> profile terms
    "embedded_dashboard": "embedded_analytics",
    "product_embedded": "product_integration",
    "white_labeling": "white_label",
    "custom_charts": "custom_visualizations",
    "hdf5": "hdf5_data",
    "hdf5_files": "hdf5_data",
    "support_for_hdf5_files": "hdf5_data",
    "end_user_access": "product_integration",

    # Scale requirements
    "horizontal_scaling": "horizontal_scaling",
    "large_scale": "large_scale_deployments",
    "scalability": "horizontal_scaling",
    "auto_scaling": "auto_scaling",

    # Quality attributes (-ilities) - keep as-is for now, profiles can match these
    "maintainability": "maintainability",
    "extensibility": "extensibility",
    "reliability": "reliability",
    "availability": "availability",
    "testability": "testability",
    "security": "security",
    "performance": "performance",
    "usability": "usability",
    "portability": "portability",
    "interoperability": "interoperability",
}


def normalize_requirements(requirements: List[str]) -> List[str]:
    """
    Normalize LLM-extracted requirements to canonical profile terms.

    This is critical for accurate matching since LLMs generate varied
    terminology while profiles use specific terms.
    """
    normalized = []
    for req in requirements:
        req_key = req.lower().replace(" ", "_").replace("-", "_")
        # Check if there's a synonym mapping
        if req_key in REQUIREMENT_SYNONYMS:
            normalized.append(REQUIREMENT_SYNONYMS[req_key])
        else:
            # Keep original (already normalized to lowercase with underscores)
            normalized.append(req_key)
    return list(set(normalized))  # Remove duplicates


def evaluate_technology_match(
    technology: str,
    requirements: List[str],
    constraints: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate how well a technology matches given requirements.

    This is deterministic matching based on the technology profiles.
    No LLM calls needed.

    Args:
        technology: Name of the technology to evaluate
        requirements: List of extracted requirements
        constraints: Optional list of constraints (team size, timeline, etc.)

    Returns:
        Dict with match analysis
    """
    profile = get_technology_profile(technology)

    if not profile:
        return {
            "technology": technology,
            "match_score": 0.5,  # Unknown - neutral
            "status": "unknown",
            "matches": [],
            "mismatches": [],
            "risk_factors": ["Technology not in knowledge base - manual evaluation needed"],
            "explanation": f"'{technology}' is not in the knowledge base. Consider manual evaluation.",
        }

    best_for = set(profile.get("best_for", []))
    not_ideal_for = set(profile.get("not_ideal_for", []))

    # Normalize requirements to canonical terms before matching
    normalized_reqs = normalize_requirements(requirements)
    requirements_set = set(normalized_reqs)

    # Find matches and mismatches
    matches = list(requirements_set & best_for)
    mismatches = list(requirements_set & not_ideal_for)

    # Calculate score
    match_score = calculate_match_score(
        total_requirements=len(requirements),
        positive_matches=len(matches),
        negative_matches=len(mismatches),
        technology_profile=profile,
        constraints=constraints
    )

    # Determine status
    if match_score >= 0.7:
        status = "match"
    elif match_score >= 0.4:
        status = "partial"
    else:
        status = "mismatch"

    # Identify risk factors
    risk_factors = []
    if mismatches:
        risk_factors.append(f"Technology has known issues with: {', '.join(mismatches)}")
    if profile.get("complexity") == "high" and constraints:
        if any("small_team" in c or "solo" in c for c in constraints):
            risk_factors.append("High complexity technology for small team")
    if profile.get("learning_curve") == "steep" and constraints:
        if any("urgent" in c or "deadline" in c for c in constraints):
            risk_factors.append("Steep learning curve may impact timeline")

    # Generate explanation
    explanation = _generate_match_explanation(
        technology=technology,
        profile=profile,
        matches=matches,
        mismatches=mismatches,
        match_score=match_score,
        status=status
    )

    return {
        "technology": technology,
        "display_name": profile.get("display_name", technology),
        "category": profile.get("_category", "unknown"),
        "match_score": match_score,
        "status": status,
        "matches": matches,
        "mismatches": mismatches,
        "risk_factors": risk_factors,
        "explanation": explanation,
        "alternatives": profile.get("typical_alternatives", []),
    }


def calculate_match_score(
    total_requirements: int,
    positive_matches: int,
    negative_matches: int,
    technology_profile: Dict[str, Any],
    constraints: Optional[List[str]] = None
) -> float:
    """
    Calculate a match score from 0.0 to 1.0.

    Scoring logic:
    - Base score from requirement matches
    - Heavy penalty for mismatches
    - Complexity/constraint penalties
    """
    if total_requirements == 0:
        return 0.5  # No requirements - neutral

    # Base score: positive matches as fraction of requirements
    base_score = positive_matches / total_requirements if total_requirements > 0 else 0.5

    # Mismatch penalty: Each mismatch reduces score significantly
    mismatch_penalty = negative_matches * 0.25

    # Constraint penalties
    constraint_penalty = 0.0
    if constraints:
        complexity = technology_profile.get("complexity", "medium")
        learning_curve = technology_profile.get("learning_curve", "moderate")

        # Small team + high complexity
        if complexity == "high" and any("small" in c or "solo" in c for c in constraints):
            constraint_penalty += 0.15

        # Urgent + steep learning curve
        if learning_curve == "steep" and any("urgent" in c or "deadline" in c for c in constraints):
            constraint_penalty += 0.1

    # Calculate final score
    score = base_score - mismatch_penalty - constraint_penalty

    # Clamp to valid range
    return max(0.0, min(1.0, score))


def _generate_match_explanation(
    technology: str,
    profile: Dict[str, Any],
    matches: List[str],
    mismatches: List[str],
    match_score: float,
    status: str
) -> str:
    """Generate human-readable explanation of match evaluation."""
    display_name = profile.get("display_name", technology)

    if status == "match":
        explanation = f"{display_name} appears to be a good fit for your requirements. "
        if matches:
            explanation += f"It aligns well with your needs for: {', '.join(matches)}."
    elif status == "partial":
        explanation = f"{display_name} partially matches your requirements. "
        if matches:
            explanation += f"It handles {', '.join(matches)} well, "
        if mismatches:
            explanation += f"but may struggle with: {', '.join(mismatches)}."
    else:  # mismatch
        explanation = f"{display_name} may not be the best fit for your requirements. "
        if mismatches:
            explanation += f"Your needs for {', '.join(mismatches)} conflict with {display_name}'s strengths. "
        if profile.get("typical_alternatives"):
            alts = profile["typical_alternatives"][:2]
            explanation += f"Consider alternatives like {', '.join(alts)}."

    return explanation


def find_requirement_conflicts(
    technology: str,
    requirements: List[str]
) -> List[Dict[str, str]]:
    """
    Find specific conflicts between a technology and requirements.

    Returns detailed conflict information for each mismatch.
    """
    profile = get_technology_profile(technology)
    if not profile:
        return []

    conflicts = []
    not_ideal_for = profile.get("not_ideal_for", [])

    # Normalize requirements before checking conflicts
    normalized_reqs = normalize_requirements(requirements)

    for req in normalized_reqs:
        if req in not_ideal_for:
            conflicts.append({
                "requirement": req,
                "technology": technology,
                "reason": f"{profile.get('display_name', technology)} is not ideal for {req}",
                "severity": "high" if req in ["acid_transactions", "data_integrity", "complex_queries"] else "medium"
            })

    return conflicts


def score_alternatives(
    requirements: List[str],
    alternatives: List[str],
    original_technology: str
) -> List[Dict[str, Any]]:
    """
    Score multiple alternative technologies against requirements.

    Returns sorted list of alternatives with scores.
    """
    scored = []

    for alt in alternatives:
        if alt.lower() == original_technology.lower():
            continue

        evaluation = evaluate_technology_match(alt, requirements)
        scored.append({
            "technology": alt,
            "display_name": evaluation.get("display_name", alt),
            "match_score": evaluation["match_score"],
            "status": evaluation["status"],
            "matches": evaluation["matches"],
            "mismatches": evaluation["mismatches"],
            "improvement_over_original": None,  # Calculated below
        })

    # Sort by match score descending
    scored.sort(key=lambda x: x["match_score"], reverse=True)

    # Calculate improvement over original
    original_eval = evaluate_technology_match(original_technology, requirements)
    original_score = original_eval["match_score"]

    for alt in scored:
        alt["improvement_over_original"] = alt["match_score"] - original_score

    return scored
