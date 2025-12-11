"""
Technology Knowledge Base

Loads and queries technology profiles for requirement matching.
This is the "brain" that knows what technologies are good/bad for.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from functools import lru_cache


# Path to technology profiles
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
PROFILES_PATH = os.path.join(CONFIG_DIR, "technology_profiles.yaml")


@lru_cache(maxsize=1)
def load_technology_profiles() -> Dict[str, Any]:
    """
    Load technology profiles from YAML file.
    Cached to avoid repeated file reads.

    Returns empty dict with default structure if file is missing or invalid.
    """
    try:
        if not os.path.exists(PROFILES_PATH):
            print(f"Warning: Technology profiles not found at {PROFILES_PATH}")
            return _get_fallback_profiles()

        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            profiles = yaml.safe_load(f)

        if not profiles or not isinstance(profiles, dict):
            print("Warning: Technology profiles file is empty or invalid")
            return _get_fallback_profiles()

        return profiles

    except yaml.YAMLError as e:
        print(f"Warning: Error parsing technology profiles YAML: {e}")
        return _get_fallback_profiles()
    except Exception as e:
        print(f"Warning: Error loading technology profiles: {e}")
        return _get_fallback_profiles()


def _get_fallback_profiles() -> Dict[str, Any]:
    """
    Return minimal fallback profiles when YAML file is unavailable.
    Allows system to function with basic recommendations.
    """
    return {
        "databases": {
            "postgresql": {
                "display_name": "PostgreSQL",
                "best_for": ["tabular_data", "acid_transactions", "complex_joins", "relational_data"],
                "not_ideal_for": ["document_storage", "horizontal_scaling"],
                "typical_alternatives": ["MySQL", "SQLite"],
                "complexity": "medium",
            },
            "mongodb": {
                "display_name": "MongoDB",
                "best_for": ["document_storage", "flexible_schema", "horizontal_scaling"],
                "not_ideal_for": ["complex_joins", "acid_transactions", "tabular_data"],
                "typical_alternatives": ["PostgreSQL", "CouchDB"],
                "complexity": "medium",
            },
        },
        "ml_models": {
            "xgboost": {
                "display_name": "XGBoost",
                "best_for": ["tabular_data", "interpretability", "small_to_medium_datasets"],
                "not_ideal_for": ["image_data", "unstructured_data"],
                "typical_alternatives": ["Random Forest", "Linear Regression"],
                "complexity": "low",
            },
            "cnn": {
                "display_name": "CNN (Convolutional Neural Network)",
                "best_for": ["image_data", "spatial_patterns", "large_datasets"],
                "not_ideal_for": ["tabular_data", "interpretability", "small_datasets"],
                "typical_alternatives": ["XGBoost", "Random Forest"],
                "complexity": "high",
            },
        },
        "architectures": {
            "monolith": {
                "display_name": "Monolith",
                "best_for": ["small_team", "mvp_stage", "simple_deployment", "tight_timeline"],
                "not_ideal_for": ["large_teams", "independent_scaling"],
                "typical_alternatives": ["Modular Monolith", "Microservices"],
                "complexity": "low",
            },
            "microservices": {
                "display_name": "Microservices",
                "best_for": ["large_teams", "independent_scaling", "polyglot_persistence"],
                "not_ideal_for": ["small_team", "mvp_stage", "tight_timeline"],
                "typical_alternatives": ["Monolith", "Modular Monolith"],
                "complexity": "high",
            },
        },
        "devops": {
            "docker_compose": {
                "display_name": "Docker Compose",
                "best_for": ["small_deployment", "simple_orchestration", "development"],
                "not_ideal_for": ["large_scale", "auto_scaling"],
                "typical_alternatives": ["Kubernetes"],
                "complexity": "low",
            },
            "kubernetes": {
                "display_name": "Kubernetes",
                "best_for": ["large_scale", "auto_scaling", "multi_service"],
                "not_ideal_for": ["small_deployment", "small_team", "tight_timeline"],
                "typical_alternatives": ["Docker Compose", "Heroku"],
                "complexity": "high",
            },
        },
        "visualization": {},
        "data_pipeline": {},
    }


def get_all_technologies() -> List[str]:
    """Get list of all known technology names."""
    profiles = load_technology_profiles()
    technologies = []

    for category_key in ["databases", "ml_models", "architectures", "devops", "visualization", "data_pipeline"]:
        if category_key in profiles:
            technologies.extend(profiles[category_key].keys())

    return technologies


def get_technology_profile(technology: str) -> Optional[Dict[str, Any]]:
    """
    Get the profile for a specific technology.

    Returns None if technology is not in knowledge base.
    """
    profiles = load_technology_profiles()
    tech_lower = technology.lower().replace(" ", "_").replace("-", "_")

    # Map display names to keys
    name_to_key = {
        "mongodb": "mongodb",
        "postgresql": "postgresql",
        "postgres": "postgresql",
        "mysql": "mysql",
        "redis": "redis",
        "sqlite": "sqlite",
        "cnn": "cnn",
        "convolutional neural network": "cnn",
        "linear regression": "linear_regression",
        "linear_regression": "linear_regression",
        "xgboost": "xgboost",
        "gradient boosting": "xgboost",
        "random forest": "random_forest",
        "random_forest": "random_forest",
        "transformer": "transformer",
        "lstm": "lstm",
        "microservices": "microservices",
        "modular monolith": "modular_monolith",
        "modular_monolith": "modular_monolith",
        "monolith": "monolith",
        "serverless": "serverless",
        "kubernetes": "kubernetes",
        "k8s": "kubernetes",
        "docker compose": "docker_compose",
        "docker_compose": "docker_compose",
        "docker-compose": "docker_compose",
        "heroku": "heroku",
        "ecs": "ecs",
        "aws ecs": "ecs",
        "elastic container service": "ecs",
        "grafana": "grafana",
        "metabase": "metabase",
        "superset": "superset",
        "apache superset": "superset",
        "custom dashboard": "custom_dashboard",
        "custom_dashboard": "custom_dashboard",
        "apache kafka": "kafka",
        "kafka": "kafka",
        "batch processing": "batch_processing",
        "batch_processing": "batch_processing",
        "rabbitmq": "rabbitmq",
        "neo4j": "neo4j",
        "graph database": "neo4j",
        "neptune": "neptune",
        "arangodb": "arangodb",
        "dgraph": "dgraph",
        "graph neural network": "gnn",
        "gnn": "gnn",
        "node2vec": "node2vec",
        "graphsage": "graphsage",
    }

    key = name_to_key.get(tech_lower, tech_lower)

    # Search in each category
    category_map = {
        "databases": "database",
        "ml_models": "ml_model",
        "architectures": "architecture",
        "devops": "devops",
        "visualization": "visualization",
        "data_pipeline": "data_pipeline",
    }

    for category_key, category_name in category_map.items():
        if category_key in profiles and key in profiles[category_key]:
            profile = profiles[category_key][key].copy()
            profile["_key"] = key
            profile["_category"] = category_name
            return profile

    return None


def get_alternatives_for_category(category: str) -> List[str]:
    """Get all technologies in a given category."""
    profiles = load_technology_profiles()

    category_map = {
        "database": "databases",
        "ml_model": "ml_models",
        "architecture": "architectures",
        "devops": "devops",
        "visualization": "visualization",
        "data_pipeline": "data_pipeline",
    }

    category_key = category_map.get(category)
    if category_key and category_key in profiles:
        return list(profiles[category_key].keys())

    return []


def get_best_alternative(
    requirements: List[str],
    category: str,
    exclude: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Find the best alternative technology for given requirements.

    Args:
        requirements: List of requirement strings (e.g., ["tabular_data", "acid_transactions"])
        category: Technology category to search in
        exclude: Technology to exclude (usually the originally requested one)

    Returns:
        Dict with best alternative info or None
    """
    profiles = load_technology_profiles()

    category_map = {
        "database": "databases",
        "ml_model": "ml_models",
        "architecture": "architectures",
        "devops": "devops",
        "visualization": "visualization",
        "data_pipeline": "data_pipeline",
    }

    category_key = category_map.get(category)
    if not category_key or category_key not in profiles:
        return None

    best_match = None
    best_score = -1

    for tech_key, tech_profile in profiles[category_key].items():
        if exclude and tech_key.lower() == exclude.lower().replace(" ", "_"):
            continue

        # Calculate match score
        best_for = tech_profile.get("best_for", [])
        not_ideal_for = tech_profile.get("not_ideal_for", [])

        positive_matches = sum(1 for req in requirements if req in best_for)
        negative_matches = sum(1 for req in requirements if req in not_ideal_for)

        score = positive_matches - (negative_matches * 2)  # Penalize negatives more

        if score > best_score:
            best_score = score
            best_match = {
                "technology": tech_profile.get("display_name", tech_key),
                "key": tech_key,
                "score": score,
                "positive_matches": positive_matches,
                "negative_matches": negative_matches,
                "best_for": best_for,
                "not_ideal_for": not_ideal_for,
                "complexity": tech_profile.get("complexity", "unknown"),
            }

    return best_match


def get_technology_comparison(tech1: str, tech2: str) -> Dict[str, Any]:
    """Compare two technologies side by side."""
    profile1 = get_technology_profile(tech1)
    profile2 = get_technology_profile(tech2)

    if not profile1 or not profile2:
        return {"error": "One or both technologies not found in knowledge base"}

    return {
        "technology_1": {
            "name": profile1.get("display_name", tech1),
            "best_for": profile1.get("best_for", []),
            "not_ideal_for": profile1.get("not_ideal_for", []),
            "complexity": profile1.get("complexity", "unknown"),
        },
        "technology_2": {
            "name": profile2.get("display_name", tech2),
            "best_for": profile2.get("best_for", []),
            "not_ideal_for": profile2.get("not_ideal_for", []),
            "complexity": profile2.get("complexity", "unknown"),
        },
        "shared_strengths": list(
            set(profile1.get("best_for", [])) & set(profile2.get("best_for", []))
        ),
        "unique_to_1": list(
            set(profile1.get("best_for", [])) - set(profile2.get("best_for", []))
        ),
        "unique_to_2": list(
            set(profile2.get("best_for", [])) - set(profile1.get("best_for", []))
        ),
    }


def search_technologies_by_requirement(requirement: str) -> List[Dict[str, Any]]:
    """Find technologies that are good for a specific requirement."""
    profiles = load_technology_profiles()
    matches = []

    for category_key in ["databases", "ml_models", "architectures", "devops", "visualization", "data_pipeline"]:
        if category_key not in profiles:
            continue

        for tech_key, tech_profile in profiles[category_key].items():
            best_for = tech_profile.get("best_for", [])
            if requirement.lower() in [bf.lower() for bf in best_for]:
                matches.append({
                    "technology": tech_profile.get("display_name", tech_key),
                    "category": category_key.rstrip("s"),  # Remove trailing 's'
                    "all_strengths": best_for,
                    "complexity": tech_profile.get("complexity", "unknown"),
                })

    return matches
