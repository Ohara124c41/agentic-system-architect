"""
Technology Classification Tools

Rule-based classification to minimize LLM API calls.
Falls back to LLM only when rule-based approach is uncertain.
"""

import re
from typing import Tuple, Optional, List

# Technology keywords mapped to canonical names and categories
TECHNOLOGY_PATTERNS = {
    # Databases
    "mongodb": ("MongoDB", "database"),
    "mongo": ("MongoDB", "database"),
    "postgresql": ("PostgreSQL", "database"),
    "postgres": ("PostgreSQL", "database"),
    "mysql": ("MySQL", "database"),
    "redis": ("Redis", "database"),
    "sqlite": ("SQLite", "database"),
    "dynamodb": ("DynamoDB", "database"),
    "cassandra": ("Cassandra", "database"),
    "elasticsearch": ("Elasticsearch", "database"),
    "neo4j": ("Neo4j", "database"),
    "neptune": ("Neptune", "database"),
    "arangodb": ("ArangoDB", "database"),
    "janusgraph": ("JanusGraph", "database"),
    "tigergraph": ("TigerGraph", "database"),
    "graph database": ("Graph Database", "database"),
    "graphdb": ("Graph Database", "database"),
    "nosql": ("NoSQL Database", "database"),
    "relational database": ("Relational Database", "database"),
    "sql database": ("Relational Database", "database"),

    # ML Models
    "cnn": ("CNN", "ml_model"),
    "convolutional neural network": ("CNN", "ml_model"),
    "linear regression": ("Linear Regression", "ml_model"),
    "logistic regression": ("Logistic Regression", "ml_model"),
    "xgboost": ("XGBoost", "ml_model"),
    "gradient boosting": ("XGBoost", "ml_model"),
    "random forest": ("Random Forest", "ml_model"),
    "decision tree": ("Decision Tree", "ml_model"),
    "neural network": ("Neural Network", "ml_model"),
    "deep learning": ("Deep Learning", "ml_model"),
    "lstm": ("LSTM", "ml_model"),
    "rnn": ("RNN", "ml_model"),
    "transformer": ("Transformer", "ml_model"),
    "bert": ("BERT", "ml_model"),
    "gpt": ("GPT", "ml_model"),
    "llm": ("LLM", "ml_model"),
    "svm": ("SVM", "ml_model"),
    "support vector": ("SVM", "ml_model"),
    "k-means": ("K-Means", "ml_model"),
    "clustering": ("Clustering", "ml_model"),
    "gnn": ("Graph Neural Network", "ml_model"),
    "graph neural network": ("Graph Neural Network", "ml_model"),
    "node2vec": ("Node2Vec", "ml_model"),
    "graphsage": ("GraphSAGE", "ml_model"),

    # Architecture Patterns
    "microservices": ("Microservices", "architecture"),
    "microservice": ("Microservices", "architecture"),
    "monolith": ("Monolith", "architecture"),
    "monolithic": ("Monolith", "architecture"),
    "modular monolith": ("Modular Monolith", "architecture"),
    "serverless": ("Serverless", "architecture"),
    "lambda": ("Serverless", "architecture"),
    "event-driven": ("Event-Driven", "architecture"),
    "event driven": ("Event-Driven", "architecture"),
    "cqrs": ("CQRS", "architecture"),
    "event sourcing": ("Event Sourcing", "architecture"),
    "hexagonal": ("Hexagonal Architecture", "architecture"),
    "clean architecture": ("Clean Architecture", "architecture"),
    "plugin architecture": ("Plugin Architecture", "architecture"),

    # DevOps
    "kubernetes": ("Kubernetes", "devops"),
    "k8s": ("Kubernetes", "devops"),
    "docker": ("Docker", "devops"),
    "docker compose": ("Docker Compose", "devops"),
    "docker-compose": ("Docker Compose", "devops"),
    "heroku": ("Heroku", "devops"),
    "railway": ("Railway", "devops"),
    "render": ("Render", "devops"),
    "aws": ("AWS", "devops"),
    "azure": ("Azure", "devops"),
    "gcp": ("GCP", "devops"),
    "terraform": ("Terraform", "devops"),
    "ansible": ("Ansible", "devops"),
    "jenkins": ("Jenkins", "devops"),
    "github actions": ("GitHub Actions", "devops"),
    "gitlab ci": ("GitLab CI", "devops"),
    "ecs": ("ECS", "devops"),
    "fargate": ("Fargate", "devops"),

    # Visualization
    "grafana": ("Grafana", "visualization"),
    "tableau": ("Tableau", "visualization"),
    "power bi": ("Power BI", "visualization"),
    "powerbi": ("Power BI", "visualization"),
    "metabase": ("Metabase", "visualization"),
    "superset": ("Superset", "visualization"),
    "d3": ("D3.js", "visualization"),
    "d3.js": ("D3.js", "visualization"),
    "chart.js": ("Chart.js", "visualization"),
    "chartjs": ("Chart.js", "visualization"),
    "plotly": ("Plotly", "visualization"),
    "matplotlib": ("Matplotlib", "visualization"),

    # Data Pipeline
    "kafka": ("Apache Kafka", "data_pipeline"),
    "apache kafka": ("Apache Kafka", "data_pipeline"),
    "rabbitmq": ("RabbitMQ", "data_pipeline"),
    "rabbit mq": ("RabbitMQ", "data_pipeline"),
    "airflow": ("Apache Airflow", "data_pipeline"),
    "apache airflow": ("Apache Airflow", "data_pipeline"),
    "spark": ("Apache Spark", "data_pipeline"),
    "apache spark": ("Apache Spark", "data_pipeline"),
    "flink": ("Apache Flink", "data_pipeline"),
    "beam": ("Apache Beam", "data_pipeline"),
    "celery": ("Celery", "data_pipeline"),
    "sqs": ("AWS SQS", "data_pipeline"),
    "kinesis": ("AWS Kinesis", "data_pipeline"),
    "pubsub": ("Google Pub/Sub", "data_pipeline"),
    "batch processing": ("Batch Processing", "data_pipeline"),
    "etl": ("ETL Pipeline", "data_pipeline"),
    "cron": ("Cron Jobs", "data_pipeline"),
}

# Category keywords when no specific technology is mentioned
CATEGORY_KEYWORDS = {
    "database": [
        "database", "db", "storage", "store data", "persist", "data store",
        "tables", "records", "queries", "sql", "nosql"
    ],
    "ml_model": [
        "predict", "prediction", "classification", "classify", "machine learning",
        "ml", "model", "training", "inference", "algorithm", "neural", "ai",
        "artificial intelligence", "forecast", "regression"
    ],
    "architecture": [
        "architecture", "design pattern", "system design", "structure",
        "services", "components", "modules", "layers", "backend"
    ],
    "devops": [
        "deploy", "deployment", "infrastructure", "hosting", "server",
        "container", "orchestration", "ci/cd", "pipeline", "cloud"
    ],
    "visualization": [
        "dashboard", "visualization", "visualize", "charts", "graphs",
        "reporting", "analytics dashboard", "metrics", "monitoring ui"
    ],
    "data_pipeline": [
        "pipeline", "streaming", "queue", "message", "etl", "batch",
        "real-time", "event", "processing", "workflow", "scheduling"
    ],
}


def classify_technology(user_request: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Classify technology from user request using rule-based matching.

    Returns:
        Tuple of (technology_name, category, confidence)
        - technology_name: Canonical name of technology (or None if not found)
        - category: Category of technology (or None if not found)
        - confidence: 0.0 to 1.0 (high confidence = rule match, low = needs LLM)
    """
    request_lower = user_request.lower()

    # First pass: Look for exact technology matches
    for pattern, (tech_name, category) in TECHNOLOGY_PATTERNS.items():
        if pattern in request_lower:
            # Check if it's a clear technology request
            confidence = 0.9 if any(
                phrase in request_lower
                for phrase in ["i want", "i need", "use ", "using ", "implement", "build with"]
            ) else 0.7
            return tech_name, category, confidence

    # Second pass: No specific technology, try to identify category
    category = get_category_from_keywords(request_lower)
    if category:
        return None, category, 0.5  # Lower confidence - no specific tech

    # No match found - needs LLM
    return None, None, 0.0


def get_category_from_keywords(text: str) -> Optional[str]:
    """Identify category from general keywords when no specific technology mentioned."""
    text_lower = text.lower()

    category_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            category_scores[category] = score

    if category_scores:
        return max(category_scores, key=category_scores.get)

    return None


def extract_context_hints(user_request: str) -> List[str]:
    """Extract contextual hints about requirements from the request."""
    hints = []
    request_lower = user_request.lower()

    # Data type hints
    data_hints = {
        "tabular": ["spreadsheet", "csv", "tabular", "columns", "rows", "excel", "table"],
        "document": ["json", "document", "nested", "flexible schema", "unstructured"],
        "image": ["image", "picture", "photo", "visual", "pixel", "video"],
        "text": ["text", "nlp", "language", "words", "sentences", "documents"],
        "time_series": ["time series", "temporal", "sensor", "stock", "forecast"],
        "graph": ["graph", "network", "nodes", "edges", "relationships", "connections", "embeddings"],
        "relational": ["relational", "joins", "foreign key", "normalized", "relations"],
    }

    for hint_type, keywords in data_hints.items():
        if any(kw in request_lower for kw in keywords):
            hints.append(f"data_type:{hint_type}")

    # Scale hints
    if any(kw in request_lower for kw in ["scale", "million", "billion", "high traffic", "enterprise"]):
        hints.append("scale:large")
    elif any(kw in request_lower for kw in ["small", "prototype", "mvp", "simple", "personal"]):
        hints.append("scale:small")

    # Team hints
    if any(kw in request_lower for kw in ["team", "teams", "developers", "engineers"]):
        hints.append("context:team")
    if any(kw in request_lower for kw in ["solo", "alone", "just me", "personal project"]):
        hints.append("context:solo")

    # Urgency hints
    if any(kw in request_lower for kw in ["quick", "fast", "deadline", "urgent", "asap"]):
        hints.append("timeline:urgent")
    if any(kw in request_lower for kw in ["long-term", "enterprise", "production", "robust"]):
        hints.append("timeline:long_term")

    return hints


def normalize_technology_name(tech: str) -> str:
    """Normalize technology name to canonical form."""
    tech_lower = tech.lower().strip()

    for pattern, (canonical, _) in TECHNOLOGY_PATTERNS.items():
        if pattern == tech_lower or tech_lower in pattern:
            return canonical

    # Return original with title case if not found
    return tech.strip().title()
