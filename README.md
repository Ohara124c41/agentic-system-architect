# Agentic Architecture Mediation System

Multi-agent system implementing Five Whys methodology for requirements elicitation in LLM-assisted technology selection. Prevents solution-jumping by separating dialogue (LLM) from evaluation (deterministic).

Accompanies the paper: **"Agentic Architecture Mediation for LLM Assistants: Preventing Solution-Jumping with Requirements Elicitation"** (IEEE Software, Special Issue on Engineering Agentic Systems, 2025)

## Overview

When developers arrive with predetermined technology preferences ("I want MongoDB," "I need Kubernetes"), they often skip requirements validation. This system uses conversational AI to extract underlying requirements through Five Whys dialogue, then applies deterministic evaluation to detect requirement mismatches and recommend appropriate alternatives.

**Key innovation:** Capability partitioning - LLMs handle natural language understanding and dialogue generation, while deterministic systems perform technology matching to eliminate hallucination risk.

## Architecture

```
┌─────────────┐
│ Interceptor │──> Identifies technology preference and domain
└─────────────┘
      │
      ▼
┌─────────────┐
│Why Validator│──> Conducts Five Whys dialogue (3 levels)
└─────────────┘    Extracts 3-5 requirements per session
      │
      ▼
┌─────────────┐
│  Evaluator  │──> Deterministic requirement matching
└─────────────┘    Knowledge-base scoring (no LLM)
      │
      ▼
┌─────────────┐
│ Recommender │──> Final recommendation with rationale
└─────────────┘
      │
      ▼
┌─────────────┐
│Ilities      │──> Quality attribute trade-off analysis
│Analyst      │    (maintainability, scalability, etc.)
└─────────────┘
```

Five specialized agents orchestrated via LangGraph state machine.

## Performance

- **Accuracy**: 100% classification accuracy across 56 test sessions
- **Cost**: $0.01 per validation session (5-6 LLM API calls)
- **Latency**: <15 seconds per session
- **Use cases**: 6 industry anti-patterns (MongoDB→PostgreSQL, CNN→XGBoost, etc.)

## Installation

### Requirements

- Python 3.9+
- OpenAI API key (GPT-4o-mini)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/agentic-architecture-mediation.git
cd agentic-architecture-mediation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Dependencies

Core dependencies (see `requirements.txt` for complete list):
- `langchain` - LLM framework
- `langgraph` - Agent orchestration
- `openai` - GPT-4o-mini API
- `pyyaml` - Configuration management
- `pydantic` - Data validation

### Docker Setup (Recommended)

For maximum reproducibility and ease of use, run via Docker:

```bash
# Clone repository
git clone https://github.com/Ohara124c41/agentic-system-architect.git
cd agentic-system-architect

# Build Docker image
docker build -t agentsysarch:latest .

# Run demo scenarios
docker run -e OPENAI_API_KEY=your_key_here agentsysarch:latest

# Or use docker-compose
cp .env.example .env
# Edit .env and add your OpenAI API key
docker-compose up demo
```

**Docker commands:**
```bash
# Run demo scenarios
docker-compose up demo

# Generate visualizations
docker-compose up visualize

# Interactive mode
docker run -it -e OPENAI_API_KEY=your_key agentsysarch:latest python main.py

# Mount outputs to persist results
docker run -v $(pwd)/outputs:/app/outputs -e OPENAI_API_KEY=your_key agentsysarch:latest
```

## Usage

### Basic Validation

```python
from src.agent import ArchitectureGovernanceAgent

agent = ArchitectureGovernanceAgent()

# User requests MongoDB for transactional banking application
result = agent.validate_request(
    "I want to use MongoDB for my banking application"
)

print(result["recommendation"])
# Output: PostgreSQL (ACID transactions required for financial data)
```

### Knowledge Base Configuration

Technology profiles define requirement tags and fit scores:

```yaml
# config/technology_profiles.yaml
- name: PostgreSQL
  category: database
  requirement_tags:
    - relational_data
    - acid_transactions
    - complex_joins
  fit_scores:
    transactional_workload: 0.95
    document_storage: 0.30
  use_cases:
    - Banking and financial systems
    - Multi-table relational data
```

### Test Scenarios

Six documented anti-patterns in `config/technology_profiles.yaml`:

1. **UC1**: MongoDB for transactional banking → Recommend PostgreSQL
2. **UC2**: CNN for tabular credit scoring → Recommend XGBoost
3. **UC3**: Microservices for small internal tool → Recommend Monolith
4. **UC4**: Kubernetes for static documentation → Recommend Lambda
5. **UC5**: D3.js for offline compliance reports → Recommend Matplotlib
6. **UC6**: LSTM for stateless API classification → Recommend CNN

Run evaluations:

```bash
python scripts/demo_cogent.py
```

## Project Structure

```
agentic_system_architect/
├── src/
│   ├── agent.py                 # Main orchestrator
│   ├── prompts.py               # System prompts
│   ├── agents/
│   │   ├── interceptor.py       # Technology identification
│   │   ├── why_validator.py     # Five Whys dialogue
│   │   ├── evaluator.py         # Deterministic matching
│   │   ├── recommender.py       # Final recommendation
│   │   └── ilities_analyst.py   # Quality attribute analysis
│   ├── tools/
│   │   ├── knowledge_base.py    # Profile management
│   │   ├── matching.py          # Requirement matching
│   │   └── classification.py    # Outcome classification
│   ├── schemas/
│   │   ├── state.py             # Agent state definitions
│   │   └── responses.py         # Structured outputs
│   └── features/
│       ├── traceability.py      # Requirement traceability
│       ├── adr_generator.py     # Architecture Decision Records
│       ├── thinking_layer.py    # Reasoning transparency
│       └── cogent_tradeoffs.py  # Trade-off analysis
├── config/
│   ├── technology_profiles.yaml # Knowledge base (15 profiles)
│   └── settings.yaml            # System configuration
├── scripts/
│   ├── demo_cogent.py           # Run test scenarios
│   └── visualize_results.py     # Generate figures
├── outputs/                     # Evaluation results (generated)
└── README.md
```

## Capability Partitioning

| Task | Approach | Rationale |
|------|----------|-----------|
| Technology identification | LLM | Natural language understanding |
| Five Whys dialogue | LLM | Conversational context tracking |
| Question generation | LLM | Adaptive follow-up questions |
| Requirement extraction | LLM | Semantic parsing of user responses |
| **Requirement matching** | **Deterministic** | **Eliminates hallucination risk** |
| **Technology scoring** | **Deterministic** | **100% reproducible results** |
| **Ranking alternatives** | **Deterministic** | **Objective comparison** |
| Recommendation rationale | LLM | Natural language explanation |

## Extending the System

### Adding New Technology Profiles

```yaml
# config/technology_profiles.yaml
- name: YourTechnology
  category: database  # or ml_model, architecture, devops, visualization
  description: "Brief technical description"
  requirement_tags:
    - your_requirement_tag
    - another_tag
  fit_scores:
    context_name: 0.85  # 0.0 to 1.0
  use_cases:
    - "Primary use case"
  limitations:
    - "Known constraint"
```

### Customizing Five Whys Depth

```yaml
# config/settings.yaml
five_whys:
  max_iterations: 3  # Adjust depth (2-5 recommended)
  min_requirements: 3  # Minimum requirements before evaluation
```

## Evaluation

Generate evaluation results:

```bash
# Run all test scenarios
python scripts/demo_cogent.py

# Visualize results
python scripts/visualize_results.py
```

Outputs:
- `outputs/transcript_*.json` - Session transcripts with structured data
- `outputs/ADR_*.md` - Architecture Decision Records
- `outputs/traceability_*.md` - Requirement-to-recommendation mappings
- `outputs/figures/*.png` - Match matrices, Pareto frontiers, coverage analysis

## Research Data

Full evaluation dataset available on IEEE DataPort:
- 56 session transcripts
- 15 technology profiles across 5 domains
- 6 documented anti-pattern scenarios
- Architecture Decision Records and traceability matrices

## Citation

```bibtex
@article{ohara2025agentic,
  title={Agentic Architecture Mediation for LLM Assistants: Preventing Solution-Jumping with Requirements Elicitation},
  author={O'Hara, Christopher Aaron},
  journal={IEEE Software},
  year={2025},
  note={Special Issue on Engineering Agentic Systems}
}
```

## License

MIT License - See LICENSE file for details

## Contact

Christopher Aaron O'Hara, PhD
Email: ohara124c41@gmail.com
IEEE Member

## Acknowledgments

Implements INCOSE Systems Engineering Handbook Five Whys methodology. Technology profiles reflect best practices from AWS Well-Architected Framework, Google Cloud Architecture Framework, and industry documentation.

Built with LangGraph (agent orchestration) and OpenAI GPT-4o-mini API.
