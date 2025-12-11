# Docker Hub: Agentic Architecture Mediation System

Multi-agent system preventing solution-jumping in LLM-assisted architecture decisions through Five Whys requirements elicitation and deterministic evaluation.

## Quick Start

```bash
# Run demo evaluation (requires OpenAI API key)
docker run -e OPENAI_API_KEY=your_key_here ohara124c41/agentsysarch:latest

# Save outputs to local directory
docker run -v $(pwd)/outputs:/app/outputs -e OPENAI_API_KEY=your_key ohara124c41/agentsysarch:latest

# Interactive mode
docker run -it -e OPENAI_API_KEY=your_key ohara124c41/agentsysarch:latest python main.py
```

## What This Does

Evaluates 6 industry anti-pattern scenarios:
1. MongoDB for transactional banking → Recommends PostgreSQL
2. CNN for tabular credit scoring → Recommends XGBoost
3. Microservices for small internal tool → Recommends Monolith
4. Kubernetes for static documentation → Recommends Lambda
5. D3.js for offline compliance reports → Recommends Matplotlib
6. LSTM for stateless API classification → Recommends CNN

## Results Generated

After running, check `outputs/` directory for:
- `transcript_*.json` - Session transcripts with structured data
- `ADR_*.md` - Architecture Decision Records
- `traceability_*.md` - Requirement traceability matrices
- `figures/*.png` - Visualizations (match matrices, Pareto frontiers)

## Environment Variables

- `OPENAI_API_KEY` (required) - Your OpenAI API key
- `OPENAI_MODEL` (optional) - Model to use (default: gpt-4o-mini)

## Performance

- **Accuracy**: 100% classification across 56 test sessions
- **Cost**: $0.01 per validation session
- **Latency**: <15 seconds per session

## Source Code

GitHub: https://github.com/Ohara124c41/agentic-system-architect

## Paper

Accompanies: **"Agentic Architecture Mediation for LLM Assistants: Preventing Solution-Jumping with Requirements Elicitation"** (IEEE Software, Special Issue on Engineering Agentic Systems, 2025)

## Dataset

Full evaluation data: IEEE DataPort (link TBD)

## Citation

```bibtex
@article{ohara2025agentic,
  title={Agentic Architecture Mediation for LLM Assistants: Preventing
         Solution-Jumping with Requirements Elicitation},
  author={O'Hara, Christopher Aaron},
  journal={IEEE Software},
  year={2025},
  note={Special Issue on Engineering Agentic Systems}
}
```

## License

MIT License with Academic Citation Requirement - See LICENSE in repository

## Contact

Christopher Aaron O'Hara, PhD
Email: ohara124c41@gmail.com
