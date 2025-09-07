# Multi‑Agent Deep Search (Python)

A minimal, batteries‑included **multi‑agent deep search** reference project inspired by the Panaversity `DeepSearch` idea.  
It coordinates multiple lightweight agents — *Requirement Gatherer → Planner → Orchestrator Researcher → synthesizer → Writer* — to search the web, read sources, and produce a structured, cited research brief.

## Features
- 🔎 Search via Tavily (API key required), Gemini(API Key), OpenAI(API key) for tracing .
- 🌐 Robust page fetching + readable extraction (trafilatura).
- 🧠 Multi‑agent pipeline with iterative critique/refinement.
- 🧵 Rich console logs.
- 📦 Deterministic outputs shown in summery text format.
- ⚙️ Configurable depth, breadth.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python deep_search.py --query "Compare the environmental impact of electric vs hybrid vs gas cars.?"
```

Outputs are shown detail text format.

## Project Layout
```
deep_search_Project/
  __main__
  agents             # Requirement Gatherer, Planner,Orchestrator,Researcher,synthesizer, Writer
  tools              # research_as_tools, synthesizer_as_tools, writer_as_tools,Search, extract_content
  deep_search.py     # Multi-agent controller with iterative refinement
  Handoff            # Requirement Gatherer → Planner → Orchestrator
deep_search.py       # CLI entrypoint
requirements.txt
README.md
```

## Notes
- Internet access is required at runtime for fetching pages.
- We can change the new search by replacing text in [PROMPT_MSG].

## Demo Video Link

https://drive.google.com/file/d/1LzBBRwsAQodmZDF9qf1qyqIz1Sh1HD_9/view?usp=drive_link

## License
MIT


