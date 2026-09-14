# JASSI — Agentic AI Learning Planner

JASSI is an autonomous learning-planning agent that continuously works toward a
student's learning goal under changing performance, availability, and resource
constraints. It's built as a **hybrid agentic system**: an LLM (Groq) handles
natural language understanding, generation, and reasoning, while deterministic
Python handles math, scheduling, and verification — wired together with
**LangGraph**.

## What makes this "agentic" and not just automation

| Capability | How it's implemented |
|---|---|
| Understands casual requests | Groq LLM parses free text → structured planning request |
| Explains its decisions | Groq LLM generates feasibility explanations, trade-offs, schedule reasoning |
| Finds relevant material | FAISS (or TF-IDF fallback) semantic search over resource embeddings — not keyword matching |
| Does the math correctly | Deterministic Python: feasibility hours, constraint-satisfaction scheduling, Bayesian mastery updates, conflict/prerequisite verification |
| Replans on its own | LangGraph conditional routing reschedules missed sessions, adjusts for quiz results, and reruns verification automatically |

## Architecture

```
jassi/
├── app/
│   ├── streamlit_app.py       # Dashboard UI
│   ├── graph.py               # LangGraph workflow (LLM + tool nodes)
│   ├── nodes/
│   │   ├── llm_nodes.py       # Groq API calls (NLU, NLG, reasoning)
│   │   └── tool_nodes.py      # Deterministic wrappers used by the graph
│   ├── tools/
│   │   ├── feasibility.py     # Required vs available hours
│   │   ├── scheduler.py       # Constraint-satisfaction session placement
│   │   ├── verifier.py        # Conflict / prerequisite / deadline checks
│   │   └── constraint_parser.py  # Manual / image / Google Calendar import
│   └── vectorstore/
│       └── resource_retriever.py  # FAISS semantic search over resources
├── data/
│   ├── topics.json            # Topic dependency graph
│   ├── resources.json         # Learning resource catalog
│   └── demo_student.json      # Seed student state (mastery, calendar)
├── tests/                     # Unit tests for the deterministic tools
├── demo_script.py             # Runs the 7(+2)-step end-to-end scenario in a terminal
├── requirements.txt
└── .env.example
```

## Setup (Windows CMD)

1. **Install Python 3.12** if you don't have it, from python.org.

2. **Create and activate a virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Get a free Groq API key:**
   - Go to https://console.groq.com/keys
   - Sign up (free tier, no card required) and create an API key

5. **Configure your `.env` file:**
   ```cmd
   copy .env.example .env
   ```
   Then open `.env` in a text editor and paste your key:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

6. **Run the dashboard:**
   ```cmd
   streamlit run app\streamlit_app.py
   ```
   Your browser should open automatically at `http://localhost:8501`.

7. **(Optional) Run the terminal demo instead:**
   ```cmd
   python demo_script.py
   ```

8. **(Optional) Run the unit tests:**
   ```cmd
   python -m unittest discover -s tests -v
   ```

### macOS / Linux
Same steps, but activate the venv with `source venv/bin/activate` and use `cp .env.example .env`.

## Semantic search note

`resource_retriever.py` tries to build a FAISS index using a small
`sentence-transformers` embedding model on first run. If your machine can't
download the model weights (e.g. no internet at judging time), it automatically
falls back to a TF-IDF vector space — the app still runs and still retrieves by
concept similarity rather than exact keyword match, it's just a lighter-weight
notion of "semantic." Nothing else in the app changes.

## Demo scenario (matches the required failure/replan flow)

1. Import calendar constraints (manual entry, image upload with LLM parsing, or Google Calendar)
2. Create a planning request in natural language: *"I wanna finish trees and graphs in 10 days with 70% mastery"*
3. LLM parses it into a structured request
4. Deterministic feasibility check finds a shortfall
5. LLM generates a Reality Check with 3 trade-off options; user picks one
6. FAISS retrieves relevant resources by meaning, not keywords; a schedule is built and verified
7. LLM explains why the schedule looks the way it does
8. User reports a missed session → LLM parses the update, the scheduler finds the next open slot, LLM explains the change
9. User submits a low quiz score → mastery updates via the Bayesian formula, LLM surfaces a progress insight and adjusts scope

`demo_script.py` runs all of this end-to-end from the terminal and prints every
LLM-generated explanation alongside the deterministic results, so you can see
both halves of the hybrid architecture working together.

## Known limitations / next steps

- Google Calendar import is stubbed with a normalizer function
  (`parse_google_calendar_events`) ready to receive fetched events — the OAuth
  redirect itself isn't wired up in this build.
- The calendar model treats a week as repeating; multi-week deadlines are
  approximated rather than modeled with real dates.
- Drag-and-drop manual adjustment of the timetable is not yet implemented;
  sessions can currently be changed only via the daily check-in flow.
