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

## Setup (Windows PowerShell or CMD)

1. Install **Python 3.12** from https://www.python.org/downloads/.
2. Open a terminal in the repository root and create the environment **once**:

   **Windows CMD**
   ```cmd
   py -3.12 -m venv .venv
   .venv\Scripts\activate.bat
   ```

   **Windows PowerShell**
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Create a Groq API key at https://console.groq.com/keys.
5. Create your local `.env` file:
   ```cmd
   copy .env.example .env
   ```
   Add your key to `.env`:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
6. Start the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```
7. Run tests:
   ```bash
   python -m unittest discover -s tests -v
   ```

**Never commit `.env` or a real API key to GitHub.** `.env` is for local development. For Streamlit Community Cloud, add `GROQ_API_KEY = "gsk_..."` in the app's **Settings → Secrets** panel. The app reads the same environment variable in both places.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
streamlit run app/streamlit_app.py
```

### Planning flow and student preferences

JASSI now presents resource paths before making a feasibility claim. Choose a curated learning path, or add your own learning resources in the planning flow using one entry per line:

```text
Topic ID | Resource title | https://resource-url | estimated minutes
```

For example:

```text
graphs | Graph Algorithms playlist | https://www.youtube.com/playlist?list=YOUR_ID | 240
```

The URL and duration are user-provided; JASSI does not claim to crawl or verify course contents. Check the provider's current syllabus and duration before relying on it.

Use **Study & routine preferences** in the sidebar to set your maximum study hours per day, wake/sleep times, meal-break start times, and recovery gap between sessions. JASSI uses these limits when estimating capacity and placing sessions. The feasibility check is run against the resources you actually select, not only a generic topic-hours estimate.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (keep `.env` out of the commit; `.gitignore` excludes it).
2. In https://share.streamlit.io/, choose **New app**, select this repository and branch, and set the main file to `app/streamlit_app.py`.
3. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
4. Deploy. `GroqClient` reads the key from Streamlit Secrets in the cloud and from `.env` locally.

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
- Calendar commitments repeat by weekday; multi-week availability is therefore
  an estimate rather than a fully imported date-specific calendar.
- User-supplied resources are selected using the title, URL, and duration the
  student enters. JASSI does not automatically crawl arbitrary course pages.
- Drag-and-drop manual timetable editing is not implemented yet.
