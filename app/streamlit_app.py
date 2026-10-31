"""JASSI — Agentic AI Learning Planner. Streamlit dashboard entry point."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
from datetime import datetime, date, timedelta
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from app.nodes.llm_nodes import GroqClient
from app.tools import feasibility as feasibility_tool
from app.tools.constraint_parser import manual_event, parse_image_with_llm, resolve_event_time_conflict
from app.tools.resource_lookup import resolve_resource_url
from app.vectorstore.resource_retriever import ResourceRetriever
from app.graph import build_planning_request_graph, build_daily_update_graph
from app.tools.schedule_recovery import reconcile_overdue_sessions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOPICS_PATH = os.path.join(DATA_DIR, "topics.json")
RESOURCES_PATH = os.path.join(DATA_DIR, "resources.json")
STUDENT_PATH = os.path.join(DATA_DIR, "demo_student.json")

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="JASSI · Learning Planner", page_icon="🧭", layout="wide",
                    initial_sidebar_state="expanded")
APP_BUILD_VERSION = "2026-10-31-plan-day-workspace-resource-relevance"
st.session_state.setdefault("_jassi_processing", False)
st.session_state.setdefault("selected_plan_day", None)
st.session_state.setdefault("confirm_delete_plan", None)

# ---------------------------------------------------------------------------
# Premium visual theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg-0: #0b0d14;
  --bg-1: #12151f;
  --bg-2: #191d2b;
  --panel: #161a26;
  --border: rgba(255,255,255,0.07);
  --border-hover: rgba(255,255,255,0.14);
  --text-0: #ffffff;
  --text-1: #e7e9f2;
  --text-2: #c3c6d8;
  --accent: #7c8cff;
  --accent-2: #b28cff;
  --accent-soft: rgba(124,140,255,0.14);
  --green: #4ade80;
  --yellow: #fbbf24;
  --red: #f87171;
  --blue: #60a5fa;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .jassi-title { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
  background: radial-gradient(circle at 15% 0%, #1a1d2e 0%, var(--bg-0) 45%) fixed;
  color: var(--text-0);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
  border-right: 1px solid var(--border);
}

* { transition: background-color 180ms ease, border-color 180ms ease, transform 180ms ease, box-shadow 220ms ease, opacity 220ms ease; }

.jassi-header {
  display:flex; align-items:center; gap:12px; margin-bottom: 4px;
}
.jassi-logo {
  width:38px; height:38px; border-radius:11px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:17px; color:white;
  box-shadow: 0 4px 18px rgba(124,140,255,0.35);
}
.jassi-title { font-size:22px; font-weight:800; letter-spacing:-0.02em; color: var(--text-0); }
.jassi-subtitle { color: var(--text-2); font-size:12.5px; margin-top:-2px; }

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
  box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset, 0 12px 30px -20px rgba(0,0,0,0.6);
}
.card:hover { border-color: var(--border-hover); }

.pr-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 13px 15px;
  margin-bottom: 10px;
  cursor: pointer;
}
.pr-card:hover { border-color: var(--accent); transform: translateY(-1px); box-shadow: 0 10px 24px -16px rgba(124,140,255,0.5); }
.pr-card.active { border-color: var(--accent); background: linear-gradient(135deg, var(--accent-soft), transparent); }

.badge {
  display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight:600;
  letter-spacing: 0.02em;
}
.badge-active { background: rgba(74,222,128,0.15); color: var(--green); }
.badge-blocked { background: rgba(248,113,113,0.15); color: var(--red); }
.badge-completed { background: rgba(96,165,250,0.15); color: var(--blue); }

.progress-outer {
  width:100%; height:8px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow:hidden; margin-top:8px;
}
.progress-inner {
  height:100%; border-radius:999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  transition: width 500ms cubic-bezier(.22,1,.36,1);
}

.slot {
  border-radius: 10px; padding: 6px 8px; font-size: 11.5px; font-weight:600;
  margin-bottom: 4px; line-height:1.25; border-left: 3px solid transparent;
  animation: fadeIn 320ms ease;
}
@keyframes fadeIn { from { opacity:0; transform: translateY(3px); } to { opacity:1; transform: translateY(0); } }
.slot-hard { background: rgba(248,113,113,0.12); border-left-color: var(--red); color: #ffb4b4; }
.slot-soft { background: rgba(251,191,36,0.10); border-left-color: var(--yellow); color: #ffdd9e; }
.slot-study { background: rgba(124,140,255,0.14); border-left-color: var(--accent); color: #d3d8ff; }

.day-col-header {
  text-align:center; font-weight:700; font-size:12.5px; color: var(--text-1);
  padding: 6px 0; border-bottom: 1px solid var(--border); margin-bottom: 8px;
  text-transform: uppercase; letter-spacing: 0.04em;
}

/* ---- Weekly timetable: fixed one-hour grid (rows = hours, cols = days) ---- */
.tt-table {
  width: 100%; border-collapse: collapse; table-layout: fixed;
}
.tt-table th, .tt-table td { border: 1px solid var(--border); }
.tt-day-header {
  text-align:center; font-weight:700; font-size:12px; color: var(--text-1);
  padding: 8px 4px; text-transform: uppercase; letter-spacing: 0.04em;
  background: var(--bg-2);
}
.tt-day-sub { display:none; }
.tt-time-col { width: 64px; background: var(--bg-2); }
.tt-time-label {
  width: 64px; font-size: 10.5px; color: var(--text-2); text-align:right;
  padding: 4px 8px; white-space:nowrap; vertical-align: top; background: var(--bg-2);
}
.tt-cell { height: 46px; padding: 0; vertical-align: top; }
.tt-empty { background: transparent; }
.tt-cell.tt-multi { height: auto; display: flex; flex-direction: column; gap: 1px; }
.tt-slot {
  height:100%; width:100%; padding: 4px 6px; font-size: 11px; font-weight:600;
  line-height:1.25; overflow:hidden; animation: fadeIn 320ms ease;
}
.tt-slot-chip { border-left: 3px solid transparent; }
.tt-cell:not(.tt-multi) .tt-slot-chip { height: 100%; }
.tt-cell.tt-multi .tt-slot-chip { flex: 1; min-height: 22px; font-size: 10px; padding: 2px 5px; }
.slot-hard.tt-slot-chip { background: rgba(248,113,113,0.14); border-left-color: var(--red); color: #ffb4b4; }
.slot-soft.tt-slot-chip { background: rgba(251,191,36,0.12); border-left-color: var(--yellow); color: #ffdd9e; }
.slot-study.tt-slot-chip { background: rgba(124,140,255,0.16); border-left-color: var(--accent); color: #d3d8ff; }

.metric-label { color: var(--text-2); font-size: 11.5px; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }
.metric-value { font-size: 26px; font-weight:800; color: var(--text-0); margin-top: 2px; }

.chat-bubble {
  background: linear-gradient(135deg, rgba(124,140,255,0.10), rgba(178,140,255,0.06));
  border: 1px solid rgba(124,140,255,0.2);
  border-radius: 14px; padding: 12px 15px; font-size: 13.5px; color: var(--text-0);
  margin-bottom: 10px; animation: fadeIn 380ms ease;
}

.timeline-item {
  border-left: 2px solid var(--border); padding-left: 14px; padding-bottom: 14px; position:relative; font-size:12.5px; color: var(--text-1);
}
.timeline-item::before {
  content:''; position:absolute; left:-5px; top:2px; width:8px; height:8px; border-radius:50%;
  background: var(--accent); box-shadow: 0 0 0 3px rgba(124,140,255,0.2);
}

.stButton>button {
  border-radius: 10px; border: 1px solid var(--border); background: var(--bg-2); color: var(--text-0);
  font-weight:600; font-size: 13px;
}
.stButton>button:hover { border-color: var(--accent); color: var(--accent); }

div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: var(--panel); }

/* ---- Force readable white/near-white text everywhere Streamlit defaults to grey ---- */
p, span, label, li, .stMarkdown, .stCaption, .stText, .stTextInput label, .stTextArea label,
.stSelectbox label, .stSlider label, .stRadio label, .stTimeInput label, .stFileUploader label,
div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li,
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p,
div[data-testid="stWidgetLabel"] p, div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p, div[data-testid="stForm"] label {
  color: var(--text-1) !important;
}
h1, h2, h3, h4, h5, h6 { color: var(--text-0) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  color: var(--text-0) !important; background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-2) !important; opacity: 1; }
div[data-testid="stTabs"] button p { color: var(--text-1) !important; }
div[data-testid="stTabs"] button[aria-selected="true"] p { color: var(--text-0) !important; }

/* ---- Bottom-center planning request bar, Claude-style ---- */
div[data-testid="stChatInput"] {
  max-width: 760px;
  margin: 0 auto;
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 22px !important;
  box-shadow: 0 18px 40px -18px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.02) inset;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(124,140,255,0.18), 0 18px 40px -18px rgba(0,0,0,0.65);
}
div[data-testid="stChatInput"] textarea {
  color: var(--text-0) !important; font-size: 14px !important; font-weight: 500;
}
div[data-testid="stChatInput"] textarea::placeholder { color: var(--text-2) !important; opacity: 1; }
div[data-testid="stBottomBlockContainer"] {
  background: linear-gradient(0deg, var(--bg-0) 55%, transparent 100%) !important;
  padding-bottom: 10px !important;
}

/* ---- Loading animations ---- */
@keyframes jassiPulse { 0%,100% { opacity:1; } 50% { opacity:0.45; } }
@keyframes jassiSpin { to { transform: rotate(360deg); } }
@keyframes jassiShimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }

div[data-testid="stSpinner"] {
  display:flex; justify-content:center;
}
div[data-testid="stSpinner"] > div {
  display:flex; align-items:center; justify-content:center; gap:10px;
}
/* Only the actual spinner icon rotates — explicitly excluded from any div that
   wraps the message text, so the label never spins along with it. */
div[data-testid="stSpinner"] > div > div:not(:has(p)):not([data-testid="stMarkdownContainer"]) {
  border-top-color: var(--accent) !important;
  border-right-color: var(--accent-2) !important;
  animation: jassiSpin 700ms linear infinite !important;
}
div[data-testid="stSpinner"] p { color: var(--text-1) !important; animation: jassiPulse 1.4s ease-in-out infinite; }

.jassi-skeleton {
  border-radius: 10px; height: 14px; margin-bottom: 8px;
  background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.09) 37%, rgba(255,255,255,0.04) 63%);
  background-size: 800px 100%;
  animation: jassiShimmer 1.6s linear infinite;
}

.jassi-typing { display:inline-flex; gap:4px; align-items:center; padding: 2px 0; }
.jassi-typing span {
  width:6px; height:6px; border-radius:50%; background: var(--accent);
  animation: jassiTypingBounce 1.1s ease-in-out infinite;
}
.jassi-typing span:nth-child(2) { animation-delay: 0.15s; }
.jassi-typing span:nth-child(3) { animation-delay: 0.3s; }
@keyframes jassiTypingBounce { 0%,60%,100% { transform: translateY(0); opacity:0.5; } 30% { transform: translateY(-4px); opacity:1; } }

/* ---- Centered progress status (only the icon spins; text just shimmers) ---- */
.jassi-status-row {
  display:flex; align-items:center; gap:10px; justify-content:center; text-align:center;
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 11px 16px; margin: 8px auto 14px auto; animation: fadeIn 250ms ease;
  max-width: 480px;
}
.jassi-status-icon {
  font-size: 18px; display:inline-block; animation: jassiSpin 1.1s linear infinite;
  transform-origin: 50% 50%; flex-shrink: 0;
}
.jassi-status-text {
  font-size: 13.5px; font-weight: 600; color: var(--text-0) !important;
  background: linear-gradient(90deg, var(--text-2) 0%, var(--text-0) 50%, var(--text-2) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  animation: jassiTextShimmer 1.6s linear infinite;
}
@keyframes jassiTextShimmer { 0% { background-position: 200% center; } 100% { background-position: -200% center; } }

/* ---- Month calendar (trajectory across the whole deadline window) ---- */
.mc-table { width:100%; border-collapse: collapse; table-layout: fixed; }
.mc-table th, .mc-table td { border: 1px solid var(--border); vertical-align: top; }
.mc-day-header {
  text-align:center; font-weight:700; font-size:12px; color: var(--text-1);
  padding: 6px 4px; text-transform: uppercase; letter-spacing: 0.04em; background: var(--bg-2);
}
.mc-cell { height: 84px; padding: 4px 5px; }
.mc-cell.mc-today { background: rgba(124,140,255,0.08); box-shadow: inset 0 0 0 1px var(--accent); }
.mc-date { font-size: 11px; color: var(--text-2); font-weight:700; margin-bottom: 3px; }
.mc-chip {
  font-size: 10px; font-weight:600; border-radius: 5px; padding: 1px 4px; margin-bottom: 2px;
  white-space: nowrap; overflow:hidden; text-overflow: ellipsis;
}
.mc-hard { background: rgba(248,113,113,0.16); color: #ffb4b4; }
.mc-soft { background: rgba(251,191,36,0.14); color: #ffdd9e; }
.mc-study { background: rgba(124,140,255,0.18); color: #d3d8ff; }
.mc-more { font-size: 9.5px; color: var(--text-2); }

/* ---- Resource path / day-preference choice cards ---- */
.path-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 16px; margin-bottom: 10px; min-height: 165px;
}
.path-pros-cons { font-size: 11.5px; margin-top: 8px; }
.path-pros-cons b { color: var(--text-1); }
.dynamic-tag {
  display:inline-block; font-size: 10px; font-weight:700; letter-spacing:0.03em;
  color: var(--accent-2); border: 1px solid rgba(178,140,255,0.35); border-radius: 999px;
  padding: 1px 8px; margin-left: 6px; vertical-align: middle;
}

/* ---- Editable hour-grid timetable (native buttons, not a raw HTML table) ---- */
.st-key-tt_grid [data-testid="stHorizontalBlock"] {
  border-bottom: 1px solid var(--border);
}
.st-key-tt_grid [data-testid="column"] {
  border-right: 1px solid var(--border);
  padding: 2px 3px !important;
}
.st-key-tt_grid [data-testid="column"]:first-child {
  border-right: 2px solid var(--border);
  background: var(--bg-2);
}
.st-key-tt_grid .stButton { margin-bottom: 2px; }
.st-key-tt_grid .stButton > button {
  padding: 3px 7px !important; font-size: 10.5px !important; font-weight: 600 !important;
  min-height: 26px !important; line-height: 1.25 !important; white-space: normal !important;
  text-align: left !important; border-radius: 7px !important;
}
.tt-empty-cell { min-height: 26px; }

/* ---- Contrast fixes: native Streamlit portals (dialogs, dropdowns, time
   pickers) render OUTSIDE .stApp's dark background, so text forced light by
   the rule above can end up light-on-light / invisible. Force dark panel
   styling explicitly wherever Streamlit renders these overlays. ---- */
div[data-testid="stDialog"] div[role="dialog"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
}
div[data-testid="stDialog"] * {
  color: var(--text-1) !important;
}
div[data-testid="stDialog"] h1, div[data-testid="stDialog"] h2, div[data-testid="stDialog"] h3 {
  color: var(--text-0) !important;
}
div[data-testid="stDialog"] .stTextInput input,
div[data-testid="stDialog"] .stSelectbox div[data-baseweb="select"],
div[data-testid="stDialog"] .stTimeInput input {
  background: var(--bg-2) !important; color: var(--text-0) !important;
}
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {
  background: var(--bg-1) !important;
}
div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[data-baseweb="menu"] * {
  color: var(--text-0) !important;
}
div[data-baseweb="calendar"], div[data-baseweb="datepicker"] {
  background: var(--bg-1) !important; color: var(--text-0) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State bootstrap
# ---------------------------------------------------------------------------
def load_student():
    with open(STUDENT_PATH) as f:
        return json.load(f)


def save_student(data):
    with open(STUDENT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _single_flight_action(fn):
    """Prevent duplicate execution of expensive actions in one Streamlit session."""
    from functools import wraps

    @wraps(fn)
    def _wrapped(*args, **kwargs):
        if st.session_state.get("_jassi_processing", False):
            st.info("JASSI is already processing that request. Please wait for it to finish.")
            return None
        st.session_state["_jassi_processing"] = True
        try:
            return fn(*args, **kwargs)
        finally:
            st.session_state["_jassi_processing"] = False
    return _wrapped


if "student" not in st.session_state:
    st.session_state.student = load_student()
st.session_state.student.setdefault("study_preferences", {
    "max_daily_study_hours": 3.0, "wake_start": "07:00", "sleep_start": "23:00",
    "lunch_start": "12:30", "dinner_start": "19:30", "break_minutes": 15,
})
# Reconcile dated study sessions on every app entry/rerun. The recovery
# function is idempotent, so a refresh cannot create duplicate replacement tasks.
_recovery_changes = reconcile_overdue_sessions(st.session_state.student)
if _recovery_changes:
    save_student(st.session_state.student)
    _recovered = sum(1 for item in _recovery_changes if item.get("replacement"))
    _unplaced = len(_recovery_changes) - _recovered
    if _recovered:
        st.toast(f"JASSI recovered {_recovered} overdue session(s) and updated your learning plan.", icon="🗓️")
    if _unplaced:
        st.warning(f"{_unplaced} overdue session(s) could not be placed automatically. Your plan keeps them marked as missed for review.")
if "selected_pr" not in st.session_state:
    st.session_state.selected_pr = None
if "roadmap_open" not in st.session_state:
    st.session_state.roadmap_open = False
if "reasoning_log" not in st.session_state:
    st.session_state.reasoning_log = []
if "pending_tradeoffs" not in st.session_state:
    st.session_state.pending_tradeoffs = None
if "pending_day_prefs" not in st.session_state:
    st.session_state.pending_day_prefs = None
if "pending_time_prefs" not in st.session_state:
    st.session_state.pending_time_prefs = None
if "pending_schedule_conflict" not in st.session_state:
    st.session_state.pending_schedule_conflict = None
if "pending_resource_choice" not in st.session_state:
    st.session_state.pending_resource_choice = None
if "editing_event_id" not in st.session_state:
    st.session_state.editing_event_id = None
if "retriever" not in st.session_state:
    with st.spinner("Indexing learning resources for semantic search..."):
        st.session_state.retriever = ResourceRetriever(RESOURCES_PATH)

topics_data = feasibility_tool.load_topics(TOPICS_PATH)


def get_client():
    try:
        return GroqClient()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


def log_reasoning(text: str):
    st.session_state.reasoning_log.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "text": text})
    st.session_state.reasoning_log = st.session_state.reasoning_log[:12]


def render_status(slot, icon: str, text: str):
    """Renders a centered spinning-icon + shimmering-text status row into `slot`.
    Only the icon spins — the text just shimmers, it never rotates."""
    slot.markdown(f"""
    <div class="jassi-status-row">
      <span class="jassi-status-icon">{icon}</span>
      <span class="jassi-status-text">{text}</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Weekly timetable — rendered as a real hour-by-hour grid (fixed one-hour
# rows, colored cells spanning the rows they cover), instead of a plain
# per-day list, so it reads like an actual timetable at a glance.
# ---------------------------------------------------------------------------
def _time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _format_hour_label(hour: int) -> str:
    hour = hour % 24
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12
    if display == 0:
        display = 12
    return f"{display}:00{suffix}"


def _format_display_time(t: str) -> str:
    h, m = t.split(":")
    h = int(h)
    suffix = "AM" if h < 12 else "PM"
    display = h % 12
    if display == 0:
        display = 12
    return f"{display}:{m}{suffix}"


def render_timetable_html(events_by_day: dict, day_order: list) -> str:
    """Builds an HTML table with one row per clock hour and one column per day.
    A cell lists EVERY event that overlaps that hour on that day, stacked
    side-by-side — so two events in the same window both show up instead of
    one silently hiding the other (the old rowspan approach only had room
    for a single event per day-hour cell)."""
    type_class = {"hard": "slot-hard", "soft": "slot-soft", "study_session": "slot-study"}
    type_icon = {"hard": "🔴", "soft": "🟡", "study_session": "🟢"}

    all_events = [ev for d in day_order for ev in events_by_day.get(d, [])]
    if all_events:
        min_start = min(_time_to_minutes(ev["start"]) for ev in all_events)
        max_end = max(_time_to_minutes(ev["end"]) for ev in all_events)
    else:
        min_start, max_end = 8 * 60, 18 * 60

    start_hour = max(0, min(min_start // 60, 8))          # never start later than 8AM
    end_hour = max(-(-max_end // 60), start_hour + 1, 18)  # never end earlier than 6PM
    end_hour = min(end_hour, 23)
    hours = list(range(start_hour, end_hour + 1))  # last entry is the trailing boundary row
    n_rows = len(hours) - 1

    header_cells = "".join(f'<th class="tt-day-header">{d[:3]}<br><span class="tt-day-sub">{d}</span></th>'
                            for d in day_order)
    rows_html = [f'<tr><th class="tt-time-col"></th>{header_cells}</tr>']

    for r in range(n_rows):
        hour_start_min = hours[r] * 60
        hour_end_min = hours[r + 1] * 60
        label = _format_hour_label(hours[r])
        cells = [f'<td class="tt-time-label">{label}</td>']
        for d in day_order:
            overlapping = [
                ev for ev in events_by_day.get(d, [])
                if _time_to_minutes(ev["start"]) < hour_end_min and _time_to_minutes(ev["end"]) > hour_start_min
            ]
            if not overlapping:
                cells.append('<td class="tt-cell tt-empty"></td>')
                continue
            overlapping.sort(key=lambda e: e["start"])
            chips = []
            for ev in overlapping:
                cls = type_class.get(ev.get("type"), "slot-study")
                icon = type_icon.get(ev.get("type"), "🟢")
                title = ev["title"]
                time_range = f'{_format_display_time(ev["start"])}–{_format_display_time(ev["end"])}'
                chips.append(
                    f'<div class="tt-slot tt-slot-chip {cls}" title="{title} ({time_range})">'
                    f'{icon} <b>{title[:22]}</b><br>{time_range}</div>'
                )
            multi_cls = " tt-multi" if len(chips) > 1 else ""
            cells.append(f'<td class="tt-cell{multi_cls}">{"".join(chips)}</td>')
        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    return f'<table class="tt-table">{"".join(rows_html)}</table>'


def render_month_calendar_html(events: list, start_date, num_weeks: int = 4) -> str:
    """Grid view spanning several weeks so a multi-week deadline is actually
    visible as a trajectory, not just "this week". Recurring hard/soft
    commitments repeat on their weekday every week shown; dated study
    sessions (from the date-aware scheduler) appear on their exact date."""
    import datetime as _dt
    type_class = {"hard": "mc-hard", "soft": "mc-soft", "study_session": "mc-study"}
    type_icon = {"hard": "🔴", "soft": "🟡", "study_session": "🟢"}

    recurring_by_weekday: dict = {}
    dated_by_date: dict = {}
    for e in events:
        if e.get("date"):
            dated_by_date.setdefault(e["date"], []).append(e)
        else:
            recurring_by_weekday.setdefault(e["day"], []).append(e)

    grid_start = start_date - _dt.timedelta(days=start_date.weekday())
    header = "".join(f'<th class="mc-day-header">{d[:3]}</th>' for d in DAY_ORDER)
    rows_html = [f"<tr>{header}</tr>"]
    for w in range(num_weeks):
        cells = []
        for i in range(7):
            d = grid_start + _dt.timedelta(days=w * 7 + i)
            weekday = DAY_ORDER[d.weekday()]
            items = list(recurring_by_weekday.get(weekday, [])) + dated_by_date.get(d.isoformat(), [])
            items.sort(key=lambda e: e["start"])
            chips = "".join(
                f'<div class="mc-chip {type_class.get(it.get("type"), "mc-study")}" title="{it["title"]}">'
                f'{type_icon.get(it.get("type"), "🟢")} {it["title"][:14]}</div>'
                for it in items[:3]
            )
            more = f'<div class="mc-more">+{len(items) - 3} more</div>' if len(items) > 3 else ""
            today_cls = " mc-today" if d == start_date else ""
            cells.append(f'<td class="mc-cell{today_cls}"><div class="mc-date">{d.day}</div>{chips}{more}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="mc-table">{"".join(rows_html)}</table>'


@st.dialog("Edit event")
def edit_event_dialog(ev: dict):
    """In-place edit/delete popup opened by clicking an event chip on the
    calendar — replaces the old separate 'Manage Events' list below it."""
    student = st.session_state.student
    is_study = ev.get("type") == "study_session"
    new_title = st.text_input("Title", value=ev["title"])
    new_day = st.selectbox("Day", DAY_ORDER, index=DAY_ORDER.index(ev["day"]))
    c1, c2 = st.columns(2)
    new_start = c1.time_input("Start", value=datetime.strptime(ev["start"], "%H:%M").time())
    new_end = c2.time_input("End", value=datetime.strptime(ev["end"], "%H:%M").time())
    new_type = st.radio("Type", ["hard", "soft"], horizontal=True,
                         index=0 if ev.get("type") != "soft" else 1, disabled=is_study)
    if is_study:
        st.caption("Study sessions keep their type — JASSI scheduled this one.")

    b1, b2 = st.columns(2)
    if b1.button("Save changes", type="primary", use_container_width=True):
        ev["title"] = new_title
        ev["day"] = new_day
        ev["start"] = new_start.strftime("%H:%M")
        ev["end"] = new_end.strftime("%H:%M")
        if not is_study:
            ev["type"] = new_type
        # study_session events also live inside a pr's own session list —
        # keep both copies consistent.
        for p in student["planning_requests"]:
            for s in p["sessions"]:
                if s["id"] == ev["id"]:
                    s["day"], s["start"], s["end"] = new_day, ev["start"], ev["end"]
        save_student(student)
        st.toast("Event updated", icon="✅")
        st.rerun()
    if b2.button("Delete event", use_container_width=True):
        student["calendar_events"] = [e for e in student["calendar_events"] if e["id"] != ev["id"]]
        for p in student["planning_requests"]:
            p["sessions"] = [s for s in p["sessions"] if s["id"] != ev["id"]]
        save_student(student)
        st.toast(f"Deleted {ev['title']}", icon="🗑️")
        st.rerun()


def render_editable_timetable_grid(events_by_day: dict, day_order: list):
    """Same fixed one-hour-row / one-day-column grid as before, but every
    chip is now a real button — click it to edit/delete via a popup. Cells
    grow taller automatically when more than one event lands in that hour."""
    type_icon = {"hard": "🔴", "soft": "🟡", "study_session": "🟢"}
    all_events = [ev for d in day_order for ev in events_by_day.get(d, [])]
    if all_events:
        min_start = min(_time_to_minutes(ev["start"]) for ev in all_events)
        max_end = max(_time_to_minutes(ev["end"]) for ev in all_events)
    else:
        min_start, max_end = 8 * 60, 18 * 60

    start_hour = max(0, min(min_start // 60, 8))          # never start later than 8AM
    end_hour = max(-(-max_end // 60), start_hour + 1, 18)  # never end earlier than 6PM
    end_hour = min(end_hour, 23)
    hours = list(range(start_hour, end_hour + 1))

    grid = st.container(key="tt_grid")
    with grid:
        header_cols = st.columns([0.7] + [1] * 7)
        header_cols[0].markdown("&nbsp;", unsafe_allow_html=True)
        for i, d in enumerate(day_order):
            header_cols[i + 1].markdown(f'<div class="tt-day-header">{d[:3]}</div>', unsafe_allow_html=True)

        for r in range(len(hours) - 1):
            hour_start_min = hours[r] * 60
            hour_end_min = hours[r + 1] * 60
            row_cols = st.columns([0.7] + [1] * 7)
            row_cols[0].markdown(f'<div class="tt-time-label">{_format_hour_label(hours[r])}</div>',
                                  unsafe_allow_html=True)
            for i, d in enumerate(day_order):
                with row_cols[i + 1]:
                    overlapping = [
                        ev for ev in events_by_day.get(d, [])
                        if _time_to_minutes(ev["start"]) < hour_end_min and _time_to_minutes(ev["end"]) > hour_start_min
                    ]
                    if not overlapping:
                        st.markdown('<div class="tt-empty-cell"></div>', unsafe_allow_html=True)
                        continue
                    overlapping.sort(key=lambda e: e["start"])
                    for ev in overlapping:
                        icon = type_icon.get(ev.get("type"), "🟢")
                        time_range = f'{_format_display_time(ev["start"])}–{_format_display_time(ev["end"])}'
                        label = f'{icon} {ev["title"][:16]}'
                        if st.button(label, key=f"evbtn_{ev['id']}_{hours[r]}", use_container_width=True,
                                     help=f"{ev['title']} · {time_range}", disabled=_processing):
                            edit_event_dialog(ev)


UPDATE_STEP_LABELS = {
    "parse_update": ("🧭", "Reading your update..."),
    "update_mastery": ("🧮", "Updating your mastery score..."),
    "reschedule_if_missed": ("🗓️", "Finding a new slot..."),
    "analyze_progress": ("💡", "Spotting patterns in your progress..."),
}


@_single_flight_action
def stream_graph_with_status(graph, initial_state: dict, step_labels: dict, slot) -> dict:
    """Streams a compiled LangGraph graph node-by-node, updating `slot` with a
    centered spinning status ('..retrieving', '..working on it', etc.)
    as each node runs, and returns the final accumulated state."""
    final_state = dict(initial_state)
    render_status(slot, "✨", "Working on it...")
    try:
        for update in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_state in update.items():
                if isinstance(node_state, dict):
                    final_state.update(node_state)
                icon, label = step_labels.get(node_name, ("✨", "Working on it..."))
                render_status(slot, icon, label)
    except Exception:
        # fall back to a single blocking call if streaming isn't supported
        render_status(slot, "✨", "Working on it...")
        final_state = graph.invoke(initial_state)
    slot.empty()
    return final_state


# ---------------------------------------------------------------------------
# Planning-request pipeline — run as plain sequential node calls (not a single
# streamed compiled-graph call) because the flow now genuinely pauses twice
# for the student's input: once to confirm preferred study days, and again to
# choose a resource path. Each pause stashes the in-progress state in
# session_state and resumes it on the next Streamlit rerun.
# ---------------------------------------------------------------------------
@_single_flight_action
def create_planning_request_flow(nl_input: str):
    """Kicks off a new planning request: parses the goal, then asks for a
    day-of-week preference (if the student didn't already state one) before
    doing anything else — so scheduling starts from an accurate picture of
    when they're actually willing to study."""
    from app.graph import build_planning_intake_graph
    client = get_client()
    student = st.session_state.student
    state = {
        "client": client,
        "topics_path": TOPICS_PATH,
        "raw_text": nl_input,
        "known_topic_ids": [t["id"] for t in topics_data["topics"]],
        "current_mastery": dict(student["mastery_map"]),
        "calendar_events": [e for e in student["calendar_events"]],
        "retriever": st.session_state.retriever,
        "existing_sessions": [],
        "moved_soft_event_ids": [],
        "study_preferences": dict(student.get("study_preferences", {})),
        "reference_resources": [],
    }
    intake_graph = build_planning_intake_graph()
    state = stream_graph_with_status(
        intake_graph, state,
        {"parse_planning_request": ("🧭", "Understanding your goal...")},
        status_slot,
    )

    if state.get("preferred_days"):
        _ask_time_prefs(state)
    else:
        st.session_state.pending_day_prefs = {"state": state}
        st.rerun()


def _ask_time_prefs(state: dict):
    """Phase between day-preference and actual scheduling: ask what time of
    day the student actually wants to study, so the smart scheduler can bias
    placement toward it (and away from lunch/dinner) instead of just filling
    the earliest gap it finds."""
    st.session_state.pending_time_prefs = {"state": state}
    st.rerun()


def _continue_planning_after_days(state: dict):
    """Phase 2: resolve the topic (matching the curated catalog, or asking
    the LLM to synthesize a fresh one if it's a genuine but out-of-catalog
    learning goal), check feasibility, and either offer resource-path choices
    or a trade-off (reality check) if it isn't feasible as asked."""
    from app.graph import build_planning_resource_graph

    # The graph owns topic synthesis/routing and resource-path generation.
    # Streamlit pauses after this graph so the student can choose a path.
    resource_graph = build_planning_resource_graph()
    state = stream_graph_with_status(
        resource_graph, state,
        {
            "synthesize_topic": ("🧪", "Checking the topic and designing it if needed..."),
            "generate_resource_paths": ("🔎", "Finding learning resources and building study paths..."),
            "no_topic_match": ("🤔", "Checking whether this is a learning goal..."),
        },
        status_slot,
    )
    if not state.get("pr_topics"):
        msg = state.get("unmatched_reason") or "That doesn't look like a learning goal JASSI can plan for."
        st.toast(msg, icon="🤔")
        log_reasoning(msg)
        st.rerun()
        return
    st.session_state.pending_resource_choice = {
        "pr_id": f"pr_{uuid.uuid4().hex[:6]}", "state": state, "chosen": {},
    }
    st.rerun()


def finalize_plan_from_paths(pr_id: str, state: dict, chosen_paths: dict):
    """Phase 3: build the actual dated schedule from the resource path(s) the
    student picked, verify it, and save the new planning request."""
    state["sessions_needed"] = build_sessions_from_chosen_paths_safe(chosen_paths, state)
    state["start_date"] = date.today()
    prefs = state.get("study_preferences") or {}
    required = round(sum(s.get("duration_mins", 30) for s in state["sessions_needed"]) / 60.0, 2)
    available = feasibility_tool.available_hours_in_window(
        state["calendar_events"], state["pr_deadline_days"],
        allowed_days=state.get("preferred_days") or None,
        max_daily_study_hours=prefs.get("max_daily_study_hours", 3.0),
        wake_start=prefs.get("wake_start", "07:00"), sleep_start=prefs.get("sleep_start", "23:00"),
        start_date=date.today())
    shortfall = round(max(0.0, required - available), 2)
    state["feasibility_result"] = {
        "required_hours": required, "available_hours": available,
        "shortfall_hours": shortfall, "feasible": shortfall <= 0.01,
        "reject": shortfall > 0 and (available <= 0.01 or shortfall > available * 2),
        "risks": feasibility_tool.assess_schedule_risks(
            state["calendar_events"], state["pr_deadline_days"],
            state.get("preferred_days") or None, start_date=date.today()),
        "daily_study_limit_hours": prefs.get("max_daily_study_hours", 3.0),
    }
    if not state["feasibility_result"]["feasible"]:
        from app.graph import node_generate_tradeoffs
        state["feasibility_explanation"] = (
            f"The resources you selected need about {required:g} study hours, but your current routine "
            f"comfortably allows about {available:g} hours before the deadline. "
            f"That leaves a gap of {shortfall:g} hours. We should adjust the deadline, workload, or commitments before scheduling."
        )
        state = node_generate_tradeoffs(state)
        st.session_state.pending_tradeoffs = {"pr_id": pr_id, "result": state}
        st.rerun()
        return
    _build_schedule_and_continue(pr_id, state)


def build_sessions_from_chosen_paths_safe(chosen_paths: dict, state: dict):
    """chosen_paths is {} when we're just re-running the scheduler after a
    conflict decision (not picking fresh resource paths) — in that case keep
    whatever sessions_needed was already computed on the first pass."""
    from app.graph import build_sessions_from_chosen_paths
    if not chosen_paths:
        return state.get("sessions_needed", [])
    return build_sessions_from_chosen_paths(chosen_paths)


@_single_flight_action
def _build_schedule_and_continue(pr_id: str, state: dict):
    """Runs the scheduler; if it finds the student's preferred time window is
    entirely blocked by an existing commitment, pauses to ask them how to
    handle it instead of silently working around it. Otherwise proceeds
    straight to verification and saving the plan."""
    from app.graph import node_schedule_explanation
    from app.nodes import tool_nodes as tn

    student = st.session_state.student
    render_status(status_slot, "🗓️", "Building your schedule...")
    state = tn.node_build_schedule(state)
    status_slot.empty()

    conflict = state["schedule_result"].get("time_conflict")
    if conflict and not state.get("ignore_time_conflict"):
        st.session_state.pending_schedule_conflict = {"pr_id": pr_id, "state": state, "conflict": conflict}
        st.rerun()
        return

    render_status(status_slot, "✅", "Verifying the plan...")
    state = tn.node_verify_plan(state)
    render_status(status_slot, "💬", "Writing up the reasoning...")
    state = node_schedule_explanation(state)
    status_slot.empty()

    sessions = state["schedule_result"]["scheduled"]
    unscheduled = state["schedule_result"].get("unscheduled", [])
    if unscheduled or not state.get("verification_result", {}).get("passed", False):
        details = []
        if unscheduled:
            details.append(f"{len(unscheduled)} selected study session(s) could not fit within your daily limit and available time.")
        if state.get("verification_result", {}).get("user_messages"):
            details.extend(state["verification_result"]["user_messages"])
        friendly = "I couldn't fit the whole plan safely. " + " ".join(details) + " Try extending the deadline, choosing a shorter resource path, or allowing another study day."
        st.error(friendly)
        log_reasoning(friendly)
        return
    for s in sessions:
        student["calendar_events"].append({
            "id": s["id"], "title": f"{s['topic'].title()} · {s['resource_title']}",
            "day": s["day"], "date": s.get("date"), "start": s["start"], "end": s["end"],
            "type": "study_session",
        })
    dynamic_topics = state.get("dynamic_topics") or []
    if dynamic_topics:
        student.setdefault("dynamic_topics", [])
        known_ids = {t["id"] for t in student["dynamic_topics"]}
        student["dynamic_topics"].extend(t for t in dynamic_topics if t["id"] not in known_ids)

    new_pr = {
        "id": pr_id, "topics": state["pr_topics"], "target_mastery": state["pr_target_mastery"],
        "deadline_days": state["pr_deadline_days"], "created_at": datetime.now().strftime("%Y-%m-%d"),
        "status": "active", "current_mastery": {t: student["mastery_map"].get(t, 0) for t in state["pr_topics"]},
        "required_hours": state["feasibility_result"]["required_hours"],
        "scheduled_hours": sum(s["duration_mins"] for s in sessions) / 60,
        "preferred_days": state.get("preferred_days") or [],
        "preferred_time_start": state.get("preferred_time_start"),
        "preferred_time_end": state.get("preferred_time_end"),
        "dynamic_topics": dynamic_topics,
        "sessions": sessions, "daily_updates": [], "replan_count": state.get("_replan_count", 0),
        "study_preferences": state.get("study_preferences", {}),
        "quiz_history": [],
    }
    student["planning_requests"].append(new_pr)
    st.session_state.selected_pr = pr_id
    st.session_state.roadmap_open = False
    log_reasoning(state["schedule_explanation"])
    st.session_state.pending_resource_choice = None
    save_student(student)
    st.toast("Plan created and scheduled!", icon="✅")
    st.rerun()



# ---------------------------------------------------------------------------
# Top-level status slot: shows the live "..retrieving" / "..working on it"
# progress row for any agentic action, wherever it's triggered from.
# ---------------------------------------------------------------------------
status_slot = st.empty()

student = st.session_state.student
prs = student["planning_requests"]
_processing = bool(st.session_state.get("_jassi_processing", False))

# ---------------------------------------------------------------------------
# Sidebar — pure navigation. Just "which plan am I looking at", nothing else.
# Progress numbers used to be repeated here AND in the main content AND in
# the right rail; they now live in exactly one place (the Progress tab).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="jassi-header">
      <div class="jassi-logo">J</div>
      <div>
        <div class="jassi-title">JASSI</div>
        <div class="jassi-subtitle">Agentic learning planner</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    with st.expander("⚙️ Study & routine preferences", expanded=False):
        prefs = student.setdefault("study_preferences", {
            "max_daily_study_hours": 3.0, "wake_start": "07:00", "sleep_start": "23:00",
            "lunch_start": "12:30", "dinner_start": "19:30", "break_minutes": 15,
        })
        max_hours = st.slider("Maximum study hours per day", 1.0, 8.0,
                              float(prefs.get("max_daily_study_hours", 3.0)), 0.5)
        wake_time = st.time_input("Usual wake-up time", value=datetime.strptime(prefs.get("wake_start", "07:00"), "%H:%M").time())
        sleep_time = st.time_input("Usual sleep time", value=datetime.strptime(prefs.get("sleep_start", "23:00"), "%H:%M").time())
        lunch_time = st.time_input("Lunch break starts", value=datetime.strptime(prefs.get("lunch_start", "12:30"), "%H:%M").time())
        dinner_time = st.time_input("Dinner break starts", value=datetime.strptime(prefs.get("dinner_start", "19:30"), "%H:%M").time())
        break_mins = st.selectbox("Break between study sessions", [5, 10, 15, 20, 30],
                                  index=[5, 10, 15, 20, 30].index(int(prefs.get("break_minutes", 15))))
        if st.button("Save routine", use_container_width=True, disabled=_processing):
            prefs.update({"max_daily_study_hours": max_hours, "wake_start": wake_time.strftime("%H:%M"),
                          "sleep_start": sleep_time.strftime("%H:%M"), "lunch_start": lunch_time.strftime("%H:%M"),
                          "dinner_start": dinner_time.strftime("%H:%M"), "break_minutes": break_mins})
            save_student(student)
            st.success("Your routine preferences have been saved.")

    st.markdown("**Your plans**")
    if not prs:
        st.caption("No plans yet — describe a learning goal in the box below to create one.")
    else:
        status_dot = {"active": "🟢", "blocked": "🔴", "completed": "🔵"}
        for pr in prs:
            is_active = st.session_state.selected_pr == pr["id"]
            dot = status_dot.get(pr["status"], "🟢")
            label = f"{dot} {', '.join(pr['topics']).title()}"
            if st.button(label, key=f"nav_{pr['id']}", use_container_width=True,
                         type="primary" if is_active else "secondary", disabled=_processing):
                st.session_state.selected_pr = pr["id"]
                st.session_state.selected_plan_day = None
                st.session_state.roadmap_open = True
                st.rerun()

    st.write("")
    st.caption("Select a plan above to view its day-by-day roadmap. Describe a new learning goal below to start another plan.")


# ---------------------------------------------------------------------------
# Pending day-preference / trade-off / resource-choice steps take over the
# screen until resolved — the one case where interrupting the normal tabbed
# layout is right, since nothing downstream makes sense until the student
# answers. Only one of these is ever active at a time.
# ---------------------------------------------------------------------------
if st.session_state.pending_day_prefs:
    dp = st.session_state.pending_day_prefs
    st.markdown("""
    <div class="card">
      <div style="font-weight:700;font-size:15px;margin-bottom:4px;">📅 Which days work for you?</div>
      <div style="color:var(--text-2);font-size:13px;">
        You didn't mention specific days, so pick the ones you're actually willing to study on —
        JASSI will only schedule sessions within them.
      </div>
    </div>
    """, unsafe_allow_html=True)
    picked = st.multiselect("Preferred study days", DAY_ORDER, default=DAY_ORDER,
                             label_visibility="collapsed")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Confirm days", type="primary", use_container_width=True, disabled=_processing):
            state = dp["state"]
            state["preferred_days"] = picked if len(picked) < 7 else []
            st.session_state.pending_day_prefs = None
            _ask_time_prefs(state)
    with c2:
        if st.button("No preference — any day is fine", use_container_width=True, disabled=_processing):
            state = dp["state"]
            state["preferred_days"] = []
            st.session_state.pending_day_prefs = None
            _ask_time_prefs(state)

elif st.session_state.pending_time_prefs:
    tp = st.session_state.pending_time_prefs
    st.markdown("""
    <div class="card">
      <div style="font-weight:700;font-size:15px;margin-bottom:4px;">⏰ What time of day works best?</div>
      <div style="color:var(--text-2);font-size:13px;">
        JASSI will try to place your sessions in this window, and will always
        route around lunch and dinner and spread sessions out so no single day
        gets overloaded.
      </div>
    </div>
    """, unsafe_allow_html=True)
    time_choice = st.radio(
        "Preferred study time", ["Morning (6–12)", "Afternoon (12–17)", "Evening (17–22)", "No preference"],
        horizontal=True, label_visibility="collapsed",
    )
    time_ranges = {
        "Morning (6–12)": (6 * 60, 12 * 60),
        "Afternoon (12–17)": (12 * 60, 17 * 60),
        "Evening (17–22)": (17 * 60, 22 * 60),
        "No preference": (None, None),
    }
    st.caption("Optional: suggest a course or playlist by name or URL. JASSI will look up a link when you provide only a name (and fall back to a targeted search link if lookup is unavailable). Add one per line. You can also use: Topic ID | title | URL | estimated minutes. Confirm the duration before scheduling.")
    user_refs_text = st.text_area("Your preferred courses / playlists", placeholder="Striver A2Z DSA Course on YouTube\nhttps://www.youtube.com/playlist?list=...\ngraphs | Graph Algorithms | https://example.com/course | 240", key="planning_user_resources")
    if st.button("Confirm timing", type="primary", use_container_width=True, disabled=_processing):
        state = tp["state"]
        start_min, end_min = time_ranges[time_choice]
        state["preferred_time_start"], state["preferred_time_end"] = start_min, end_min
        refs = []
        for line in user_refs_text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = [part.strip() for part in raw.split("|")]
            # Structured form: topic | title | URL | minutes
            if len(parts) == 4 and parts[1]:
                try:
                    minutes = max(15, int(parts[3]))
                except ValueError:
                    minutes = 120
                supplied_url = parts[2] if parts[2].startswith(("https://", "http://")) else ""
                resolved_url, url_source = (supplied_url, "direct") if supplied_url else resolve_resource_url(parts[1])
                refs.append({"topic": parts[0] or "all", "title": parts[1],
                             "url": resolved_url, "url_source": url_source,
                             "duration_mins": minutes, "duration_confirmed": parts[3].isdigit()})
            elif len(parts) == 1:
                # Natural-language course name or URL. Do not silently discard it.
                is_url = raw.startswith(("https://", "http://"))
                title = raw.rstrip("/").split("/")[-1].replace("-", " ") if is_url else raw
                resolved_url, url_source = resolve_resource_url(title or raw)
                refs.append({"topic": "all", "title": title or raw,
                             "url": resolved_url, "url_source": url_source,
                             "duration_mins": 120, "duration_confirmed": False})
        state["reference_resources"] = refs
        st.session_state.pending_time_prefs = None
        _continue_planning_after_days(state)

elif st.session_state.pending_tradeoffs:
    tr = st.session_state.pending_tradeoffs
    fr = tr["result"]["feasibility_result"]
    risks = fr.get("risks") or []
    risk_html = "".join(f'<li style="margin-bottom:4px;">{r}</li>' for r in risks)
    risk_block = (
        f'<ul style="margin:8px 0 0 0;padding-left:18px;color:var(--text-1);font-size:12.5px;">{risk_html}</ul>'
        if risks else ""
    )

    if fr.get("reject"):
        # The shortfall is severe enough that none of the usual nudges (move
        # a soft event, extend by a few days, drop one topic) would actually
        # close the gap — showing them as 3 clickable "fixes" would be
        # misleading, so refuse the plan outright and say what to change.
        st.markdown(f"""
        <div class="card" style="border-color: var(--red);">
          <div style="font-weight:700;font-size:15px;margin-bottom:6px;">🚫 This plan isn't realistic as stated</div>
          <div style="color:var(--text-1);font-size:13.5px;">{tr["result"]["feasibility_explanation"]}</div>
          <div style="margin-top:8px;font-size:12.5px;color:var(--text-2);">
            Required: {fr['required_hours']}h · Available: {fr['available_hours']}h · Short by: {fr['shortfall_hours']}h
            — that's more than double what's actually available, so moving one soft commitment or nudging the
            deadline by a couple of days won't close the gap.
          </div>
          {risk_block}
          <div style="margin-top:10px;font-size:12.5px;color:var(--text-2);">
            Try a meaningfully later deadline, a smaller set of topics, or a lower target mastery, and JASSI
            will re-check feasibility from scratch.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start a new request", use_container_width=True, disabled=_processing):
            st.session_state.pending_tradeoffs = None
            st.rerun()
        st.stop()

    st.markdown(f"""
    <div class="card" style="border-color: var(--yellow);">
      <div style="font-weight:700;font-size:15px;margin-bottom:6px;">⚠️ Reality Check</div>
      <div style="color:var(--text-1);font-size:13.5px;">{tr["result"]["feasibility_explanation"]}</div>
      <div style="margin-top:8px;font-size:12.5px;color:var(--text-2);">
        Required: {fr['required_hours']}h · Available: {fr['available_hours']}h · Shortfall: {fr['shortfall_hours']}h
      </div>
      {risk_block}
    </div>
    """, unsafe_allow_html=True)

    options = tr["result"]["tradeoffs"]["options"]
    cols = st.columns(3)
    for i, opt in enumerate(options):
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="min-height:130px;">
              <b>[{opt['id']}] {opt['title']}</b>
              <div style="color:var(--text-1);font-size:12.5px;margin-top:6px;">{opt['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Choose {opt['id']}", key=f"tradeoff_{opt['id']}", use_container_width=True, disabled=_processing):
                student = st.session_state.student
                state = tr["result"]
                moved_ids = []
                if opt["id"] == "A":
                    moved_ids = [e["id"] for e in student["calendar_events"] if e.get("type") == "soft"]
                elif opt["id"] == "B":
                    ext = feasibility_tool.min_deadline_extension_days(fr["shortfall_hours"])
                    state["pr_deadline_days"] += ext
                elif opt["id"] == "C" and len(state["pr_topics"]) > 1:
                    state["pr_topics"] = state["pr_topics"][:1]

                state["moved_soft_event_ids"] = moved_ids
                state["calendar_events"] = student["calendar_events"]
                state["retriever"] = st.session_state.retriever
                state["existing_sessions"] = []
                # re-run feasibility with the adjusted constraints, then hand off
                # to the same resource-path choice every other plan goes through
                from app.nodes import tool_nodes as tn
                from app.graph import node_generate_resource_paths
                render_status(status_slot, "🧮", "Recalculating feasibility...")
                state = tn.node_calculate_feasibility(state)
                render_status(status_slot, "🔎", "Putting together resource options...")
                state = node_generate_resource_paths(state)
                status_slot.empty()
                log_reasoning(f"Applied trade-off {opt['id']}: {opt['title']}")
                st.session_state.pending_resource_choice = {
                    "pr_id": tr["pr_id"], "state": state, "chosen": {}, "replan_count": 1,
                }
                st.session_state.pending_tradeoffs = None
                st.rerun()

elif st.session_state.pending_resource_choice:
    rc = st.session_state.pending_resource_choice
    options = rc["state"]["resource_path_options"]
    st.markdown("""
    <div class="card">
      <div style="font-weight:700;font-size:15px;margin-bottom:4px;">🧭 Pick your path</div>
      <div style="color:var(--text-2);font-size:13px;">
        Compare the learning paths and choose one for each topic. JASSI checks the actual
        time needed against your work, sleep, meals, breaks and daily study limit after you choose.
      </div>
    </div>
    """, unsafe_allow_html=True)

    for topic_opt in options:
        topic = topic_opt["topic"]
        already_chosen = rc["chosen"].get(topic)
        dyn_tag = ' <span class="dynamic-tag">AI-BUILT TOPIC</span>' if topic_opt.get("dynamic") else ""
        st.markdown(f"**{topic_opt['topic_name']}**{dyn_tag}", unsafe_allow_html=True)
        if already_chosen:
            st.caption(f"✅ Chosen: {already_chosen['title']}")
        else:
            cols = st.columns(max(1, len(topic_opt["paths"])))
            for i, path in enumerate(topic_opt["paths"]):
                with cols[i % len(cols)]:
                    pros = "".join(f"<li>{p}</li>" for p in path.get("pros", []))
                    cons = "".join(f"<li>{c}</li>" for c in path.get("cons", []))
                    total_mins = sum(s.get("duration_mins", 0) for s in path.get("sessions", []))
                    resource_links = "".join(
                        f'<li><a href="{s.get("resource_url", "")}" target="_blank" rel="noopener">{s.get("title") or "Open learning resource"}</a></li>'
                        for s in path.get("sessions", [])
                        if isinstance(s.get("resource_url"), str)
                        and s.get("resource_url", "").startswith(("https://", "http://"))
                    )
                    st.markdown(f"""
                    <div class="path-card">
                      <b>{path['title']}</b>
                      <div style="color:var(--text-2);font-size:11.5px;margin-top:2px;">
                        {len(path.get('sessions', []))} sessions · ~{total_mins // 60}h {total_mins % 60}m total
                      </div>
                      <div class="path-pros-cons"><b>Pros</b><ul>{pros}</ul></div>
                      <div class="path-pros-cons"><b>Cons</b><ul>{cons}</ul></div>
                      {('<div class="path-pros-cons"><b>Your links</b><ul>'+resource_links+'</ul></div>') if resource_links else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    if path.get("user_suggested"):
                        st.info("This option includes the course you suggested. Its duration is an initial estimate; edit the session duration in the path before building your schedule if you know the actual course length.")
                        for session_index, suggested_session in enumerate(path.get("sessions", [])):
                            suggested_session["duration_mins"] = st.number_input(
                                f"Duration in minutes · {suggested_session['title']}",
                                min_value=15, max_value=100000,
                                value=int(suggested_session.get("duration_mins", 120)),
                                step=15, key=f"suggested_duration_{topic}_{i}_{session_index}")
                    if st.button(f"Choose “{path['title']}”", key=f"path_{topic}_{i}", use_container_width=True, disabled=_processing):
                        rc["chosen"][topic] = path
                        st.rerun()
        st.write("")

    all_chosen = len(rc["chosen"]) == len(options)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Build my schedule", type="primary", use_container_width=True, disabled=(not all_chosen) or _processing):
            ordered_chosen = {opt["topic"]: rc["chosen"][opt["topic"]] for opt in options}
            state = rc["state"]
            if rc.get("replan_count"):
                state["_replan_count"] = rc["replan_count"]
            finalize_plan_from_paths(rc["pr_id"], state, ordered_chosen)
    with c2:
        if st.button("Cancel this plan", use_container_width=True, disabled=_processing):
            st.session_state.pending_resource_choice = None
            st.rerun()
    if not all_chosen:
        st.caption("Choose a path for every topic above to continue.")

elif st.session_state.pending_schedule_conflict:
    sc = st.session_state.pending_schedule_conflict
    conflict = sc["conflict"]
    is_soft = conflict["type"] == "soft"
    st.markdown(f"""
    <div class="card" style="border-color: var(--yellow);">
      <div style="font-weight:700;font-size:15px;margin-bottom:6px;">⏰ Your preferred time is fully blocked</div>
      <div style="color:var(--text-1);font-size:13.5px;">
        <b>{conflict['title']}</b> ({conflict['day']}, {conflict['start']}–{conflict['end']})
        {"is a movable commitment that" if is_soft else "is a fixed commitment that"} covers your whole
        preferred study window on that day. Do you want to reschedule it out of the way, or should
        JASSI just work around it and use a different time/day for study sessions instead?
      </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        label = "Reschedule that block" if is_soft else "It's fixed — can't move it"
        if st.button(label, type="primary", use_container_width=True, disabled=(not is_soft) or _processing):
            state = sc["state"]
            state.setdefault("moved_soft_event_ids", [])
            state["moved_soft_event_ids"] = list(set(state["moved_soft_event_ids"] + [conflict["id"]]))
            st.session_state.pending_schedule_conflict = None
            _build_schedule_and_continue(sc["pr_id"], state)
    with c2:
        if st.button("Work around it — schedule elsewhere", use_container_width=True, disabled=_processing):
            state = sc["state"]
            state["ignore_time_conflict"] = True
            st.session_state.pending_schedule_conflict = None
            _build_schedule_and_continue(sc["pr_id"], state)
    st.caption("Note: rescheduling that block only moves it out of THIS scheduling pass — "
               "you can still move it manually anytime from the calendar.")

# ---------------------------------------------------------------------------
# Main content — one section visible at a time via tabs, instead of three
# dense columns fighting for attention. "Today" is the default because
# logging a check-in is the single most frequent thing to do here.
# ---------------------------------------------------------------------------
sel_id = st.session_state.selected_pr
pr = next((p for p in prs if p["id"] == sel_id), None) if sel_id else None

tab_today, tab_timetable, tab_settings = st.tabs(
    ["🏠 Today", "🗓️ Timetable", "⚙️ Settings"]
)

# ---- Plan workspace: plan -> day cards -> day detail ---------------------
with tab_today:
    if not prs:
        st.info("You don't have a plan yet. Describe a learning goal below and JASSI will build one.")
    elif not pr:
        st.markdown("### Your learning plans")
        st.caption("Choose a plan to open its dedicated workspace.")
        for plan in prs:
            with st.container(border=True):
                st.markdown(f"**{', '.join(plan.get('topics', [])).title()}** · {plan.get('status', 'active').title()}")
                st.caption(f"{len(plan.get('sessions', []))} study sessions · {plan.get('deadline_days', '?')} day target")
                if st.button("Open plan", key=f"open_plan_main_{plan['id']}", disabled=_processing, use_container_width=True):
                    st.session_state.selected_pr = plan['id']
                    st.session_state.selected_plan_day = None
                    st.rerun()
    else:
        sessions = pr.get("sessions", [])
        st.markdown(f"### {', '.join(pr.get('topics', [])).title()}")
        st.caption("Your plan workspace · choose a day to see its tasks, learning links, and check-in.")
        top_a, top_b = st.columns([1, 1])
        with top_a:
            if st.button("← All plans", key="back_to_plans", disabled=_processing):
                st.session_state.selected_pr = None
                st.session_state.selected_plan_day = None
                st.rerun()
        with top_b:
            if st.button("🗑️ Delete entire plan", key=f"delete_plan_{pr['id']}", disabled=_processing, use_container_width=True):
                st.session_state.confirm_delete_plan = pr['id']
                st.rerun()

        # Make adaptive decisions inspectable during normal use and demos.
        replan_history = pr.get("replan_history", [])
        recovery_failures = [s for s in sessions if s.get("replan_failure")]
        if replan_history or recovery_failures:
            with st.expander("↻ Roadmap changes & recovery history", expanded=bool(recovery_failures)):
                for entry in reversed(replan_history[-5:]):
                    if entry.get("type") == "whole_roadmap_recovery":
                        st.markdown(
                            f"**Whole-roadmap recovery** · {entry.get('sessions_rescheduled', 0)} unfinished session(s) replanned"
                        )
                        st.caption(entry.get("reason", "Remaining roadmap rebuilt and verified."))
                        st.caption(f"Missed: {', '.join(entry.get('missed_session_ids') or [])} · "
                                   f"Replacement IDs: {', '.join(x for x in (entry.get('replacement_ids') or []) if x)}")
                    else:
                        st.markdown(f"**{entry.get('type', 'Plan update').replace('_', ' ').title()}**")
                        st.caption(f"{entry.get('from_date', '—')} → {entry.get('to_date', '—')} · {entry.get('reason', '')}")
                for failed in recovery_failures:
                    st.error(f"Recovery needs attention for {failed.get('topic', 'a session')}: {failed['replan_failure']}")

        if st.session_state.get("confirm_delete_plan") == pr['id']:
            st.warning("Delete this plan and all of its study events? This cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Yes, delete plan", type="primary", key="confirm_delete_plan_yes"):
                    session_ids = {str(item.get("id")) for item in pr.get("sessions", [])}
                    student["calendar_events"] = [ev for ev in student.get("calendar_events", []) if str(ev.get("id")) not in session_ids]
                    student["planning_requests"] = [item for item in student.get("planning_requests", []) if item.get("id") != pr['id']]
                    st.session_state.selected_pr = None
                    st.session_state.selected_plan_day = None
                    st.session_state.confirm_delete_plan = None
                    save_student(student)
                    st.toast("Plan and all its timetable events deleted", icon="🗑️")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key="confirm_delete_plan_no"):
                    st.session_state.confirm_delete_plan = None
                    st.rerun()

        # Group only dates that actually contain sessions; no empty day cards.
        day_groups = {}
        for sess in sorted(sessions, key=lambda item: (item.get("date") or "9999-99-99", item.get("start", "00:00"))):
            raw_date = sess.get("date")
            try:
                day_date = datetime.strptime(raw_date, "%Y-%m-%d").date() if raw_date else None
            except (TypeError, ValueError):
                day_date = None
            key = day_date.isoformat() if day_date else sess.get("day", "Unscheduled")
            day_groups.setdefault(key, {"date": day_date, "sessions": []})["sessions"].append(sess)

        selected_day = st.session_state.get("selected_plan_day")
        if selected_day not in day_groups:
            st.markdown("#### Your days")
            if not day_groups:
                st.info("This plan has no scheduled study days yet.")
            else:
                cols = st.columns(2)
                for idx, (day_key, info) in enumerate(day_groups.items()):
                    day_sessions = info["sessions"]
                    done = sum(1 for item in day_sessions if item.get("status") == "completed")
                    date_label = info["date"].strftime("%A, %d %b %Y") if info["date"] else str(day_key)
                    with cols[idx % 2]:
                        with st.container(border=True):
                            st.markdown(f"**🗓️ {date_label}**")
                            st.caption(f"{len(day_sessions)} task(s) · {done}/{len(day_sessions)} completed")
                            for item in day_sessions[:3]:
                                st.write(f"• {item.get('resource_title') or item.get('topic', 'Study task').replace('_', ' ').title()}")
                            if len(day_sessions) > 3:
                                st.caption(f"+ {len(day_sessions)-3} more tasks")
                            if st.button("Open day →", key=f"open_day_{pr['id']}_{day_key}", disabled=_processing, use_container_width=True):
                                st.session_state.selected_plan_day = day_key
                                st.rerun()
        else:
            day_info = day_groups[selected_day]
            date_label = day_info["date"].strftime("%A, %d %b %Y") if day_info["date"] else str(selected_day)
            if st.button("← Back to days", key="back_to_days", disabled=_processing):
                st.session_state.selected_plan_day = None
                st.rerun()
            st.markdown(f"#### {date_label}")
            for sess in day_info["sessions"]:
                with st.container(border=True):
                    task_title = sess.get("resource_title") or sess.get("topic", "Study task").replace("_", " ").title()
                    st.markdown(f"**{task_title}**")
                    st.caption(f"{sess.get('start', 'Time TBD')} · {sess.get('duration_mins', 0)} min · {sess.get('status', 'scheduled').replace('_', ' ').title()}")
                    st.write(sess.get("task_description") or sess.get("description") or f"Work through {sess.get('topic', 'the selected topic').replace('_', ' ')} and practise the key ideas.")
                    links = sess.get("resources") or []
                    if links:
                        st.caption("Relevant learning resources")
                        for n, resource in enumerate(links[:3], 1):
                            url = resource.get("url", "")
                            title = resource.get("title") or f"Resource {n}"
                            if isinstance(url, str) and url.startswith(("https://", "http://")):
                                st.markdown(f"{n}. [{title}]({url})")
                    elif isinstance(sess.get("resource_url"), str) and sess.get("resource_url", "").startswith(("https://", "http://")):
                        st.markdown(f"[Open learning resource]({sess['resource_url']})")
                    else:
                        st.caption("No verified resource found yet. Use the topic-specific YouTube search below rather than an unrelated video.")
                        from urllib.parse import quote_plus
                        search_text = f"{sess.get('topic', '')} {task_title} tutorial"
                        st.markdown(f"[Search YouTube for this task](https://www.youtube.com/results?search_query={quote_plus(search_text)})")

                    if sess.get("status", "scheduled") != "completed":
                        with st.form(f"day_update_{sess['id']}"):
                            update_text = st.text_area("Your progress update", placeholder="What did you finish? What was difficult? What will you do next?", key=f"day_update_text_{sess['id']}")
                            took_quiz = st.checkbox("I took a quiz or practice test", key=f"day_quiz_taken_{sess['id']}")
                            quiz_score = st.number_input("Score (%)", min_value=0, max_value=100, value=0, step=1, disabled=not took_quiz, key=f"day_quiz_score_{sess['id']}")
                            submitted = st.form_submit_button("Send update", type="primary", disabled=_processing)
                        if submitted and update_text.strip():
                            client = get_client()
                            graph = build_daily_update_graph()
                            state = {
                                "client": client, "topics_path": TOPICS_PATH, "raw_text": update_text,
                                "session": sess, "quiz_score": int(quiz_score) if took_quiz else None,
                                "update_topic": sess["topic"], "current_mastery": pr["current_mastery"],
                                "calendar_events": student["calendar_events"],
                                "existing_sessions": pr["sessions"], "quiz_history": pr.get("quiz_history", []),
                            }
                            result = stream_graph_with_status(graph, state, UPDATE_STEP_LABELS, status_slot)
                            parsed = result["parsed_update"]
                            for existing in pr["sessions"]:
                                if existing["id"] == sess["id"]:
                                    existing["status"] = parsed["status"]
                                    if parsed["status"] == "partial":
                                        # Partial work uses the smaller catch-up task from
                                        # the check-in graph; don't process it as a full miss.
                                        existing["replan_processed_at"] = datetime.now().isoformat(timespec="seconds")
                            pr.setdefault("daily_updates", []).append({
                                "session_id": sess["id"], "status": parsed["status"], "text": update_text,
                                "percent_done": parsed.get("percent_done"), "struggle_note": parsed.get("struggle_note"),
                                "mastery_delta": result.get("mastery_delta"), "mastery_source": result.get("mastery_update_source"),
                                "time": datetime.now().isoformat(),
                            })
                            if took_quiz:
                                pr.setdefault("quiz_history", []).append({"topic": sess["topic"], "score": int(quiz_score)})
                            if parsed.get("status") == "missed":
                                # A missed check-in is a forced recovery trigger even
                                # when its scheduled date is today. Rebuild the whole
                                # plan, not just this one session.
                                roadmap_changes = reconcile_overdue_sessions(
                                    student, today=date.today(), force_session_ids={sess["id"]}
                                )
                                recovered = sum(1 for item in roadmap_changes if item.get("replacement"))
                                if recovered:
                                    st.toast(f"JASSI rebuilt the remaining roadmap and rescheduled {recovered} missed session(s).", icon="🗓️")
                                else:
                                    st.warning("JASSI marked the session as missed, but could not safely rebuild the remaining roadmap. Review the recovery reason in the plan history.")
                            elif result.get("rescheduled_session"):
                                new_sess = result["rescheduled_session"]
                                # A replacement is a distinct session, not a second
                                # record with the original session's ID.
                                import uuid as _uuid
                                old_session = next((item for item in pr["sessions"] if item["id"] == sess["id"]), None)
                                new_sess["id"] = f"sess_{_uuid.uuid4().hex[:10]}"
                                new_sess["rescheduled_from"] = sess["id"]
                                new_sess["original_date"] = sess.get("date")
                                if old_session is not None:
                                    old_session["replanned_to"] = new_sess["id"]
                                    old_session["replan_processed_at"] = datetime.now().isoformat(timespec="seconds")
                                pr["sessions"].append(new_sess)
                                student["calendar_events"] = [ev for ev in student["calendar_events"] if str(ev.get("id")) != str(sess["id"])]
                                student["calendar_events"].append({"id": new_sess["id"], "title": f"{new_sess['topic'].title()} · {new_sess.get('resource_title','')}", "day": new_sess["day"], "date": new_sess.get("date"), "start": new_sess["start"], "end": new_sess["end"], "type": "study_session"})
                                pr.setdefault("replan_history", []).append({"type": "checkin_recovery", "session_id": sess["id"], "replacement_id": new_sess["id"], "from_date": sess.get("date"), "to_date": new_sess.get("date"), "at": datetime.now().isoformat(timespec="seconds")})
                            log_reasoning(result.get("progress_insight", "Progress updated."))
                            save_student(student)
                            st.success("Progress saved!")
                            st.rerun()

# ---- Timetable: full width, nothing competing for attention -------------
with tab_timetable:
    view = st.radio("View", ["This week", "Month (full trajectory)"], horizontal=True, label_visibility="collapsed")
    events = st.session_state.student["calendar_events"]
    if view == "This week":
        # A dated event (e.g. a study session scheduled for a specific future
        # Wednesday several weeks out) only belongs in THIS week if its date
        # actually falls in it — otherwise every dated session sharing that
        # weekday name piles into one column regardless of which week it's
        # really on. Recurring hard/soft events (no "date") always repeat.
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        this_week_events = [
            ev for ev in events
            if not ev.get("date") or (week_start.isoformat() <= ev["date"] <= week_end.isoformat())
        ]
        events_by_day = {d: [] for d in DAY_ORDER}
        for ev in this_week_events:
            events_by_day.setdefault(ev["day"], []).append(ev)
        for d in events_by_day:
            events_by_day[d].sort(key=lambda e: e["start"])
        st.caption(f"Week of {week_start.strftime('%b %d')} – {week_end.strftime('%b %d')} · click any event to edit or delete it")
        render_editable_timetable_grid(events_by_day, DAY_ORDER)
    else:
        # Deadlines can run for weeks, so this shows the whole window at a
        # glance instead of only ever showing "this week" on repeat.
        max_deadline = max((p["deadline_days"] for p in prs), default=7)
        num_weeks = max(4, -(-max_deadline // 7) + 1)
        st.markdown(render_month_calendar_html(events, date.today(), num_weeks=num_weeks),
                    unsafe_allow_html=True)
    st.caption("🔴 Fixed commitments · 🟡 Movable/soft commitments · 🟢 Study sessions")

# ---- Progress dashboard intentionally left out for now (not wired up
# reliably enough to show live — see request to hide it until it's ready).

# ---- Settings: importing constraints — not something you need on screen
# every time, so it's tucked away instead of permanently occupying a column.
with tab_settings:
    st.markdown('<div class="card"><div class="metric-label">Import Constraints</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Manual", "Image", "Google Cal"])
    with tab1:
        with st.form("manual_event_form", clear_on_submit=True):
            title = st.text_input("Title")
            day = st.selectbox("Day", DAY_ORDER)
            c1, c2 = st.columns(2)
            start = c1.time_input("Start")
            end = c2.time_input("End")
            etype = st.radio("Type", ["hard", "soft"], horizontal=True)
            if st.form_submit_button("Add event", use_container_width=True, disabled=_processing):
                existing = st.session_state.student["calendar_events"]
                resolved = resolve_event_time_conflict(
                    existing, day, start.strftime("%H:%M"), end.strftime("%H:%M")
                )
                if not resolved["fits"]:
                    st.error(
                        f"'{title}' overlaps with {', '.join(resolved['bumped'])} on {day}, and there's no "
                        f"clear gap of that length later in the day. Try a different day, an earlier start, "
                        f"or a shorter duration."
                    )
                else:
                    ev = manual_event(title, day, resolved["start"], resolved["end"], etype)
                    st.session_state.student["calendar_events"].append(ev)
                    save_student(st.session_state.student)
                    if resolved["moved"]:
                        st.warning(
                            f"'{title}' overlapped with {', '.join(resolved['bumped'])}, so it's been placed "
                            f"at {resolved['start']}–{resolved['end']} instead, right after it ends."
                        )
                    else:
                        st.success(f"Added {title} ({resolved['start']}–{resolved['end']})")
                    st.rerun()
    with tab2:
        img_file = st.file_uploader("Upload timetable screenshot", type=["png", "jpg", "jpeg"])
        if img_file and st.button("Parse with AI", use_container_width=True, disabled=_processing):
            client = get_client()
            render_status(status_slot, "🖼️", "Reading your timetable...")
            try:
                events = parse_image_with_llm(client, img_file.read())
                status_slot.empty()
                st.session_state.student["calendar_events"].extend(events)
                save_student(st.session_state.student)
                st.success(f"Imported {len(events)} events")
                st.rerun()
            except RuntimeError as e:
                status_slot.empty()
                st.error(str(e))
    with tab3:
        st.caption("Connect Google Calendar to import this week's events.")
        if st.button("Connect Google Calendar", use_container_width=True, disabled=_processing):
            st.info("OAuth flow not wired up in this demo build — set GOOGLE_OAUTH_CLIENT_ID/SECRET "
                    "in .env and implement the redirect in app/tools/constraint_parser.py "
                    "(parse_google_calendar_events is ready to receive the fetched events).")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("To edit or delete an existing event, click it directly on the Timetable tab's calendar.")

st.markdown("""
<div style="text-align:center;color:var(--text-2);font-size:11px;margin-top:24px;padding:14px;">
  JASSI · Agentic AI Learning Planner · Powered by Groq + LangGraph + FAISS · Build 2026-10-31
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bottom-center planning request bar (Claude-style docked chat input)
# ---------------------------------------------------------------------------
_awaiting_answer = bool(
    st.session_state.pending_day_prefs or st.session_state.pending_time_prefs
    or st.session_state.pending_tradeoffs or st.session_state.pending_resource_choice
    or st.session_state.pending_schedule_conflict
)
_chat_placeholder = (
    "Answer the question above before starting a new plan..." if _awaiting_answer
    else "Describe your learning goal — e.g. \"I wanna finish trees and graphs in 10 days\""
)
new_goal_text = st.chat_input(_chat_placeholder, disabled=(_awaiting_answer or _processing))
if new_goal_text and new_goal_text.strip() and not _awaiting_answer:
    create_planning_request_flow(new_goal_text.strip())
