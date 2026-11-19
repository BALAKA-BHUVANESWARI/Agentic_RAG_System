"""Visual identity for the Streamlit UI: injected CSS and small HTML builders.

Design idea: evidence first. Every answer is a "ruling" with a coloured left rule
(cobalt = company documents, amber = public web, slate = no reliable answer) and
every trace starts with a route line built from the graph nodes that really ran.

Only fixed strings and integers are put into HTML here. Text that comes from the
model, documents or the web is never interpolated into HTML (it goes through
``st.markdown`` without ``unsafe_allow_html``).
"""
from __future__ import annotations

from collections import Counter
from html import escape

import streamlit as st

# Station label for every graph node, in graph order.
STATIONS: list[tuple[str, str]] = [
    ("query_classifier", "Classify"),
    ("private_retriever", "Company docs"),
    ("private_evidence_grader", "Grade"),
    ("query_rewriter", "Rewrite"),
    ("external_search", "Public web"),
    ("external_evidence_grader", "Grade web"),
    ("answer_generator", "Answer"),
    ("final_answer_validator", "Validate"),
    ("audit_logger", "Log"),
]

STARTERS = [
    "How do I connect to the company VPN from home?",
    "What are the password rules and how often must I change mine?",
    "How do I set up MFA on a new phone?",
    "What is the latest Microsoft Teams outage guidance?",
]

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--paper:#EDF0EA;--panel:#F8FAF6;--ink:#141B3D;--cobalt:#2C3FD1;--amber:#9A5B00;--amber-bg:#FFF1CF;--slate:#5A6275;--green:#17794A;--line:#CBD2C6;}
html,body,.stApp,.stApp p,.stApp label,.stApp li,.stApp input,.stApp textarea,.stApp button,.stApp [data-baseweb]{font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;}
/* Streamlit draws its icons as font ligatures: never replace the icon font. */
[data-testid="stIconMaterial"],.material-symbols-rounded,.material-icons{font-family:'Material Symbols Rounded','Material Icons'!important;}
h1,h2,h3,[data-testid="stHeading"] *{font-family:'Bricolage Grotesque','IBM Plex Sans',system-ui,sans-serif!important;letter-spacing:-0.01em;color:var(--ink);}
code,pre,kbd{font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace!important;font-size:.86em;}
.block-container{padding-top:2.2rem;max-width:62rem;}
[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
.wordmark{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.35rem;line-height:1.1;color:var(--ink);margin:.2rem 0 .1rem;}
.wordmark i{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;background:var(--cobalt);margin-right:.5rem;box-shadow:.9rem 0 0 -.05rem var(--amber);}
.tagline{color:var(--slate);font-size:.82rem;margin:0 0 .8rem 0;}
.lede{color:var(--slate);max-width:44rem;margin-bottom:1.2rem;}
.verdict{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.05rem;margin:0 0 .35rem 0;}
.verdict.company{color:var(--cobalt)} .verdict.public{color:var(--amber)} .verdict.none{color:var(--slate)}
.verdict small{font-family:'IBM Plex Sans',sans-serif;font-weight:500;color:var(--green);margin-left:.6rem;font-size:.8rem;}
[data-testid="stChatMessage"]{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:1rem 1.2rem;}
[data-testid="stChatMessage"]:has(.verdict.company){border-left:5px solid var(--cobalt);}
[data-testid="stChatMessage"]:has(.verdict.public){border-left:5px solid var(--amber);}
[data-testid="stChatMessage"]:has(.verdict.none){border-left:5px solid var(--slate);}
[data-testid="stAlert"]{border-radius:4px;}
.route{display:flex;flex-wrap:wrap;gap:.2rem 0;margin:.4rem 0 1rem 0;padding:0;}
.route .st{position:relative;flex:1 1 0;min-width:5.2rem;text-align:center;font-size:.74rem;color:var(--slate);padding-top:1.35rem;}
.route .st::before{content:"";position:absolute;top:.42rem;left:0;right:0;height:2px;background:var(--line);}
.route .st::after{content:"";position:absolute;top:0;left:calc(50% - .4rem);width:.8rem;height:.8rem;border-radius:50%;background:var(--paper);border:2px solid var(--line);}
.route .st.ran{color:var(--ink);font-weight:600;}
.route .st.ran::before{background:var(--cobalt);transform-origin:left;animation:draw .6s ease-out both;}
.route .st.ran::after{background:var(--cobalt);border-color:var(--cobalt);}
.route .st.web.ran::before{background:var(--amber)} .route .st.web.ran::after{background:var(--amber);border-color:var(--amber);}
.route .st .x{font-family:'IBM Plex Mono',monospace;font-weight:500;margin-left:.2rem;color:var(--cobalt);}
@keyframes draw{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@media (prefers-reduced-motion:reduce){.route .st.ran::before{animation:none}}
[data-testid="stMetric"]{background:transparent;border-top:2px solid var(--ink);padding:.45rem 0 0 0;}
[data-testid="stMetricLabel"]{color:var(--slate);}
[data-testid="stMetricValue"]{font-family:'Bricolage Grotesque',sans-serif;font-size:1.6rem;line-height:1.2;}
[data-testid="stMetricValue"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:anywhere;}
.stButton>button{border-radius:4px;border:1.5px solid var(--ink);background:transparent;color:var(--ink);font-weight:500;}
.stButton>button:hover{background:var(--ink);color:var(--paper);border-color:var(--ink);}
.stButton>button:focus-visible,input:focus-visible{outline:3px solid var(--cobalt);outline-offset:2px;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:4px;background:var(--panel);}
[data-testid="stTabs"] button[role="tab"]{font-family:'Bricolage Grotesque',sans-serif;font-size:1rem;}
.starters-title{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.5rem;margin:.6rem 0 .2rem 0;}
"""


def inject_theme() -> None:
    """Add the app's fonts and styles to the page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def wordmark() -> str:
    """Sidebar brand block."""
    return '<div class="wordmark"><i></i>IT Support Copilot</div><div class="tagline">Answers with receipts</div>'


def verdict_html(abstained: bool, used_external: bool, validated: bool | None) -> str:
    """One plain line saying what kind of answer this is. Built from fixed text only."""
    if abstained:
        kind, text = "none", "No reliable answer found"
    elif used_external:
        kind, text = "public", "Answer uses public web sources"
    else:
        kind, text = "company", "Answer from company documents"
    check = "<small>checked against sources</small>" if validated and not abstained else ""
    return f'<div class="verdict {kind}">{text}{check}</div>'


def route_html(path: list[str]) -> str:
    """Route line from the nodes that really ran. Repeats show as a count (e.g. x2)."""
    counts = Counter(path)
    cells = []
    for node, label in STATIONS:
        n = counts.get(node, 0)
        cls = "st ran" if n else "st"
        if node.startswith("external"):
            cls += " web"
        times = f'<span class="x">×{n}</span>' if n > 1 else ""
        cells.append(f'<div class="{cls}">{label}{times}</div>')
    return f'<div class="route" role="img" aria-label="Nodes that ran: {escape(", ".join(path), quote=True)}">{"".join(cells)}</div>'
