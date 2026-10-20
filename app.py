"""Streamlit UI for the AI Career Path & Skill-Gap Recommender — Fancy redesign.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.matcher import (
    build_career_path,
    embedder_status,
    match_jobs,
    recommend_courses,
)
from src.resume_parser import extract_text_from_pdf
from src.skill_extractor import extract_skills, skills_as_names

SAMPLE_PDF = Path(__file__).parent / "sample" / "sample_resume.pdf"

# --------------------------------------------------------------------------- #
# Custom CSS — design tokens & component styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Career Path & Skill-Gap Recommender",
    page_icon="",
    layout="wide",
)

_CSS = """
:root {
  --brand-primary: #4F46E5;       /* indigo-600 */
  --brand-on-primary: #FFFFFF;
  --surface-sidebar: #F1F5F9;     /* slate-100 */
  --surface-card: #FFFFFF;
  --border-default: #CBD5E1;       /* slate-300 */
  --text-on-primary: #FFFFFF;
  --success: #10B981;             /* green-500 */
  --warning: #F59E0B;             /* amber-500 */
  --error: #EF4444;               /* red-500 */
}

[data-testid="stSidebar"] {
  background: var(--surface-sidebar);
  border-right: 1px solid var(--border-default);
}

[data-testid="stMetricValue"] {
  font-size: 36px !important;
  font-weight: 700 !important;
  color: var(--brand-primary) !important;
}

[data-testid="stMetricLabel"] {
  font-size: 14px !important;
  font-weight: 500 !important;
  color: #64748B !important;       /* slate-500 */
}

.main .block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
}

/* Question-mark circle badges on metrics */
.metric-qmark {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #F1F5F9;
  border: 1px solid var(--border-default);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  margin-right: 4px;
  vertical-align: middle;
}

/* Purple underline on active tab */
[data-testid="stTab"] {
  border-radius: 4px 4px 0 0;
  gap: 0;
  padding-top: 6px;
  padding-bottom: 6px;
}

/* Active tab gets solid purple underline */
[data-testid="stTab"][aria-selected="true"] {
  background: var(--surface-card);
  border-bottom: 3px solid var(--brand-primary);
}

/* Inactive tabs */
[data-testid="stTab"] > div:first-child {
  color: #64748B;
}

/* Skill-badge pills (custom class avoids Streamlit's native .stMarkdownBadge
   rule, which forces overflow:hidden + text-overflow:ellipsis and truncates
   long skill names inside narrow grid columns) */
/* Pills flow in one wrapping flex row so each sizes to its own text and
   never breaks mid-word inside a narrow column */
.pill-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: flex-start;
}

.skill-pill {
  display: inline-block;
  background: rgba(79, 70, 229, 0.15);
  color: var(--brand-primary);
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
  padding: 4px 11px;
  line-height: 1.45;
  white-space: nowrap;
  overflow: visible;
  text-overflow: clip;
}

/* Metric cards (HTML, so long values wrap instead of being ellipsised) */
.metric-card {
  background: var(--surface-card);
  border: 1px solid var(--border-default);
  border-radius: 10px;
  padding: 14px 16px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 8px;
}

.metric-label {
  font-size: 12px;
  font-weight: 600;
  color: #64748B;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  line-height: 1.35;
  overflow-wrap: break-word;
}

.metric-value {
  font-size: 32px;
  font-weight: 700;
  color: var(--brand-primary);
  line-height: 1.1;
}

.metric-value-text {
  font-size: 17px;
  font-weight: 700;
  color: var(--brand-primary);
  line-height: 1.3;
  overflow-wrap: break-word;
  hyphens: auto;
}

.metric-sub {
  font-size: 12px;
  color: #64748B;
  line-height: 1.35;
  overflow-wrap: break-word;
}

/* Primary call-to-action button */
.stButton > button[kind="primary"] {
  background: var(--brand-primary) !important;
  color: var(--text-on-primary) !important;
  border-radius: 24px !important;
  font-weight: 600 !important;
  border: none !important;
}

/* Sidebar button standout */
.sidebar-cta button {
  width: 100%;
  background: var(--brand-primary) !important;
  color: var(--text-on-primary) !important;
  border-radius: 24px !important;
  font-weight: 600 !important;
  border: none !important;
  margin-top: 1rem;
}

/* Job & career card */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border-default) !important;
  border-radius: 8px !important;
  background: var(--surface-card) !important;
}

/* Progress bar track */
[data-testid="stProgressBarTrack"] {
  background: rgba(79, 70, 229, 0.2) !important;
  border-radius: 4px !important;
}

/* Expandable evidence box */
.evidence-box {
  border: 1px solid var(--border-default);
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 8px;
  background: var(--surface-card);
  font-size: 0.875rem;
}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def analyze(pdf_bytes: bytes) -> dict:
    """Full pipeline: PDF -> text -> skills -> job matches -> career path."""
    t0 = time.perf_counter()
    text = extract_text_from_pdf(pdf_bytes)
    skills = extract_skills(text)
    skill_names = skills_as_names(skills)
    t1 = time.perf_counter()
    results = match_jobs(text, skill_names)
    path = build_career_path(results)
    t2 = time.perf_counter()
    print(
        f"[analyze] parse+skills {t1 - t0:.2f}s | match+path {t2 - t1:.2f}s "
        f"| embedder={embedder_status()} | skills={len(skill_names)} "
        f"jobs={len(results)}",
        flush=True,
    )
    return {
        "text": text,
        "skills": skills,
        "skill_names": skill_names,
        "results": results,
        "career_path": path,
    }


# --------------------------------------------------------------------------- #
# Sidebar — light gray, numbered steps, primary CTA button
# --------------------------------------------------------------------------- #

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

with st.sidebar:
    st.title("AI Career Advisor")

    st.markdown(
        """
**How it works (the 10/10 architecture)**

1. **Parse** - extracts text from your resume PDF
2. **NLP (spaCy)** - finds your skills in unstructured text
3. **Semantic matching** - the all-MiniLM-L6-v2 sentence model
   (ONNX Runtime, CPU) ranks job postings by fit
4. **Skill-gap analysis** - diffs your skills vs. each job
5. **Recommendations** - picks courses that close the most
   important gaps and orders them into a learning path
        """
    )

    st.markdown("**Upload your resume (PDF)**")
    uploaded = st.file_uploader(
        "Upload your resume (PDF)", type=["pdf"], label_visibility="collapsed"
    )
    st.markdown('<div class="sidebar-cta">', unsafe_allow_html=True)
    use_sample = st.button(
        "Or use the sample resume", type="primary", use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pdf_bytes is not None and st.button("Clear resume"):
        st.session_state.pdf_bytes = None
        st.rerun()

# --------------------------------------------------------------------------- #
# Resolve PDF bytes — kept in session state so the analysis survives reruns
# (a Streamlit button is only True for a single run, so without this the
# results would disappear the moment any other widget is touched).
# --------------------------------------------------------------------------- #

try:
    if uploaded is not None:
        st.session_state.pdf_bytes = uploaded.read()
    elif use_sample:
        st.session_state.pdf_bytes = SAMPLE_PDF.read_bytes()
except Exception as exc:
    st.error(f"Could not read that resume: {exc}")
    st.stop()

pdf_bytes = st.session_state.pdf_bytes

# --------------------------------------------------------------------------- #
# Early exit — nothing selected yet
# --------------------------------------------------------------------------- #

if pdf_bytes is None:
    st.title("AI Career Path & Skill-Gap Recommender")
    st.caption(
        "A resume-driven career matcher: parse → NLP skills → semantic job fit "
        "→ skill gaps → learning path."
    )
    st.markdown(
        "Upload a **resume PDF** in the sidebar (or click **Or use the sample resume**) to get:\n\n"
        "- Your extracted **skill profile**\n"
        "- **Top matched jobs** ranked by semantic fit\n"
        "- The **skill gap** for each job\n"
        "- **Courses** and an ordered **career path** to close those gaps"
    )
    st.stop()

try:
    with st.spinner("Running NLP pipeline and semantic matching..."):
        data = analyze(pdf_bytes)
except Exception as exc:
    st.error(f"Could not analyze this resume: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Header title (filled by set_page_config + manual tweak)
# --------------------------------------------------------------------------- #

st.title("AI Career Path & Skill-Gap Recommender")

# --------------------------------------------------------------------------- #
# Metric cards row (4 columns) with question-mark icons
# --------------------------------------------------------------------------- #

top = data["results"][:3]
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Skills extracted</div>'
        f'<div class="metric-value">{len(data["skill_names"])}</div>'
        '<div class="metric-sub">from your resume</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Jobs analyzed</div>'
        f'<div class="metric-value">{len(data["results"])}</div>'
        '<div class="metric-sub">open roles matched</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with c3:
    best_job = data["results"][0]["job"]
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Best match</div>'
        f'<div class="metric-value-text">{best_job["title"]}</div>'
        f'<div class="metric-sub">{best_job["company"]}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with c4:
    best_score = data["results"][0]["score"]
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Best fit</div>'
        f'<div class="metric-value">{best_score:.0%}</div>'
        '<div class="metric-sub">50% semantic + 50% skills</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------------------------------- #
# Tabs: Skill Profile | Job Matches | Career Path
# --------------------------------------------------------------------------- #

tab_skills, tab_jobs, tab_path = st.tabs(
    ["Skill Profile", "Job Matches", "Career Path"]
)

# --- Tab 1: Skill Profile ---
with tab_skills:
    st.subheader(f"Extracted skills ({len(data['skill_names'])})")

    # 4-column grid of skill-badge pills
    pills = "".join(
        f'<span class="skill-pill">{s.skill} &middot; x{s.count}</span>'
        for s in data["skills"]
    )
    st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)

    with st.expander("See evidence snippets from your resume"):
        for skill in data["skills"]:
            st.markdown(f"**{skill.skill}** — *{skill.evidence}*")

# --- Tab 2: Job Matches ---
with tab_jobs:
    st.subheader("Semantic job matching")
    st.caption(
        "Score = 50% embedding similarity between your resume and the job "
        "postings + 50% explicit skill coverage."
    )

    for result in data["results"]:
        job = result["job"]
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"### {job['title']}")
            left.caption(f"{job['company']} - {job['location']}")
            right.metric("Fit", f"{result['score']:.0%}")

            st.progress(result["score"], text="overall fit: {:.0%}".format(result["score"]))

            g1, g2 = st.columns(2)
            g1.markdown(f"**Matched skills:** {', '.join(result['matched_skills']) or 'none'}")
            g2.markdown(
                f"**Skill gap:** {', '.join(result['missing_skills']) or 'none - you are fully qualified!'}"
            )

            courses = recommend_courses(result["missing_skills"], top_k=3)
            if courses:
                st.markdown("**Recommended courses to close this gap:**")
                for course in courses:
                    st.markdown(
                        f"- [{course['title']}]({course['url']}) — "
                        f"{course['provider']} ({course['level']}, {course['duration_weeks']}w) "
                        f"covers *{', '.join(course['covered_skills'])}*"
                    )

# --- Tab 3: Career Path ---
with tab_path:
    st.subheader("Your personalized career path")

    if not data["career_path"]:
        st.success(
            "Your skills cover the top matched jobs - no critical gaps found. "
            "Try a resume with fewer skills to see the path in action."
        )
    else:
        st.caption(
            "Courses are chosen greedily to cover the most gap skills across "
            "your top matched jobs, then ordered beginner-first."
        )

        for course in data["career_path"]:
            with st.container(border=True):
                level = course["level"]
                level_color = (
                    "#10B981"
                    if level == "Beginner"
                    else ("#F59E0B" if level == "Intermediate" else "#EF4444")
                )
                st.markdown(
                    f"**Step {course['step']}: [{course['title']}]({course['url']})**",
                    help=f"Level: {course['level']}",
                )
                st.caption(
                    f"{course['provider']} - {course['level']} - {course['duration_weeks']} weeks"
                )
                covered = ", ".join(course["covered_skills"])
                st.markdown(
                    f'<span style="background:{level_color}20;color:{level_color};'
                    f'border-radius:4px;padding:2px 6px;font-size:0.75rem;">'
                    f'covers {covered}</span>',
                    unsafe_allow_html=True,
                )

        total_weeks = sum(c["duration_weeks"] for c in data["career_path"])
        st.info(f"Estimated total time: ~{total_weeks} weeks of focused study.")



# --------------------------------------------------------------------------- #
# Footer / top-control hints (optional, visible on deployed share)
# --------------------------------------------------------------------------- #

st.caption(
    "Built with Streamlit • ONNX Runtime (CPU) • spaCy • Open-source career matcher"
)