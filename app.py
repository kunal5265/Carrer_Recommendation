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

/* Skill-badge pills */
.stMarkdownBadge {
  background: rgba(79, 70, 229, 0.15) !important;
  color: var(--brand-primary) !important;
  border-radius: 20px !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  padding: 2px 8px !important;
  margin: 2px !important;
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

st.markdown(_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# App config
# --------------------------------------------------------------------------- #

st.title(" AI Career Path & Skill-Gap Recommender")  # filled in later

# --------------------------------------------------------------------------- #
# Sidebar — light gray, numbered steps, primary CTA button
# --------------------------------------------------------------------------- #

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
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"], label_visibility="collapsed")
    st.markdown('<div class="sidebar-cta">', unsafe_allow_html=True)
    use_sample = st.button("Or use the sample resume", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Early exit — nothing selected yet
# --------------------------------------------------------------------------- #

if not uploaded and not use_sample:
    st.title("AI Career Path & Skill-Gap Recommender")
    st.markdown(
        "Upload a **resume PDF** in the sidebar (or click **Or use the sample resume**) to get:\n\n"
        "- Your extracted **skill profile**\n"
        "- **Top matched jobs** ranked by semantic fit\n"
        "- The **skill gap** for each job\n"
        "- **Courses** and an ordered **career path** to close those gaps"
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Resolve PDF bytes
# --------------------------------------------------------------------------- #

try:
    if uploaded is not None:
        pdf_bytes = uploaded.read()
    else:
        pdf_bytes = SAMPLE_PDF.read_bytes()
except Exception as exc:
    st.error(f"Could not analyze this resume: {exc}")
    st.stop()

with st.spinner("Running NLP pipeline and semantic matching..."):
    data = analyze(pdf_bytes)

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
    st.metric(
        "**Skills extracted**",
        len(data["skill_names"]),
    )

with c2:
    st.metric("**Jobs analyzed**", len(data["results"]))

with c3:
    best_title = data["results"][0]["job"]["title"][:40] + ("..." if len(data["results"][0]["job"]["title"]) > 40 else "")
    st.metric("**Best match**", best_title)

with c4:
    best_score = data["results"][0]["score"]
    st.metric("**Best fit score**", f"{best_score:.0%}")

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
    cols = st.columns(4)
    for i, skill in enumerate(data["skills"]):
        cols[i % 4].markdown(
            f'<span class="stMarkdownBadge">{skill.skill}  (x{skill.count})</span>',
            unsafe_allow_html=True,
        )

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
# Footer / top-control hints (optional, visible on deployed share)
# --------------------------------------------------------------------------- #

st.caption(
    "Built with Streamlit • ONNX Runtime (CPU) • spaCy • Open-source career matcher"
)