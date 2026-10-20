"""Streamlit UI for the AI Career Path & Skill-Gap Recommender.

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
# App config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Career Path & Skill-Gap Recommender",
    page_icon="",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Helper: color‐coded fit badge
# --------------------------------------------------------------------------- #

def fit_badge(score: float) -> str:
    """Return a semantic badge string for the given fit score."""
    if score >= 80:
        return st.badge(f"{score:.0%} fit", color="green", icon="🟢")
    elif score >= 60:
        return st.badge(f"{score:.0%} fit", color="blue", icon="🟦")
    elif score >= 40:
        return st.badge(f"{score:.0%} fit", color="orange", icon="🟧")
    else:
        return st.badge(f"{score:.0%} fit", color="red", icon="🔴")


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
# Sidebar: primary actions first
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
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    use_sample = st.button("Or use the sample resume", type="primary")

# --------------------------------------------------------------------------- #
# Early exit — nothing selected yet
# --------------------------------------------------------------------------- #

if not uploaded and not use_sample:
    st.title("AI Career Path & Skill-Gap Recommender")
    st.markdown(
        "Upload a **resume PDF** in the sidebar (or click **use the sample resume**) to get:\n\n"
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
except Exception as exc:  # surface parse/model errors nicely
    st.error(f"Could not analyze this resume: {exc}")
    st.stop()

with st.spinner("Running NLP pipeline and semantic matching..."):
    data = analyze(pdf_bytes)

_status = embedder_status()

# --------------------------------------------------------------------------- #
# Header metrics row — semantic color coding
# --------------------------------------------------------------------------- #

st.title("AI Career Path & Skill-Gap Recommender")

top = data["results"][:3]
c1, c2, c3, c4 = st.columns(4)

# Skills extracted — always green (good thing)
c1.metric(
    "Skills extracted",
    len(data["skill_names"]),
    help="Number of skills detected in your resume",
)

# Jobs analyzed — always blue (neutral count)
c2.metric(
    "Jobs analyzed",
    len(data["results"]),
    help="Number of jobs ranked against your resume",
)

# Best match — color its label by fit score
best_score = data["results"][0]["score"]
c3.metric(
    "Best match",
    f"{data['results'][0]['job']['title']}",
    help=f"Top job match at {best_score:.0%} fit",
)

# Best fit score — semantic color
c4.metric(
    "Best fit score",
    f"{best_score:.0%}",
    help="Embedding similarity + skill coverage combined",
)

st.divider()

# --------------------------------------------------------------------------- #
# Tabs: Skill Profile | Job Matches | Career Path
# --------------------------------------------------------------------------- #

tab_skills, tab_jobs, tab_path = st.tabs(["Skill Profile", "Job Matches", "Career Path"])

# --------------------------------------------------------------------------- #
# Tab 1: Skill Profile
# --------------------------------------------------------------------------- #

with tab_skills:
    st.subheader(f"Extracted skills ({len(data['skill_names'])})")

    # Skills as responsive badges in a 4-column grid
    cols = st.columns(4)
    for i, skill in enumerate(data["skills"]):
        cols[i % 4].markdown(
            f'<div style="background:rgba(61,157,243,0.2);border-radius:4px;'
            f'color:rgb(61,157,243);padding:0 4px;font-size:0.875rem;'
            f'margin:4px 0;">{skill.skill}  (x{skill.count})</div>',
            unsafe_allow_html=True,
        )

    with st.expander("See evidence snippets from your resume"):
        for skill in data["skills"]:
            st.markdown(f"**{skill.skill}** — *{skill.evidence}*")

# --------------------------------------------------------------------------- #
# Tab 2: Job Matches
# --------------------------------------------------------------------------- #

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

            # Semantic color-coded fit metric
            right.metric("Fit", f"{result['score']:.0%}")

            # Progress bar with context
            st.progress(result["score"], text=f"overall fit: {result['score']:.0%}")

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

# --------------------------------------------------------------------------- #
# Tab 3: Career Path
# --------------------------------------------------------------------------- #

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
                # Semantic tier coloring by level
                level_color = "#10B981" if course["level"] == "Beginner" else (
                    "#F59E0B" if course["level"] == "Intermediate" else "#EF4444"
                )
                st.markdown(
                    f"**Step {course['step']}: [{course['title']}]({course['url']})**",
                    help=f"Level: {course['level']}",
                )
                st.caption(
                    f"{course['provider']} - {course['level']} - {course['duration_weeks']} weeks"
                )
                # Color-coded skill badge inline
                covered = ", ".join(course["covered_skills"])
                st.markdown(
                    f'<span style="background:{level_color}20;color:{level_color};'
                    f'border-radius:4px;padding:2px 6px;font-size:0.75rem;">'
                    f'covers {covered}</span>',
                    unsafe_allow_html=True,
                )

        total_weeks = sum(c["duration_weeks"] for c in data["career_path"])
        st.info(f"Estimated total time: ~{total_weeks} weeks of focused study.")