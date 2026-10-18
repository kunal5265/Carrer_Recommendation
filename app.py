"""Streamlit UI for the AI Career Path & Skill-Gap Recommender.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.matcher import build_career_path, match_jobs, recommend_courses
from src.resume_parser import extract_text_from_pdf
from src.skill_extractor import extract_skills, skills_as_names

SAMPLE_PDF = Path(__file__).parent / "sample" / "sample_resume.pdf"

st.set_page_config(
    page_title="AI Career Path & Skill-Gap Recommender",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def analyze(pdf_bytes: bytes) -> dict:
    """Full pipeline: PDF -> text -> skills -> job matches -> career path."""
    text = extract_text_from_pdf(pdf_bytes)
    skills = extract_skills(text)
    skill_names = skills_as_names(skills)
    results = match_jobs(text, skill_names)
    path = build_career_path(results)
    return {
        "text": text,
        "skills": skills,
        "skill_names": skill_names,
        "results": results,
        "career_path": path,
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title(" AI Career Advisor")
    st.markdown(
        """
**How it works (the 10/10 architecture)**

1. **Parse** - extracts text from your resume PDF
2. **NLP (spaCy)** - finds your skills in unstructured text
3. **Semantic matching** - sentence-transformers embeddings rank
   job postings by fit
4. **Skill-gap analysis** - diffs your skills vs. each job
5. **Recommendations** - picks courses that close the most
   important gaps and orders them into a learning path
        """
    )
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    use_sample = st.button("Or use the sample resume", type="secondary")

if not uploaded and not use_sample:
    st.title(" AI Career Path & Skill-Gap Recommender")
    st.markdown(
        "Upload a **resume PDF** in the sidebar (or click *use the sample "
        "resume*) to get:\n\n"
        "- Your extracted **skill profile**\n"
        "- **Top matched jobs** ranked by semantic fit\n"
        "- The **skill gap** for each job\n"
        "- **Courses** and an ordered **career path** to close those gaps"
    )
    st.stop()

try:
    if uploaded is not None:
        pdf_bytes = uploaded.read()
    else:
        pdf_bytes = SAMPLE_PDF.read_bytes()

    with st.spinner("Running NLP pipeline and semantic matching..."):
        data = analyze(pdf_bytes)
except Exception as exc:  # surface parse/model errors nicely
    st.error(f"Could not analyze this resume: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Header metrics
# --------------------------------------------------------------------------- #
st.title(" AI Career Path & Skill-Gap Recommender")
top = data["results"][:3]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Skills extracted", len(data["skill_names"]))
c2.metric("Jobs analyzed", len(data["results"]))
c3.metric("Best match", f"{data['results'][0]['job']['title']}")
c4.metric("Best fit score", f"{data['results'][0]['score']:.0%}")
st.divider()

tab_skills, tab_jobs, tab_path = st.tabs(
    [" Skill Profile", " Job Matches", " Career Path"]
)

# --------------------------------------------------------------------------- #
# Tab 1: skills
# --------------------------------------------------------------------------- #
with tab_skills:
    st.subheader(f"Extracted skills ({len(data['skill_names'])})")
    cols = st.columns(4)
    for i, skill in enumerate(data["skills"]):
        cols[i % 4].badge(f"{skill.skill}  (x{skill.count})")
    with st.expander("See evidence snippets from your resume"):
        for skill in data["skills"]:
            st.markdown(f"**{skill.skill}** - *{skill.evidence}*")

# --------------------------------------------------------------------------- #
# Tab 2: job matches
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
            right.metric("Fit", f"{result['score']:.0%}")
            st.progress(result["score"], text="overall fit")
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
                        f"- [{course['title']}]({course['url']}) - "
                        f"{course['provider']} ({course['level']}, {course['duration_weeks']}w) "
                        f"covers *{', '.join(course['covered_skills'])}*"
                    )

# --------------------------------------------------------------------------- #
# Tab 3: career path
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
                st.markdown(
                    f"**Step {course['step']}: [{course['title']}]({course['url']})**"
                )
                st.caption(
                    f"{course['provider']} - {course['level']} - "
                    f"{course['duration_weeks']} weeks"
                )
                st.markdown(
                    f"Covers: {', '.join(course['covered_skills'])}"
                )
        total_weeks = sum(c["duration_weeks"] for c in data["career_path"])
        st.info(f" Estimated total time: ~{total_weeks} weeks of focused study.")
