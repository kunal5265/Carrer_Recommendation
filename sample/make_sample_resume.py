"""Generate a sample resume PDF used to demo the app without a real resume."""

from fpdf import FPDF

SAMPLE_RESUME_TEXT = """\
ARJUN MEhta
Bengaluru, India | arjun.mehta@email.com | +91 98765 43210 | github.com/arjunm

SUMMARY
Final-year Computer Science student focused on data science and NLP. Built
semantic search and text classification prototypes with Python, pandas and
spaCy. Comfortable with Git, SQL and presenting results to non-technical teams.

EDUCATION
B.Tech, Computer Science & Engineering            2022 - 2026
Relevant coursework: Data Structures, Databases, Discrete Mathematics

TECHNICAL SKILLS
Languages: Python, SQL
Data: pandas, NumPy, matplotlib
Tools: Git, Excel

PROJECTS
Resume Skill Extractor            (Python, spaCy, pandas)
  - Parsed 500+ resumes and extracted skills with an NLP pipeline.
  - Built a semantic search demo over resume text.

Movie Rating Predictor            (Python, NumPy)
  - Trained predictive models and evaluated their accuracy.

Retail Sales Dashboard            (SQL, Tableau, Excel)
  - Built an interactive dashboard tracking weekly KPIs for a club project.

EXPERIENCE
Data Science Intern, Nimbus Analytics (Remote)     Jun 2025 - Aug 2025
  - Cleaned and analyzed datasets with pandas, produced charts for stakeholders.
  - Documented pipelines and presented findings to the analytics team.

STRENGTHS
Teamwork, communication, problem solving, quick learner.
"""


def build_sample_resume(path: str = "sample/sample_resume.pdf") -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(pdf.epw, 10, "ARJUN MEHTA")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(pdf.epw, 6, "Bengaluru, India | arjun.mehta@email.com | +91 98765 43210")
    pdf.ln(4)

    lines = SAMPLE_RESUME_TEXT.split("\n")[4:]  # skip duplicated header block
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(3)
            continue
        is_heading = stripped.isupper() and len(stripped) < 30
        if is_heading:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(pdf.epw, 6, stripped)
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 5, line)
    pdf.output(path)
    return path


if __name__ == "__main__":
    print("Wrote", build_sample_resume())
