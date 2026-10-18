"""End-to-end pipeline verification (no UI).

Usage: .venv/bin/python scripts/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sample.make_sample_resume import build_sample_resume  # noqa: E402
from src.matcher import build_career_path, match_jobs, recommend_courses  # noqa: E402
from src.resume_parser import extract_text_from_pdf  # noqa: E402
from src.skill_extractor import extract_skills, skills_as_names  # noqa: E402


def main() -> int:
    pdf_path = build_sample_resume(str(ROOT / "sample" / "sample_resume.pdf"))
    text = extract_text_from_pdf(pdf_path)
    assert len(text) > 500, f"resume text too short: {len(text)}"

    skills = extract_skills(text)
    names = skills_as_names(skills)
    print(f"[1] Extracted {len(names)} skills: {', '.join(names)}")
    assert "Python" in names, "Python not extracted"
    assert "SQL" in names, "SQL not extracted"
    assert "pandas" in names or "Pandas" in names, "pandas not extracted"

    results = match_jobs(text, names)
    print(f"[2] Matched {len(results)} jobs; top 3:")
    for r in results[:3]:
        print(
            f"    {r['score']:.0%}  {r['job']['title']} "
            f"(semantic={r['semantic_score']:.2f}, "
            f"coverage={r['skill_coverage']:.2f}, "
            f"gap={len(r['missing_skills'])})"
        )
    assert all(0.0 <= r["score"] <= 1.0 for r in results)
    assert results[0]["score"] >= results[-1]["score"], "results not sorted"

    best = results[0]
    courses = recommend_courses(best["missing_skills"], top_k=3)
    print(f"[3] Gap for '{best['job']['title']}': {best['missing_skills']}")
    for c in courses:
        print(f"    course: {c['title']} -> covers {c['covered_skills']}")

    path = build_career_path(results)
    print(f"[4] Career path ({len(path)} steps):")
    for step in path:
        print(f"    {step['step']}. {step['title']} ({step['level']})")
    assert path, "career path should not be empty for the sample resume"
    covered = {s.lower() for step in path for s in step["covered_skills"]}
    top_gaps = {s.lower() for r in results[:3] for s in r["missing_skills"]}
    print(f"    covers {len(covered & top_gaps)}/{len(top_gaps)} gap skills from top 3 jobs")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
