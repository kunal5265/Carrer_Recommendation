"""Semantic job matching and skill-gap course recommendation.

Pipeline:
1. Embed the candidate's resume profile and every job description with a
   sentence-transformer (all-MiniLM-L6-v2) and compute cosine similarity.
2. Blend the semantic score with an explicit skill-coverage score derived
   from extracted resume skills vs. the job's required skills.
3. For the best-matching jobs, diff required vs. held skills to get the
   "skill gap", then rank catalog courses by how many gap skills they cover
   and assemble a ordered learning path (beginner -> advanced).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "all-MiniLM-L6-v2"

# Weights for the final match score.
SEMANTIC_WEIGHT = 0.5
SKILL_WEIGHT = 0.5


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_json(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_jobs() -> tuple[dict, ...]:
    return tuple(_load_json("jobs.json"))


@lru_cache(maxsize=1)
def load_courses() -> tuple[dict, ...]:
    return tuple(_load_json("courses.json"))


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
_embedder = None
_embedder_failed = False


def _get_embedder():
    """Lazy-load the sentence-transformer (first call downloads the model)."""
    global _embedder, _embedder_failed
    if _embedder is None and not _embedder_failed:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(MODEL_NAME)
        except Exception:
            _embedder_failed = True
    return _embedder


def _fallback_embed(texts: list[str]) -> np.ndarray:
    """Lightweight bag-of-words fallback if the model can't load.

    Keeps the app functional offline (used only as a last resort); quality
    is lower than real sentence embeddings.
    """
    vocab: dict[str, int] = {}
    docs: list[dict[int, int]] = []
    for text in texts:
        counts: dict[int, int] = {}
        for token in text.lower().split():
            idx = vocab.setdefault(token, len(vocab))
            counts[idx] = counts.get(idx, 0) + 1
        docs.append(counts)
    matrix = np.zeros((len(docs), max(len(vocab), 1)), dtype=np.float32)
    for row, counts in enumerate(docs):
        for idx, count in counts.items():
            matrix[row, idx] = count
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (len(texts), dim)."""
    model = _get_embedder()
    if model is None:
        return _fallback_embed(texts)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def skill_coverage(required: list[str], held: set[str]) -> float:
    if not required:
        return 0.0
    held_lower = {s.lower() for s in held}
    hit = sum(1 for s in required if s.lower() in held_lower)
    return hit / len(required)


def skill_gap(required: list[str], held: set[str]) -> tuple[list[str], list[str]]:
    """Return (missing_skills, matched_skills) for a job."""
    held_lower = {s.lower(): s for s in held}
    missing, matched = [], []
    for skill in required:
        if skill.lower() in held_lower:
            matched.append(skill)
        else:
            missing.append(skill)
    return missing, matched


def profile_text(resume_text: str, skills: list[str]) -> str:
    """Compact profile document to embed for the candidate."""
    return (
        "Candidate skills: " + ", ".join(skills) + ". Resume: " + resume_text
    )[:6000]


def match_jobs(
    resume_text: str, skills: list[str], top_n: int | None = None
) -> list[dict]:
    """Rank jobs for a candidate. Each result carries scores, gap & courses."""
    jobs = load_jobs()
    held = set(skills)

    candidate_profile = profile_text(resume_text, skills)
    job_texts = [f"{j['title']}. {j['company']}. {j['description']}" for j in jobs]

    embeddings = embed_texts([candidate_profile] + job_texts)
    cand_emb, job_embs = embeddings[0], embeddings[1:]

    results = []
    for job, job_emb in zip(jobs, job_embs):
        semantic = cosine_sim(cand_emb, job_emb)
        coverage = skill_coverage(job["required_skills"], held)
        score = SEMANTIC_WEIGHT * semantic + SKILL_WEIGHT * coverage
        missing, matched = skill_gap(job["required_skills"], held)
        results.append(
            {
                "job": dict(job),
                "score": round(score, 4),
                "semantic_score": round(semantic, 4),
                "skill_coverage": round(coverage, 4),
                "missing_skills": missing,
                "matched_skills": matched,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    if top_n:
        results = results[:top_n]
    return results


# --------------------------------------------------------------------------- #
# Course recommendation
# --------------------------------------------------------------------------- #
def recommend_courses(
    missing_skills: list[str], top_k: int = 3
) -> list[dict]:
    """Rank courses by how many of the requested gap skills they teach."""
    if not missing_skills:
        return []
    wanted = {s.lower() for s in missing_skills}
    courses = []
    for course in load_courses():
        covered = [s for s in course["skills"] if s.lower() in wanted]
        if not covered:
            continue
        courses.append(
            {
                **course,
                "covered_skills": covered,
                "coverage": round(len(covered) / len(wanted), 4),
            }
        )
    courses.sort(
        key=lambda c: (len(c["covered_skills"]), -c["duration_weeks"]), reverse=True
    )
    return courses[:top_k]


_LEVEL_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}


def build_career_path(results: list[dict], max_jobs: int = 3, max_courses: int = 6) -> list[dict]:
    """Greedy learning path across the top-matching jobs' skill gaps.

    Picks courses that cover the most still-uncovered gap skills, then
    orders them beginner-first so the path is prerequisite friendly.
    """
    gap_skills: list[str] = []
    seen: set[str] = set()
    for result in results[:max_jobs]:
        for skill in result["missing_skills"]:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                gap_skills.append(skill)

    if not gap_skills:
        return []

    remaining = {s.lower() for s in gap_skills}
    path: list[dict] = []
    available = list(load_courses())
    while remaining and len(path) < max_courses:
        best, best_covered = None, []
        for course in available:
            covered = [s for s in course["skills"] if s.lower() in remaining]
            if len(covered) > len(best_covered):
                best, best_covered = course, covered
        if best is None:
            break
        remaining -= {s.lower() for s in best_covered}
        path.append(
            {
                **best,
                "covered_skills": best_covered,
                "step": len(path) + 1,
            }
        )
        available = [c for c in available if c["id"] != best["id"]]

    path.sort(key=lambda c: (_LEVEL_ORDER.get(c["level"], 9), c["duration_weeks"]))
    for i, course in enumerate(path, start=1):
        course["step"] = i
    return path
