"""Semantic job matching and skill-gap course recommendation.

Pipeline:
1. Embed the candidate's resume profile and every job description with the
   all-MiniLM-L6-v2 sentence model (ONNX Runtime - no PyTorch) and compute
   cosine similarity. Loading is bounded by a timeout and degrades to a
   lexical fallback, so the app never hangs on a slow model download.
2. Blend the semantic score with an explicit skill-coverage score derived
   from extracted resume skills vs. the job's required skills.
3. For the best-matching jobs, diff required vs. held skills to get the
   "skill gap", then rank catalog courses by how many gap skills they cover
   and assemble a ordered learning path (beginner -> advanced).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

# Must be set before huggingface_hub is imported so a stalled download
# raises instead of blocking the script forever.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# How long a run may wait for the embedding model before falling back.
EMBED_LOAD_TIMEOUT_S = float(os.environ.get("EMBED_LOAD_TIMEOUT_S", "180"))

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
_embedder_error: str | None = None
_embedder_state = "idle"  # idle -> loading -> ready | failed
_embedder_thread: threading.Thread | None = None
_embedder_lock = threading.Lock()


def _load_embedder_blocking():
    """Import fastembed and construct the ONNX model (downloads on first use)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL_ID)


def _embedder_worker() -> None:
    global _embedder, _embedder_error, _embedder_state
    started = time.monotonic()
    try:
        model = _load_embedder_blocking()
    except Exception as exc:  # noqa: BLE001 - degrade, never crash the app
        with _embedder_lock:
            _embedder_error = f"{type(exc).__name__}: {exc}"
            _embedder_state = "failed"
        print(
            f"[embed] model load FAILED after {time.monotonic() - started:.1f}s: "
            f"{_embedder_error}",
            flush=True,
        )
        return
    with _embedder_lock:
        _embedder = model
        _embedder_state = "ready"
    print(f"[embed] {EMBED_MODEL_ID} ready in {time.monotonic() - started:.1f}s", flush=True)


def embedder_status() -> str:
    """One of 'idle', 'loading', 'ready', 'failed' (for UI/log diagnostics)."""
    return _embedder_state


def _get_embedder(timeout: float = EMBED_LOAD_TIMEOUT_S):
    """Return the ONNX embedder, starting its one-time background load.

    Blocks at most ``timeout`` seconds waiting for the model. Returns ``None``
    when it is not ready yet (still downloading, timed out, or failed) so
    callers can fall back to the lexical embedder instead of hanging.
    """
    global _embedder_thread, _embedder_state
    with _embedder_lock:
        if _embedder_state == "idle":
            _embedder_state = "loading"
            _embedder_thread = threading.Thread(
                target=_embedder_worker, daemon=True, name="embed-model-load"
            )
            _embedder_thread.start()
        thread = _embedder_thread
    if _embedder is not None:
        return _embedder
    if thread is not None and timeout > 0:
        thread.join(timeout)
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
    """Return L2-normalised embeddings, shape (len(texts), dim).

    Uses the ONNX sentence model when it is ready; otherwise falls back to
    the in-process lexical embedder so a run always completes.
    """
    model = _get_embedder()
    if model is None:
        print(
            f"[embed] embedding model not ready (status={embedder_status()}) "
            "-> using lexical fallback for this run",
            flush=True,
        )
        return _fallback_embed(texts)
    embeddings = np.asarray(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype(np.float32)


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
