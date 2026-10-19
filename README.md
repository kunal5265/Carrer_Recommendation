#   AI Career Path & Skill-Gap Recommender

A mini project that reads a **resume PDF**, extracts skills with **NLP (spaCy)**,
matches the candidate to **job postings via semantic embeddings
(the `all-MiniLM-L6-v2` sentence model served by ONNX Runtime - no PyTorch)**,
and recommends **specific courses** to bridge the
missing skills — then orders them into an actionable **career path**.

## How It Works (the "10/10" architecture)

```
 resume.pdf ──▶ pypdf ──▶ raw text
                            │
                   spaCy (en_core_web_sm)
                   sentence segmentation + tokenization
                            │
              skill taxonomy matcher (aliases, word boundaries)
                            │
                    extracted skills + evidence
                            │
   ┌────────────────────────┴────────────────────────┐
   │  all-MiniLM-L6-v2 (ONNX Runtime, via fastembed) │
   │  embed(resume profile)  vs  embed(job postings) │
   │  cosine similarity = semantic fit score         │
   └────────────────────────┬────────────────────────┘
                            │
     score = 0.5 · semantic + 0.5 · skill coverage
                            │
     skill gap = job.required_skills − candidate.skills
                            │
   greedy set-cover over gap skills ──▶ ranked courses ──▶ ordered career path
```

**What makes it go beyond a simple rating predictor:**

- Handles **unstructured PDF input** (real resumes, not structured forms).
- Uses **semantic embeddings**, so a resume saying "LLM / RAG prototypes"
  still matches an NLP Engineer posting even when keywords don't line up.
- Computes an explicit **skill gap** per job, not just a score.
- Recommends **specific courses** that close the most important gaps and
  sequences them beginner-first into a study plan.

## Project Structure

```
.
├── app.py                    # Streamlit UI (entry point)
├── requirements.txt
├── data/
│   ├── jobs.json             # 15 sample job postings with required skills
│   └── courses.json          # 24-course catalog with skills taught
├── src/
│   ├── resume_parser.py      # PDF -> text (pypdf)
│   ├── skill_extractor.py    # text -> skills (spaCy + taxonomy matcher)
│   ├── skills_taxonomy.py    # canonical skills + aliases
│   └── matcher.py            # embeddings, matching, skill gap, recommender
└── sample/
    ├── make_sample_resume.py # generates a demo resume PDF
    └── sample_resume.pdf
```

## Setup & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # includes the en_core_web_sm model wheel

python sample/make_sample_resume.py   # create the demo resume
streamlit run app.py
```

Then upload your own resume PDF in the sidebar (or click *use the sample
resume*) to see your skill profile, top matched jobs, per-job skill gaps,
course recommendations, and your ordered career path.

> The first analysis run downloads the `all-MiniLM-L6-v2` ONNX model
> (~90 MB) from Hugging Face, in a background thread with a bounded wait
> (`EMBED_LOAD_TIMEOUT_S`, default 180 s). If the model isn't ready in time
> or can't be downloaded (e.g. offline), the app falls back to a lightweight
> bag-of-words similarity for that run so it **always** finishes instead of
> hanging — the banner in the UI tells you which ranker was used.

## Key Design Decisions

| Decision | Why |
|---|---|
| Hybrid NLP + taxonomy skill extraction | Pure NER misses skills ("spaCy" isn't a PERSON); pure keyword matching misses phrasing. spaCy provides span context, taxonomy gives precision. |
| Blend semantic similarity with skill coverage | Embeddings alone over-credit keyword-adjacent resumes; coverage alone misses paraphrases. The blend is robust to both. |
| Greedy set-cover for the career path | Maximizes gap coverage per course chosen, then orders beginner-first so prerequisites hold. |
| Lazy-loaded model + cached analysis | Keeps Streamlit startup instant; repeated uploads with the same PDF return instantly. |
| ONNX Runtime instead of PyTorch (`fastembed`) | Same MiniLM weights, ~140 MB less peak memory (measured: 576 MB → 439 MB for the pipeline) and no multi-GB torch install — the app fits inside Streamlit Community Cloud's small free-tier container instead of being OOM-killed mid-run. |
| Bounded model load with lexical fallback | A stalled Hugging Face download can no longer freeze a run forever: the load runs in a background thread, the run waits at most `EMBED_LOAD_TIMEOUT_S`, then ranks lexically and says so in the UI. |

## Extending It

- Add jobs/courses by appending to `data/*.json` — no code changes needed.
- Swap the embedding model in `src/matcher.py` (`EMBED_MODEL_ID`) for any
  model in `fastembed.TextEmbedding.list_supported_models()`.
- Add PDF OCR (e.g. `pytesseract`) in `resume_parser.py` for scanned resumes.
- Persist user profiles (e.g. SQLite) to track progress over time.
