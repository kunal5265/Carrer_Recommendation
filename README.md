#   AI Career Path & Skill-Gap Recommender

A mini project that reads a **resume PDF**, extracts skills with **NLP (spaCy)**,
matches the candidate to **job postings via semantic embeddings
(sentence-transformers)**, and recommends **specific courses** to bridge the
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
   │  sentence-transformers (all-MiniLM-L6-v2)       │
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
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python sample/make_sample_resume.py   # create the demo resume
streamlit run app.py
```

Then upload your own resume PDF in the sidebar (or click *use the sample
resume*) to see your skill profile, top matched jobs, per-job skill gaps,
course recommendations, and your ordered career path.

> First analysis run downloads the `all-MiniLM-L6-v2` model (~90 MB) from
> Hugging Face. If the model can't be loaded (e.g. offline), the app falls
> back to a lightweight bag-of-words similarity so it still runs.

## Key Design Decisions

| Decision | Why |
|---|---|
| Hybrid NLP + taxonomy skill extraction | Pure NER misses skills ("spaCy" isn't a PERSON); pure keyword matching misses phrasing. spaCy provides span context, taxonomy gives precision. |
| Blend semantic similarity with skill coverage | Embeddings alone over-credit keyword-adjacent resumes; coverage alone misses paraphrases. The blend is robust to both. |
| Greedy set-cover for the career path | Maximizes gap coverage per course chosen, then orders beginner-first so prerequisites hold. |
| Lazy-loaded model + cached analysis | Keeps Streamlit startup instant; repeated uploads with the same PDF return instantly. |

## Extending It

- Add jobs/courses by appending to `data/*.json` — no code changes needed.
- Swap the embedding model in `src/matcher.py` (`MODEL_NAME`) for a domain model.
- Add PDF OCR (e.g. `pytesseract`) in `resume_parser.py` for scanned resumes.
- Persist user profiles (e.g. SQLite) to track progress over time.
