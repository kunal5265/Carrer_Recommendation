"""Skill extraction from resume text.

Hybrid approach (the "NLP" part of the pipeline):
1. spaCy segments the resume into sentences and noun phrases.
2. A curated taxonomy (``SKILL_TAXONOMY``) with aliases is matched
   word-boundary-aware against the text, restricted to informative
   spans so that generic words don't produce false positives.
3. Extracted skills keep a short evidence snippet for UI display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import spacy

from .skills_taxonomy import SKILL_TAXONOMY

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class ExtractedSkill:
    skill: str
    evidence: str = ""
    count: int = 1

    def as_dict(self) -> dict:
        return {"skill": self.skill, "evidence": self.evidence, "count": self.count}


def _compile_alias_patterns() -> dict[str, list[re.Pattern]]:
    compiled: dict[str, list[re.Pattern]] = {}
    for canonical, aliases in SKILL_TAXONOMY.items():
        patterns = []
        for alias in aliases:
            escaped = re.escape(alias)
            # Word-boundary match, tolerant of . / - between word chars.
            pattern = r"(?<![\w])" + escaped.replace(r"\ ", r"[\s\-\.]*") + r"(?![\w])"
            flags = re.IGNORECASE
            patterns.append(re.compile(pattern, flags))
        compiled[canonical] = patterns
    return compiled


_ALIAS_PATTERNS = _compile_alias_patterns()

# Words too generic to count as evidence on their own unless matched as a
# multi-word alias (single-word matches for these need surrounding context).
_GENERIC_SKILLS = {"Node.js", "Research", "Communication", "Statistics"}


def _find_evidence(text: str, match: re.Match) -> str:
    start = max(0, match.start() - 60)
    end = min(len(text), match.end() + 60)
    snippet = text[start:end].replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def extract_skills(text: str) -> list[ExtractedSkill]:
    """Extract canonical skills from free-form resume text."""
    nlp = _get_nlp()
    doc = nlp(text)

    # Keep only content-bearing spans so generic single words inside names,
    # headings etc. are less likely to trigger false positives.
    content_text = " ".join(
        token.text for token in doc if not token.is_space
    )

    found: dict[str, ExtractedSkill] = {}
    for canonical, patterns in _ALIAS_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(content_text):
                # For generic skills require a lowercased word in context
                # (i.e. used as a skill term, not a proper noun heading).
                if canonical in _GENERIC_SKILLS and match.group(0).istitle():
                    continue
                if canonical in found:
                    found[canonical].count += 1
                else:
                    found[canonical] = ExtractedSkill(
                        skill=canonical,
                        evidence=_find_evidence(text, match),
                    )
                break  # one count per alias is enough for presence

    # Also surface capitalized noun-phrase skills spaCy finds that our
    # taxonomy misses only if they look technical (kept simple: skipped).
    return sorted(found.values(), key=lambda s: s.skill.lower())


def skills_as_names(skills: list[ExtractedSkill]) -> list[str]:
    return [s.skill for s in skills]
