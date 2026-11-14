"""
backend/app/services/medicine_extraction.py
Prescription medicine mention and attribute candidate extraction service.

Phase 3 Scope & Safety Notice:
- Extracts candidate mentions from raw OCR text using explainable rules, regex patterns,
  and generic prescription-layout patterns.
- Preserves original OCR text strings verbatim without autocorrecting, normalizing,
  substituting, or inferring missing attributes.
- All candidate mentions begin in review_status="pending_review".
- This is an academic research component, not a clinical decision-support system.
"""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Maximum allowed OCR input text length to prevent DoS / memory exhaustion
MAX_OCR_TEXT_LENGTH = 50_000


# ---------------------------------------------------------------------------
# Regular Expression Rule Patterns
# ---------------------------------------------------------------------------

# Strengths: numeric value followed by common pharmaceutical units
# Examples: 500mg, 500 mg, 0.5%, 250mcg, 10 IU, 5 ml, 100 mg/ml
STRENGTH_PATTERN = re.compile(
    r"(?:\b(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mg\/ml|mcg\/ml|mg|mcg|μg|ug|g|gm|ml|iu|meq)\b|"
    r"\b(?P<val_pct>\d+(?:\.\d+)?)\s*(?P<unit_pct>%))",
    re.IGNORECASE,
)

# Dosage forms: common prescription presentations
DOSAGE_FORM_PATTERN = re.compile(
    r"\b(?P<form>tablet|tablets|tab|tabs|capsule|capsules|cap|caps|syrup|syr|"
    r"injection|inj|suspension|susp|solution|sol|ointment|oint|cream|crm|"
    r"drops|drp|inhaler|patch|suppository|supp|lotion|gel)\b",
    re.IGNORECASE,
)

# Route of administration
ROUTE_PATTERN = re.compile(
    r"\b(?P<route>oral|po|iv|im|sc|subcutaneous|sublingual|sl|topical|rectal|pr|"
    r"ophthalmic|otic|nasal|inhalation)\b",
    re.IGNORECASE,
)

# Dosing frequencies & Sig instructions
FREQUENCY_PATTERN = re.compile(
    r"\b(?P<freq>once daily|twice daily|thrice daily|three times daily|four times daily|"
    r"every \d+ hours|q\d+h|q\d+hr|qd|bid|tid|qid|qhs|qam|qpm|prn|daily|"
    r"as needed|at bedtime|before meals|after meals|ac|pc|"
    r"for \d+ days|x \d+ days|x\d+d)\b",
    re.IGNORECASE,
)

# Non-medicine metadata lines or tokens to avoid labeling as drug names
HEADER_METADATA_PATTERN = re.compile(
    r"^(date|patient|name|age|sex|gender|dr|doctor|physician|clinic|hospital|"
    r"address|phone|tel|license|reg|regd|dispense|disp|refill|refills|sig|rx|sign|signature)\b[:\s]?",
    re.IGNORECASE,
)

STOPWORDS_NON_MEDICINE = {
    "patient", "doctor", "physician", "clinic", "hospital", "prescription",
    "address", "phone", "date", "name", "years", "year", "male", "female",
    "age", "weight", "height", "bp", "pulse", "signature", "signed",
    "dispense", "disp", "refill", "refills", "sig", "take", "apply",
    "instill", "use", "oral", "po", "daily", "tabs", "caps", "tablet", "capsule",
    "day", "days", "week", "weeks", "month", "months", "mg", "ml", "mcg",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CandidateType(str, Enum):
    """Categorization of extracted prescription candidate mention."""

    MEDICINE_NAME = "possible_medicine_name"
    STRENGTH = "strength"
    DOSAGE_FORM = "dosage_form"
    FREQUENCY_INSTRUCTION = "frequency_instruction"
    ROUTE = "route"
    OTHER_EVIDENCE = "other_evidence"


class ExtractionMethod(str, Enum):
    """Transparent identification of the rule/mechanism that extracted the candidate."""

    REFERENCE_DICTIONARY = "reference_dictionary_match"
    CONTEXTUAL_PATTERN = "contextual_prescription_pattern"
    REGEX_RULE = "regex_rule"
    GEMMA_STRUCTURED = "gemma_structured_output"


class ExtractedMention(BaseModel):
    """
    An individual candidate mention identified in the raw OCR text.
    Maintains transparent evidence and verbatim text preservation.
    """

    candidate_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for tracking the candidate through the review lifecycle",
    )
    text: str = Field(description="Verbatim text exactly as extracted from OCR output")
    candidate_type: CandidateType
    start_char: int = Field(ge=0, description="Global character start offset in raw OCR text")
    end_char: int = Field(ge=0, description="Global character end offset in raw OCR text")
    line_number: int = Field(ge=1, description="Line number in original OCR text (1-indexed)")
    extraction_method: ExtractionMethod
    review_status: str = Field(
        default="pending_review",
        description="Mandatory review state. Mentions are never automatically approved.",
    )
    evidence_context: str = Field(description="Original line context containing this candidate")
    heuristic_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Uncalibrated rule-based heuristic weight; not a verified statistical probability.",
    )
    ocr_engine_confidence: float | None = Field(
        default=None,
        description="Confidence value directly from OCR engine, or null if unavailable. Never fabricated.",
    )
    ocr_confidence_available: bool = Field(
        default=False,
        description="Whether usable confidence was supplied by the OCR engine.",
    )
    has_dictionary_match: bool = False
    has_contextual_pattern: bool = False
    has_spillover_or_ambiguity: bool = False


class PrescriptionLineCandidate(BaseModel):
    """Structured grouping of extracted attributes belonging to a single prescription line."""

    line_number: int
    raw_line: str
    is_likely_prescription_line: bool = False
    medicine_name_candidate: ExtractedMention | None = None
    strength_candidate: ExtractedMention | None = None
    dosage_form_candidate: ExtractedMention | None = None
    frequency_candidate: ExtractedMention | None = None
    route_candidate: ExtractedMention | None = None
    unmatched_tokens: list[str] = Field(default_factory=list)


class MedicineExtractionResult(BaseModel):
    """Complete extraction payload returned for a given OCR text input."""

    status: str = "success"
    raw_text: str
    line_count: int
    candidates: list[ExtractedMention] = Field(default_factory=list)
    prescription_lines: list[PrescriptionLineCandidate] = Field(default_factory=list)
    unmatched_lines: list[str] = Field(default_factory=list)
    review_status: str = "pending_review"
    review_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Medicine Candidate Extraction Service
# ---------------------------------------------------------------------------


class MedicineCandidateExtractor:
    """
    Explainable, rule-based candidate extractor for prescription text.
    Preserves exact substrings and offsets without autocorrecting or modifying tokens.
    """

    def __init__(
        self,
        reference_vocab: set[str] | None = None,
    ) -> None:
        # Compatibility argument retained, but not used: unknown medicine names
        # must not be filtered by a fixed vocabulary.
        del reference_vocab

    def extract(self, raw_text: str) -> MedicineExtractionResult:
        """
        Analyze raw OCR text and extract candidate medicine names, strengths,
        dosage forms, and administration instructions.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Input OCR text is empty.")

        if len(raw_text) > MAX_OCR_TEXT_LENGTH:
            raise ValueError(
                f"OCR text exceeds maximum allowed length of {MAX_OCR_TEXT_LENGTH} characters."
            )

        lines = raw_text.splitlines(keepends=True)
        candidates: list[ExtractedMention] = []
        prescription_lines: list[PrescriptionLineCandidate] = []
        unmatched_lines: list[str] = []

        global_char_offset = 0

        for line_idx, raw_line in enumerate(lines, start=1):
            line_start_char = global_char_offset
            line = raw_line.splitlines()[0] if raw_line.splitlines() else ""
            global_char_offset += len(raw_line)

            trimmed_line = line.strip()
            if not trimmed_line:
                continue

            # Process attributes on the line
            line_candidates, line_group = self._process_line(
                line=line,
                line_idx=line_idx,
                line_start_char=line_start_char,
            )

            candidates.extend(line_candidates)
            prescription_lines.append(line_group)

            if not line_group.is_likely_prescription_line:
                unmatched_lines.append(trimmed_line)

        warnings = [
            "Candidate mentions are unverified rule-based extractions from raw OCR output.",
            "Spelling, dosages, and drug identities have NOT been confirmed or normalized.",
            "All extracted candidates require an explicit human review action.",
            "Heuristic scores are uncalibrated rule weights, not probabilities, and have not been validated on a labeled dataset.",
        ]

        return MedicineExtractionResult(
            raw_text=raw_text,
            line_count=len(lines),
            candidates=candidates,
            prescription_lines=prescription_lines,
            unmatched_lines=unmatched_lines,
            review_status="pending_review",
            warnings=warnings,
            metadata={
                "extractor_version": "1.0.0",
                "rules_applied": [
                    "regex_strength_detection",
                    "regex_dosage_form_detection",
                    "regex_frequency_detection",
                    "prescription_layout_phrase_extraction",
                ],
                "total_candidates_found": len(candidates),
                "heuristic_score_interpretation": (
                    "Uncalibrated rule-based weight; not a probability and not validated on a labeled dataset."
                ),
            },
        )

    def _process_line(
        self,
        line: str,
        line_idx: int,
        line_start_char: int,
    ) -> tuple[list[ExtractedMention], PrescriptionLineCandidate]:
        """Process a single line of text and extract attributes and medicine mentions."""
        line_mentions: list[ExtractedMention] = []

        # 1. Detect Strengths
        strength_mentions: list[ExtractedMention] = []
        strength_mention = None
        for match in STRENGTH_PATTERN.finditer(line):
            start = line_start_char + match.start()
            end = line_start_char + match.end()
            mention = ExtractedMention(
                text=line[match.start():match.end()],
                candidate_type=CandidateType.STRENGTH,
                start_char=start,
                end_char=end,
                line_number=line_idx,
                extraction_method=ExtractionMethod.REGEX_RULE,
                evidence_context=line.strip(),
                heuristic_score=0.85,
                has_dictionary_match=False,
                has_contextual_pattern=False,
            )
            line_mentions.append(mention)
            strength_mentions.append(mention)
            if strength_mention is None:
                strength_mention = mention

        # 2. Detect Dosage Form
        dosage_form_mentions: list[ExtractedMention] = []
        dosage_form_mention = None
        for match in DOSAGE_FORM_PATTERN.finditer(line):
            start = line_start_char + match.start()
            end = line_start_char + match.end()
            mention = ExtractedMention(
                text=line[match.start():match.end()],
                candidate_type=CandidateType.DOSAGE_FORM,
                start_char=start,
                end_char=end,
                line_number=line_idx,
                extraction_method=ExtractionMethod.REGEX_RULE,
                evidence_context=line.strip(),
                heuristic_score=0.85,
                has_dictionary_match=False,
                has_contextual_pattern=False,
            )
            line_mentions.append(mention)
            dosage_form_mentions.append(mention)
            if dosage_form_mention is None:
                dosage_form_mention = mention

        # 3. Detect Route
        route_mention = None
        for match in ROUTE_PATTERN.finditer(line):
            start = line_start_char + match.start()
            end = line_start_char + match.end()
            mention = ExtractedMention(
                text=line[match.start():match.end()],
                candidate_type=CandidateType.ROUTE,
                start_char=start,
                end_char=end,
                line_number=line_idx,
                extraction_method=ExtractionMethod.REGEX_RULE,
                evidence_context=line.strip(),
                heuristic_score=0.80,
                has_dictionary_match=False,
                has_contextual_pattern=False,
            )
            line_mentions.append(mention)
            if route_mention is None:
                route_mention = mention

        # 4. Detect Frequency / Instructions
        frequency_mention = None
        for match in FREQUENCY_PATTERN.finditer(line):
            start = line_start_char + match.start()
            end = line_start_char + match.end()
            mention = ExtractedMention(
                text=line[match.start():match.end()],
                candidate_type=CandidateType.FREQUENCY_INSTRUCTION,
                start_char=start,
                end_char=end,
                line_number=line_idx,
                extraction_method=ExtractionMethod.REGEX_RULE,
                evidence_context=line.strip(),
                heuristic_score=0.80,
                has_dictionary_match=False,
                has_contextual_pattern=False,
            )
            line_mentions.append(mention)
            if frequency_mention is None:
                frequency_mention = mention

        # 5. Detect Medicine Name Candidates (including repeated/combination mentions)
        med_mentions = self._extract_medicine_names(
            line=line,
            line_idx=line_idx,
            line_start_char=line_start_char,
            strength_mentions=strength_mentions,
            dosage_form_mentions=dosage_form_mentions,
        )

        line_mentions.extend(med_mentions)
        primary_med_mention = med_mentions[0] if med_mentions else None

        # Determine if this line looks like a valid prescription entry
        is_rx_line = (primary_med_mention is not None) or (
            strength_mention is not None and dosage_form_mention is not None
        )

        # Identify leftover unmatched tokens
        matched_spans = [
            (m.start_char - line_start_char, m.end_char - line_start_char)
            for m in line_mentions
        ]
        unmatched_tokens = self._find_unmatched_tokens(line, matched_spans)

        line_candidate = PrescriptionLineCandidate(
            line_number=line_idx,
            raw_line=line,
            is_likely_prescription_line=is_rx_line,
            medicine_name_candidate=primary_med_mention,
            strength_candidate=strength_mention,
            dosage_form_candidate=dosage_form_mention,
            frequency_candidate=frequency_mention,
            route_candidate=route_mention,
            unmatched_tokens=unmatched_tokens,
        )

        return line_mentions, line_candidate

    def _extract_medicine_names(
        self,
        line: str,
        line_idx: int,
        line_start_char: int,
        strength_mentions: list[ExtractedMention],
        dosage_form_mentions: list[ExtractedMention],
    ) -> list[ExtractedMention]:
        """Extract verbatim name-position phrases without a medicine-name list."""
        if HEADER_METADATA_PATTERN.search(line.strip()):
            if not re.match(r"^rx\b", line.strip(), re.IGNORECASE):
                return []
        if not (strength_mentions or dosage_form_mentions or
                ROUTE_PATTERN.search(line) or FREQUENCY_PATTERN.search(line)):
            return []

        prefix = re.match(r"^(\d+[\.)]\s*|rx[:\s]+|[-?]\s*)", line, re.IGNORECASE)
        line_start = prefix.end() if prefix else 0
        trim_chars = " \t:;,-.()[]"
        segments: list[tuple[int, int]] = []
        strengths = sorted(strength_mentions, key=lambda m: m.start_char)
        if strengths:
            first_strength = strengths[0].start_char - line_start_char
            first_end_candidates = [first_strength]
            first_end_candidates.extend(
                m.start_char - line_start_char for m in dosage_form_mentions
                if m.start_char - line_start_char >= line_start
                and m.start_char - line_start_char != line_start
            )
            first_end_candidates.extend(
                m.start() for pattern in (ROUTE_PATTERN, FREQUENCY_PATTERN)
                for m in pattern.finditer(line) if m.start() >= line_start
            )
            first_leading_form = next((
                m for m in dosage_form_mentions
                if m.start_char - line_start_char == line_start
            ), None)
            first_segment_start = (
                first_leading_form.end_char - line_start_char
                if first_leading_form else line_start
            )
            segments.append((first_segment_start, min(first_end_candidates)))
            for previous, current in zip(strengths, strengths[1:]):
                seg_start = previous.end_char - line_start_char
                seg_end = current.start_char - line_start_char
                fragment = line[seg_start:seg_end]
                # A second medicine in a combined line is usually separated by
                # "and", a comma, plus, or ampersand; discard the earlier sig.
                separators = list(re.finditer(r"(?:\band\b|[,;&+])", fragment, re.IGNORECASE))
                if separators:
                    segments.append((seg_start + separators[-1].end(), seg_end))
        else:
            if dosage_form_mentions:
                first_form = min(dosage_form_mentions, key=lambda m: m.start_char)
                form_pos = first_form.start_char - line_start_char
                after_form = first_form.end_char - line_start_char
                if form_pos == line_start:
                    line_start = after_form
            end_candidates = [
                m.start_char - line_start_char for m in dosage_form_mentions
                if m.start_char - line_start_char > line_start
            ]
            end_candidates.extend(
                m.start() for pattern in (ROUTE_PATTERN, FREQUENCY_PATTERN)
                for m in pattern.finditer(line) if m.start() >= line_start
            )
            segments.append((line_start, min(end_candidates, default=len(line))))

        found: list[ExtractedMention] = []
        seen: set[tuple[int, int]] = set()
        for start, end in segments:
            while start < end and line[start] in trim_chars:
                start += 1
            while end > start and line[end - 1] in trim_chars:
                end -= 1
            tokens = list(re.finditer(r"[A-Za-z][A-Za-z0-9'?\-]*", line[start:end]))
            while tokens and tokens[0].group().casefold() in STOPWORDS_NON_MEDICINE:
                start += tokens[0].end()
                while start < end and line[start] in trim_chars:
                    start += 1
                tokens = list(re.finditer(r"[A-Za-z][A-Za-z0-9'?\-]*", line[start:end]))
            while tokens and tokens[-1].group().casefold() in STOPWORDS_NON_MEDICINE:
                end = start + tokens[-1].start()
                tokens = list(re.finditer(r"[A-Za-z][A-Za-z0-9'?\-]*", line[start:end]))
            if not tokens:
                continue
            end = start + tokens[-1].end()
            text = line[start:end]
            if text.casefold() in STOPWORDS_NON_MEDICINE or (start, end) in seen:
                continue
            seen.add((start, end))
            found.append(ExtractedMention(
                text=text,
                candidate_type=CandidateType.MEDICINE_NAME,
                start_char=line_start_char + start,
                end_char=line_start_char + end,
                line_number=line_idx,
                extraction_method=ExtractionMethod.CONTEXTUAL_PATTERN,
                evidence_context=line.strip(),
                heuristic_score=0.65,
                has_contextual_pattern=True,
                has_spillover_or_ambiguity=True,
            ))
        return found

    def _find_unmatched_tokens(
        self, line: str, matched_spans: list[tuple[int, int]]
    ) -> list[str]:
        """Collect non-trivial word tokens that were not claimed by any candidate entity."""
        unmatched: list[str] = []
        for m in re.finditer(r"\b[A-Za-z0-9\-\.]{2,}\b", line):
            m_start, m_end = m.start(), m.end()
            # Check overlap with any matched span
            is_matched = any(
                not (m_end <= s or m_start >= e) for s, e in matched_spans
            )
            if not is_matched and m.group().lower() not in STOPWORDS_NON_MEDICINE:
                unmatched.append(m.group())
        return unmatched


def extract_gemma_medicine_candidates(
    raw_text: str,
    medicines: list[dict[str, Any]],
) -> MedicineExtractionResult:
    """Create review candidates only for Gemma names found verbatim in its transcription."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Input OCR text is empty.")
    if len(raw_text) > MAX_OCR_TEXT_LENGTH:
        raise ValueError(
            f"OCR text exceeds maximum allowed length of {MAX_OCR_TEXT_LENGTH} characters."
        )

    candidates: list[ExtractedMention] = []
    prescription_lines: list[PrescriptionLineCandidate] = []
    used_spans: set[tuple[int, int]] = set()
    unmatched_names = 0

    for medicine in medicines:
        name = medicine.get("medicine_name")
        if not isinstance(name, str) or not name.strip():
            continue

        match = next(
            (
                found
                for found in re.finditer(re.escape(name.strip()), raw_text, re.IGNORECASE)
                if (found.start(), found.end()) not in used_spans
            ),
            None,
        )
        if match is None:
            unmatched_names += 1
            continue

        start, end = match.span()
        used_spans.add((start, end))
        line_start = raw_text.rfind("\n", 0, start) + 1
        line_end = raw_text.find("\n", end)
        if line_end == -1:
            line_end = len(raw_text)
        line_number = raw_text.count("\n", 0, start) + 1
        line = raw_text[line_start:line_end].strip()
        candidate = ExtractedMention(
            text=raw_text[start:end],
            candidate_type=CandidateType.MEDICINE_NAME,
            start_char=start,
            end_char=end,
            line_number=line_number,
            extraction_method=ExtractionMethod.GEMMA_STRUCTURED,
            evidence_context=line,
            heuristic_score=0.5,
        )
        candidates.append(candidate)
        prescription_lines.append(
            PrescriptionLineCandidate(
                line_number=line_number,
                raw_line=line,
                is_likely_prescription_line=True,
                medicine_name_candidate=candidate,
            )
        )

    warnings = [
        "Medicine-name candidates were proposed by Gemma and are not clinically verified.",
        "Review every name, dosage, and instruction against the original image.",
        "Only medicine names found verbatim in Gemma's transcription are included in the review session.",
    ]
    if unmatched_names:
        warnings.append(
            f"{unmatched_names} Gemma medicine name(s) could not be matched to the transcription "
            "and were omitted from the review session."
        )

    lines = raw_text.splitlines()
    return MedicineExtractionResult(
        raw_text=raw_text,
        line_count=len(lines),
        candidates=candidates,
        prescription_lines=prescription_lines,
        review_status="pending_review",
        warnings=warnings,
        metadata={
            "extractor_version": "gemma-structured-1.0.0",
            "source": "gemma_structured_output",
            "unmatched_gemma_medicine_names": unmatched_names,
            "total_candidates_found": len(candidates),
        },
    )
