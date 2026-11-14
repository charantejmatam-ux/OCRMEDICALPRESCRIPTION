# Medical Prescription OCR & Drug Interaction Checker

An academic prototype that extracts candidate text from medical prescriptions,
requires explicit human review, and can report source-backed interaction lookup
results only when a genuine evidence provider is configured.

> **Status:** Phases 1â€“6 workflow implemented; PaddleOCR, Gemma (API/Ollama), and configured Gemini are selectable OCR engines. No drug-interaction evidence
> provider is configured; the default report explicitly returns `not_configured`.

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| **1** | Project scaffold, FastAPI `/health`, Streamlit homepage | âœ… Done |
| **2** | PaddleOCR, Gemma API/Ollama, optional Gemini recognition, image upload & processing router | âœ… Done |
| **3** | Rule-based medicine candidate mention & attribute extraction | âœ… Done |
| **4** | Candidate uncertainty evidence, mandatory human review, and workflow eligibility gate | âœ… Done |
| **5** | Safe medicine terminology normalization with separate human review | âœ… Done |
| **6** | Accepted-concept-gated interaction evidence lookup and report | âœ… Workflow done; provider not configured |

---

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”        HTTP        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Streamlit Frontend    â”‚ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶ â”‚    FastAPI Backend       â”‚
â”‚   frontend/Home.py      â”‚                    â”‚    backend/app/main.py   â”‚
â”‚   (port 8501)           â”‚ â—€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ â”‚    (port 8000)           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     JSON resp.     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                              â”‚
                                               (Phase 2+) OCR Engine
                                               (Phase 3+) NER / Drug NLP
                                               (Phase 4) In-memory review session & safety gate
                                               (Phase 5) Provider-backed terminology review
                                               (Phase 6) Replaceable provider; disabled by default
```

---

## Prerequisites

The Home screen starts with the title and simultaneous prescription image
upload and clipboard-paste controls. **Search medicines** uses Gemma only; the
uploaded image, pasted image, or both are processed in sequence. Detected
medicines are listed in order with the fields read from the prescription.
Search an individual medicine to find DailyMed product labels, select the exact
label, and ask Gemma for label-grounded information. The former multi-phase
review interface remains available as **Advanced Review**.

- Python 3.11+
- Windows PowerShell (or any shell)

---

## Quick Start

### 1 â€” Clone & enter the project

```powershell
git clone <your-repo-url>
cd final
```

### 2 â€” Create & activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 â€” Install dependencies

```powershell
pip install -r requirements.txt
```

### 4 â€” Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env if needed
```

### 5 â€” Run the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8000
```

`--reload-dir backend` keeps Uvicorn's file watcher out of `.venv`; otherwise
package installation or extraction there can trigger repeated server restarts.

API docs available at: <http://localhost:8000/docs>

The backend loads `.env` before importing services that read environment
settings. `CORS_ORIGINS` is parsed as a comma-separated list with whitespace
trimmed; credentialed CORS requests are disabled because this prototype does
not use browser credentials.

### 6 â€” Run the frontend (new terminal, venv activated)

```powershell
streamlit run frontend/Home.py
```

Frontend available at: <http://localhost:8501>

### Local handwriting OCR selection

The application prefers Gemma through its API or local Ollama, then selects
PaddleOCR PP-OCRv6 medium if Gemma is unavailable. Gemini is selectable only
when its API key is configured. Gemma supplies the transcription and structured
medicine names/instructions; when Gemma is selected, only its structured
medicine-name findings are used to create review candidates. Verify all output
against the image. PaddleOCR performs local full-page text detection and
recognition, which matched these prescription images better than the existing
line-only TrOCR pipeline. PaddleOCR uses CPU inference and local model files;
it does not download weights while handling a prescription.

PaddleOCR and PaddlePaddle are installed from `requirements.txt`. The first
PaddleOCR CLI run downloads its detector and recognizer into the standard local
cache. For example, run `paddleocr ocr -i .\images\sample_line.jpeg` once
while setting up the environment. The app then uses the cached PP-OCRv6 medium
models. If they are stored elsewhere, set `PADDLEOCR_DET_MODEL_PATH` and
`PADDLEOCR_REC_MODEL_PATH` in `.env` to the directories containing
`inference.yml`, `inference.json`, and `inference.pdiparams`.

`OCR_ENGINE=gemma` is the preferred default. The OCR selector automatically
prefers a ready Gemma engine, then a ready PaddleOCR engine; Gemini is never
selected automatically as a fallback because it sends images to Google.
Configure a Gemma API key or install the configured local Ollama model to use
Gemma. PaddleOCR is lazy-loaded, uses CPU with
oneDNN disabled for Windows compatibility, and reports missing local models as
not configured rather than downloading them. Prescription images stay in
memory and are not sent to a service by default. Select **GEMINI** in the OCR
engine selector to send an image to Google's Gemini API for transcription;
PaddleOCR remains local. In one Gemini API request, Gemini returns both the
verbatim transcription and a separate list of possible medicine-name
candidates; the UI labels the candidates as unverified. The transcription can
also be sent through the existing candidate extraction workflow to identify
possible attributes. Configure `GEMINI_API_KEY`
in the backend `.env` file and optionally set `GEMINI_MODEL`. The API key is
used by the backend request only; never place it in browser/frontend code or
commit it. Both backend and frontend resolve `.env` from the project root, so
starting either process from another working directory does not lose settings.
Restart the backend after changing the key. The OCR panel can create a review
session in the same request as OCR. With Gemma selected, candidate names come
from Gemma's structured image analysis; other OCR engines use the existing
text-based candidate extractor. Results still go through the normal
human-review steps.

Select **GEMMA** to send the prescription image directly to the Gemma API and
receive transcription, medicine names, dosage, frequency, duration, and
instructions as structured JSON. If `GEMMA_API_KEY` is not set, Gemma runs
locally through Ollama without an API key. The cloud service is primary when a
key is configured; network, quota, and server failures fall back to the
configured local model. Install it with `ollama pull gemma3:4b`; override the
model or local server with `OLLAMA_GEMMA_MODEL` or `OLLAMA_BASE_URL`. Inference
may take several minutes on CPU; `GEMMA_TIMEOUT_SECONDS` defaults to 600.
Gemma results are unverified and remain subject to the existing human-review
workflow.

The same optional `GEMMA_API_KEY` also enables cloud Gemma answers in the
medicine chatbot. Medicine-name search remains a DailyMed lookup; Gemma answers
from the selected DailyMed label. The question and relevant label sections are
sent to Google when cloud Gemma is used. Without a key, chat uses local Ollama;
temporary cloud failures fall back to Ollama. See
[README_Medicine_Chatbot.md](README_Medicine_Chatbot.md) for API-key setup and
the data flow.

The optional **Gemini second opinion** checkbox remains available when PaddleOCR
is selected. Gemma, Gemini, or PaddleOCR output does not bypass the existing
human-review workflow. Do not send prescription data without appropriate
permission and approval for external processing.
The Gemma and Gemini engine readiness details and OCR selector show the
configured provider/model.
If Google explicitly reports that a configured model is retired, the backend
retries once with the model named in Google's response (or the current default).

OCR inference runs in a worker thread so a slow prediction does not block
backend health checks. The first PaddleOCR image may take longer because its
models load on demand. The frontend allows up to 300 seconds for PaddleOCR,
240 seconds for Gemini (the backend allows 180 seconds for Google's response),
540 seconds for PaddleOCR with the Gemini second opinion, and 660 seconds for
Gemma. Gemini timeouts
return HTTP 504 with a retry/use-Paddle message instead of a generic server error.

### Local comparison on `images/`

I ran PaddleOCR, TrOCR, and the prescription-specific Donut model on the same
six images. Paddle produced the most complete page text on four images; it was
still weak on two difficult samples. TrOCR returned only a small fraction of
page text because the existing row-projection segmenter missed most lines.
Donut recognized some printed headings and parts of one prescription, but
repeated punctuation across most outputs. Tesseract could not be run because
its executable is not installed. PaddleOCR remains the default local engine;
Gemma and Gemini are optional engines and were not part of this local comparison.

This is a qualitative comparison, not a character-error-rate benchmark: the
folder has no manually corrected reference transcripts. PaddleOCR still
misreads medicine names, strengths, and dosage instructions, so its output is
not safe to use without human verification.

### 7 — Run tests

```powershell
pytest backend/tests -v
```
## Phase 4: Candidate Review

Each extracted candidate starts in `pending_review`. The reviewer must explicitly
choose one action for each candidate:

| Action | Resulting state | Can pass the workflow gate? |
|---|---|---|
| `confirm` | `confirmed` | Yes, using the original candidate text |
| `correct` | `corrected` | Yes, using the separately stored correction |
| `reject` | `rejected` | No |
| `mark_uncertain` | `uncertain` | No |

Review actions are one-way from `pending_review`; repeated or conflicting actions
return HTTP 409. Scores never approve candidates. The displayed `heuristic_score`
is an uncalibrated rule weight, not a probability, and has not been validated on
a labeled dataset. OCR confidence is `null`/unavailable unless a usable value is
actually supplied; raw-text extraction currently has no per-candidate OCR confidence.

The reusable integration point for normalization and any future downstream consumer is
`get_eligible_candidates_for_downstream` in
`backend/app/services/review_workflow.py`. Normalization consumes this function's
results; future downstream or interaction logic must do the same, never raw extracted candidates.
Passing the gate is a workflow control only; it does not prove medicine identity
or clinical correctness. Phase 4 does not itself implement drug-interaction
checking or treatment recommendations; the separate Phase 5 normalization
workflow is described below.

### Review API examples

Extraction creates a review session and returns its `review_id` along with the
candidate evidence:

```http
POST /prescriptions/extract-medicines
Content-Type: application/json

{"ocr_text":"Amoxicillin 500mg capsule PO"}
```

```json
{
  "review_id": "36d1d420-7b35-49d4-bbbd-7732c2350b30",
  "review_status": "pending_review",
  "candidates": [
    {
      "candidate_id": "73715483-a14d-429c-9eaf-f38a77ac165c",
      "text": "Amoxicillin",
      "heuristic_score": 0.9,
      "ocr_engine_confidence": null,
      "review_status": "pending_review"
    }
  ]
}
```

Retrieve the review state and full original evidence:

```http
GET /prescriptions/review/36d1d420-7b35-49d4-bbbd-7732c2350b30
```

Record one explicit candidate action:

```http
POST /prescriptions/review/36d1d420-7b35-49d4-bbbd-7732c2350b30/candidates/73715483-a14d-429c-9eaf-f38a77ac165c
Content-Type: application/json

{"action":"correct","corrected_text":"Amoxicillin"}
```

The correction is stored separately from both the original candidate and full
original OCR text. A correction must be non-empty and at most 500 characters.
Actions are associated with candidate IDs inside the identified review session;
unknown or cross-session candidates return 404. Invalid action payloads return
422.

### Review UI and limitations

The Streamlit interface separates pending from completed candidates, displays
original evidence, extraction method and flags, explains heuristic scores, and
offers confirm, correct, reject, and uncertain actions individually. Corrections
show the original and reviewer-entered value side by side. The prototype does not
collect reviewer identity or credentials and does not verify reviewer
qualifications.

Review sessions are held only in process memory, expire after one hour of
inactivity, are bounded to 100 sessions, and may be lost when the service restarts.
No patient record or review decision is persisted. Deployments with multiple
backend workers have separate in-memory stores and therefore are not suitable for
shared review sessions.

**Research prototype only. Extracted medicine candidates may be wrong. Review
status does not establish clinical correctness. Do not use this tool to make
treatment decisions.**

---

## Phase 5: Safe Medicine Normalization

The normalization endpoint accepts only candidates present in
`get_eligible_candidates_for_downstream` output. Phase 4 `confirmed` candidates
are looked up using their original extracted text; `corrected` candidates use
the separately stored reviewer correction. The original OCR text, candidate
text, offsets, line, extraction method, and evidence context are copied unchanged
into the normalization record. Strength, dosage form, route, and frequency are
parsed separately from the original evidence line when present.

Each proposed concept includes its source identifier, source-provided name and
attributes, exact/fuzzy/partial match method, evidence, and source name/version/
release metadata. The provider owns retrieval and matching; the application does
not autocorrect unknown names or generate local substitute concepts. Differences
between parsed strength/form and concept attributes are shown explicitly and
require an explicit mismatch acknowledgement before acceptance. A source may
return zero, one, or multiple concepts. Zero results are `unmatched`; multiple
results are `ambiguous`; a single result is `pending_review`. None is accepted
automatically. Acceptance, rejection, or leaving a result unresolved is a
separate, one-time normalization decision and does not alter Phase 4 review.

### Terminology provider and provenance

Medicine-name extraction uses the name position and generic strength/form/route
boundaries; it does not use a built-in medicine-name list or autocorrect OCR.
Possible names remain unverified until a reviewer confirms or corrects them.

After that explicit review, the normalization button sends only the reviewed
medicine text to the U.S. National Library of Medicine's RxNav approximate-term
API. The provider returns at most five source-backed RxNorm suggestions with
RxCUI identifiers, the API's native rank and score, and available source details.
RxNav scores rank candidates for that query. They are not probabilities of a
correct medicine identification. No remote call is made during OCR or candidate
extraction. Network failures result in an unmatched response; they do not imply
that the medicine is absent from RxNorm. Service failures return an unavailable
response instead of a false no-match. Terminology records are not
drug-interaction evidence.

Each returned concept records its terminology source identifier and provider
provenance. Attribute mismatches are surfaced and require explicit reviewer
acknowledgement before acceptance. `UnavailableRxNormProvider` remains available
for offline deployments and tests.

### Normalization API

Request a normalization record only after the candidate has passed Phase 4:

```http
POST /prescriptions/review/{review_id}/candidates/{candidate_id}/normalization
```

The response contains the distinct original/reviewed input evidence, parsed
attributes, provider status and provenance, proposed concepts, and normalization
review state. Retrieve it with:

```http
GET /prescriptions/normalization/{normalization_id}
```

Accept a particular proposed source concept explicitly:

```http
POST /prescriptions/normalization/{normalization_id}/decision
Content-Type: application/json

{"action":"accept","selected_concept_id":"<source-provided identifier>"}
```

Other supported actions are `reject`, `mark_ambiguous` (requires multiple
concepts), and `mark_unmatched`. An accepted concept with strength or form
mismatches also requires `"acknowledge_attribute_mismatches": true`. Invalid
decisions return HTTP 422, repeated decisions return HTTP 409, unknown records
return HTTP 404, and a candidate that has not passed the Phase 4 gate returns
HTTP 409.

Normalization records are bounded in-memory data (500 records, one-hour
inactivity expiry). They are lost on service restart and are not shared across
backend worker processes. The Streamlit UI does not offer bulk acceptance; the
reviewer must choose a proposed concept and explicitly accept it. Synthetic
terminology records used in tests are labeled as synthetic and must never be
treated as real RxNorm entries.

No interaction provider is configured, so the Phase 6 API does not perform
network requests. The interaction report workflow and provider contract are
implemented, but a `not_configured` report is not an interaction assessment.

---

## Phase 6: Source-Backed Interaction Evidence

The interaction checker accepts only Phase 5 normalization record IDs. It
retrieves each record from the in-memory normalization store and verifies the
explicit `accepted` state, `accept` action and timestamp, that the accepted
concept is among the record's proposed concepts, and the recorded Phase 4
decision. A rejected, pending, ambiguous, unmatched, missing, or inconsistent
normalization is blocked. Arbitrary medicine strings and client-supplied
terminology IDs are not accepted by the interaction API.

Each unique unordered pair of accepted standardized concepts is requested once.
The provider receives accepted concept identifiers and provenance, never the
OCR image, raw prescription text, patient name, date of birth, or other patient
identifier. Reports retain accepted concept references, both review decisions,
pair-level results, and any source evidence/provenance.

### Provider and source status

There is **no interaction evidence provider or dataset configured in this
repository**. Network access is disabled by default; `UnavailableInteractionProvider`
returns pair-level `not_configured` results without making a lookup. RxNorm/RxNav
terminology is not pairwise interaction evidence. The official
[RxNav API documentation](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html)
describes RxNorm terminology APIs. The
[DailyMed web services](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm)
and [openFDA drug-label API](https://open.fda.gov/apis/drug/label/) expose
labeling data, but these should not be treated as a complete, verified
pairwise-interaction database. Access conditions, license, coverage, identifier
support, provenance, source updates, and intended use must be checked before
enabling a future provider. No large dataset is downloaded or remote API
queried automatically. A proprietary source's availability or license has not
been verified for this project.

To configure a genuine source in a future change, implement the replaceable
`InteractionProvider` interface in
`backend/app/services/drug_interactions.py`, validate its pair response and
coverage semantics, record its source/version/provenance, and explicitly wire
the configured instance into the prescription router. Do not treat a terminology
provider or locally seeded vocabulary as interaction evidence. Synthetic
interaction records are confined to tests and are not production data.

### Report statuses and interpretation

| Status | Meaning |
|---|---|
| `interaction_found` | The configured source returned a matching evidence record for at least one pair. This is source-reported evidence, not a clinical interpretation. |
| `no_matching_record_found` | The configured source reported no matching record for every checked pair with complete pair coverage. This does **not** establish safety. |
| `source_unavailable` | Every attempted pair lookup failed because the configured source was unavailable. |
| `not_configured` | No provider is configured; no pair lookup was made. |
| `incomplete` | At least one required pair or source coverage result was incomplete, unavailable, malformed, or otherwise not safely interpretable. |
| `blocked` | The request had no usable accepted concepts, or one or more requested normalization records failed the Phase 5 acceptance gate. |

`checked_pair_count` counts attempted pair lookups; it is zero when no provider
is configured. The report separately returns `expected_pair_count`, every
pair-level status, evidence records, source-provided severity/classification
(or `null` when not supplied), source version/release/retrieval details when
available, and limitations. A partial lookup is not presented as a complete
check. A missing record, unsupported identifier, malformed response, timeout,
or provider error is never converted into a no-match result.

### Interaction API

Submit only explicit Phase 5 normalization record IDs (between 1 and 20):

```http
POST /prescriptions/interactions/check
Content-Type: application/json

{"normalization_ids":["<accepted-normalization-uuid-1>","<accepted-normalization-uuid-2>"]}
```

The response contains a `report_id`, overall status, selected accepted concepts,
required and attempted pair counts, and each pair result. Retrieve it with:

```http
GET /prescriptions/interactions/reports/{report_id}
```

Malformed IDs, empty/oversized requests, duplicate normalization IDs, and
duplicate accepted concept identifiers are rejected with HTTP 422. Missing or
expired normalization records and reports return HTTP 404. A valid request
containing candidates that did not pass Phase 5 acceptance returns a `blocked`
report; they are never sent to the interaction provider.

### Privacy, retention, and safety limits

Interaction reports are held in a bounded process-local store (up to 100 reports
with one-hour inactivity expiry). Reports and normalization/review sessions are
lost on restart and are not shared across backend workers. A provider adapter,
if configured later, must be reviewed before use and must not transmit patient
identifiers or prescription images. The current default makes no external
requests.

The Streamlit interface allows selection only from accepted Phase 5 concepts,
displays report status and pair-by-pair coverage, source evidence,
source-provided severity only, provenance, and limitations. It does not offer a
bulk normalization acceptance action. No dose/treatment recommendations,
medicine substitutions, personalized risk predictions, interaction severity
interpretation, or start/stop instructions are implemented. The prototype is
not clinical decision support and must not be used to make treatment decisions.

---

## Project Structure

```
final/
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ app/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ main.py               # FastAPI application
â”‚   â”‚   â”œâ”€â”€ routers/
â”‚   â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”‚   â””â”€â”€ prescriptions.py  # Upload, OCR, review, normalization & evidence APIs
â”‚   â”‚   â””â”€â”€ services/
â”‚   â”‚       â”œâ”€â”€ __init__.py
â”‚   â”‚       â”œâ”€â”€ image_processing.py # Image validation & in-memory buffering
â”‚   â”‚       â”œâ”€â”€ medicine_extraction.py # Phase 3 candidate extraction engine
â”‚   â”‚       â”œâ”€â”€ review_workflow.py # Phase 4 review lifecycle & safety gate
â”‚   â”‚       â”œâ”€â”€ medicine_normalization.py # Phase 5 provider & normalization review
â”‚   â”‚       â”œâ”€â”€ drug_interactions.py # Phase 6 provider contract & evidence reports
â”‚   â”‚       â”œâ”€â”€ gemini_second_opinion.py # Optional, explicit Gemini comparison
â”‚   â”‚       â””â”€â”€ ocr/
â”‚   â”‚           â”œâ”€â”€ __init__.py   # Engine registry & factory
â”‚   â”‚           â”œâ”€â”€ base.py       # Abstract OCREngine & data models
â”‚   â”‚           â”œâ”€â”€ paddle_engine.py # Local PaddleOCR page detection and recognition
â”‚   â”‚           â””â”€â”€ gemini_engine.py # Optional Google Gemini image transcription
â”‚   â””â”€â”€ tests/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ test_health.py        # Tests for /health endpoint
â”‚       â”œâ”€â”€ test_prescriptions.py # Tests for upload & OCR services
â”‚       â”œâ”€â”€ test_medicine_extraction.py # Tests for candidate extraction
â”‚       â”œâ”€â”€ test_review_workflow.py # Tests for review and eligibility gate
â”‚       â”œâ”€â”€ test_medicine_normalization.py # Tests for normalization review
â”‚       â””â”€â”€ test_drug_interactions.py # Synthetic-only interaction provider tests
â”œâ”€â”€ frontend/
â”‚   â””â”€â”€ Home.py                   # Streamlit review, normalization & evidence UI
â”œâ”€â”€ data/                         # Local patient data â€” never committed
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ .env.example
â”œâ”€â”€ .gitignore
â””â”€â”€ README.md
```

---

## Privacy & Data Handling

- **Zero-Persistence Default**: Prescription images are strictly processed **in memory by default** (`PERSIST_IMAGES=false`). No images are written to disk unless persistent storage is explicitly enabled in `.env`.
- **Upload Protection**: File size is capped at **10 MB** (enforced chunk-by-chunk during upload streaming) with decompression-bomb limits (`Image.MAX_IMAGE_PIXELS = 25,000,000`) and dimension limits (max 8,000 x 8,000 px).
- **Image Format Detection**: The image format is determined from decoded file content, not the filename suffix. A supported image whose name ends in `.jpeg` but contains WEBP data is accepted and reported as WEBP; if persistence is enabled, it is stored with a `.webp` suffix.
- **No Path Leakage**: Internal filesystem paths are never returned in public API payloads.
- **Local Isolation**: When persistence is opted in (`PERSIST_IMAGES=true`), files are saved under `data/`, which is ignored by `.gitignore`.
- **External Processing Is Opt-In**: PaddleOCR and local Ollama remain local. Prescription images are sent to a cloud Gemma/Gemini service only when selected in the UI or when the Gemini second opinion is explicitly enabled.
- **Ephemeral Review Data**: Candidate text and review actions are held in memory only and are lost on service restart or session expiry.
- **Ephemeral Normalization Data**: Proposed terminology matches and decisions are in memory only and are lost on restart or expiry; RxNav is queried only after the explicit reviewed-candidate action.
- **Ephemeral Interaction Reports**: Pair results are held in memory only, expire, and are not shared between backend worker processes; no interaction provider is configured and no external interaction lookup is performed.

---

## License

MIT
#   O C R M E D I C A L P R E S C R I P T I O N  
 