# Standalone Gemma prescription integration

This package is independent of the existing backend and frontend. It uses two
Gemma passes: the first transcribes the prescription image, and the second
receives only that transcription to identify prescribed medicines and
instructions. Medicine names must be supported by the extracted text. If the
cloud API is unavailable or over quota (`429`), the pipeline falls back to
local Gemma through Ollama.

The output includes `extracted_text`, `medicines` (medicine name, dosage,
frequency, duration, and medicine-specific instructions),
`medical_instructions`, the OCR and medicine-stage provider/model, and
`human_review_required: true`. It is extraction assistance, not medical advice;
verify every result against the original prescription before use.

## Configure and run (PowerShell)

Install dependencies into the project's existing environment or another
Python 3.11+ environment:

```powershell
python -m pip install -r gemma_integration\requirements.txt
```

Set a Google AI Studio API key for the primary Gemma API:

```powershell
$env:GEMMA_API_KEY = "your-api-key"
```

The defaults are `gemma-3-27b-it` for the API, `http://localhost:11434` for
Ollama, `gemma3:4b` locally, and a 600-second timeout per model request. The
two-stage flow can therefore take longer than 600 seconds. Local inference can
take several minutes, especially the first time the model loads. Set
`GEMMA_TIMEOUT_SECONDS` to adjust the timeout. Set `GEMMA_API_MODEL`,
`GEMMA_API_BASE_URL`, `OLLAMA_BASE_URL`, or `OLLAMA_GEMMA_MODEL` to override
the other defaults. For local fallback, install and start
Ollama and make the configured image-capable Gemma model available, for example:

```powershell
ollama pull gemma3:4b
```

From the project root, run:

```powershell
python -m gemma_integration.run ".\images\handwritten-prescription-1568x1254.jpg"
```

The command prints the structured JSON. If the cloud key is not set, it uses
Ollama directly. If both services fail, it reports an error instead of
returning a partial/success-shaped result.

## Independent tests

The tests use an in-memory image and mock the API and Ollama HTTP responses, so
they validate the two-stage image-to-text-to-medicines flow and fallback
behavior without calling a live service or sending prescription data externally:

```powershell
python -m pytest gemma_integration\tests -q
```
