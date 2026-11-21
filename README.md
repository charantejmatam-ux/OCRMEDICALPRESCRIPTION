# Medical Prescription OCR

Medical Prescription OCR is a prototype for reading prescription images and
looking up medicine information. It combines a Streamlit user interface with a
FastAPI backend. OCR and language-model output can be incomplete or incorrect:
the application is for experimentation and must not be used to make medical
decisions.

## What the project does

- Accepts a prescription image by upload or clipboard paste.
- Extracts candidate prescription text and medicine details using the configured
  OCR engine (Gemma, PaddleOCR, or optional Gemini).
- Searches DailyMed for product labels for a detected medicine and can ask
  Gemma questions grounded in a selected label.
- Provides medicine Q&A about medicines found in the uploaded prescription.
- Includes an **Advanced Review** page for reviewing OCR candidates and
  terminology normalization before the optional interaction-evidence workflow.

The Advanced Review interaction workflow has no evidence provider configured
by default, so it reports `not_configured`; it does not perform an interaction
check. OCR results and medicine names are suggestions, not verified facts.

## Requirements

- Python 3.11 or newer
- Git
- Windows PowerShell for the commands below (other operating systems can use
  their equivalent virtual-environment activation commands)
- For local Gemma inference: [Ollama](https://ollama.com/) and the configured
  Gemma model

## Clone and install

Run these commands in PowerShell:

```powershell
git clone https://github.com/charantejmatam-ux/OCRMEDICALPRESCRIPTION.git
cd OCRMEDICALPRESCRIPTION

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell prevents virtual-environment activation, either allow activation
for the current terminal with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or invoke the
environment's Python executable directly as shown in the run commands below.

## Configure OCR and optional services

The example `.env` defaults to Gemma. Set `GEMMA_API_KEY` in `.env` to use the
configured cloud Gemma service. Without that key, the application uses local
Ollama; install and start Ollama, then download the model:

```powershell
ollama pull gemma3:4b
```

You can change the local model with `OLLAMA_GEMMA_MODEL`, the Ollama server
address with `OLLAMA_BASE_URL`, and the default OCR engine with `OCR_ENGINE`.
PaddleOCR is the local OCR alternative. Gemini OCR is optional and requires a
`GEMINI_API_KEY`; it is used only when explicitly selected. Review `.env.example`
for the available settings. Restart the backend after changing environment
variables.

Do not commit `.env` or put API keys in frontend code. Local Ollama and
PaddleOCR process images on your machine. Selecting a cloud provider sends
prescription image or medicine-question data to that provider. Use only data
you have permission to process.

## Run the application

Start the backend in the first PowerShell terminal, from the project directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
```

In a second terminal, from the same project directory, start the Streamlit UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/Home.py
```

Open the frontend at <http://localhost:8501>. The API health check is at
<http://localhost:8000/health>, and interactive API documentation is at
<http://localhost:8000/docs>.

## Use the application

1. Open **Medicine Prescription Search** in the Streamlit app.
2. Upload a prescription image or paste one from the clipboard.
3. Select **Search medicines** and wait for OCR and extraction to complete.
4. Review the extracted text and candidate medicine details against the image.
5. Search for a medicine's DailyMed labels, select the matching product label,
   and ask for label-based information when available.
6. Use the medicine Q&A section for general questions about the extracted
   medicines. It does not provide personalized treatment advice.
7. Open **Advanced Review** in the Streamlit sidebar for the separate
   candidate-review and terminology-normalization workflow.

## Run tests

With the virtual environment active:

```powershell
pytest backend/tests gemma_integration/tests -v
```

## Project layout

```text
backend/
  app/
    main.py                 FastAPI application
    routers/                Prescription and medicine-chat endpoints
    services/               OCR, image handling, review, and related services
  tests/                    Backend tests
frontend/
  Home.py                   Main Streamlit app
  pages/                    Additional Streamlit pages
gemma_integration/           Gemma extraction integration and tests
images/                      Sample images used for local evaluation
requirements.txt             Python dependencies
.env.example                 Example configuration
```

## Privacy and safety

Prescription recognition can misread medicine names, strengths, and
instructions. Always verify the result against the original document and ask a
qualified healthcare professional about medicines or treatment. This
prototype is not a medical device or clinical decision-support system.

Uploaded images are processed in memory by default. Persistent image storage
is disabled unless explicitly enabled in `.env`. Review sessions and related
workflow data are temporary and may be lost when the backend restarts.

## License

This project declares the MIT license in `pyproject.toml`.
