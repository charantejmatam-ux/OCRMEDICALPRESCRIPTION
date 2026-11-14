"""
frontend/Home.py
Streamlit UI for Medical Prescription OCR & Drug Interaction Checker.

Phase 1:
- Live backend connection & system health probe

Phase 2:
- In-memory prescription image upload/paste and raw OCR text extraction
- Pasted prescription text can skip OCR and go to candidate extraction

Phase 3:
- Layout-based extraction of medicine candidate mentions
- Structured prescription line grouping (name, strength, form, frequency)
- Verbatim text preservation with strict unverified / human-review flags

Phase 4:
- Explicit per-candidate human-review actions and visible audit state
- Heuristic and OCR confidence are displayed as distinct evidence

Phase 5:
- Source-backed terminology proposals are shown separately from Phase 4 review
- A proposed concept requires a separate explicit normalization decision

Phase 6:
- Pairwise interaction evidence lookup is gated on accepted Phase 5 concepts
- No interaction evidence provider is configured by default
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import dotenv_values

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_dotenv_config = dotenv_values(PROJECT_ENV_FILE, interpolate=False)

BACKEND_URL: str = (
    os.getenv("BACKEND_URL")
    or _dotenv_config.get("BACKEND_URL")
    or "http://localhost:8000"
)
DEFAULT_OCR_ENGINE: str = (
    os.getenv("OCR_ENGINE") or _dotenv_config.get("OCR_ENGINE") or "gemma"
).strip().lower()
del _dotenv_config

MAX_PASTED_IMAGE_BYTES = 10 * 1024 * 1024
_CLIPBOARD_IMAGE_INPUT = components.declare_component(
    "rx_clipboard_image_input",
    path=Path(__file__).resolve().parents[1] / "components" / "clipboard_image",
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Prescription OCR",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --rx-ink: #173042;
        --rx-teal: #087f82;
        --rx-teal-dark: #06676a;
        --rx-line: #dce8e9;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 92% 0%, rgba(196, 235, 228, .36), transparent 34rem),
            #f4f8f8;
        color: var(--rx-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 248, 248, .88);
    }

    [data-testid="stMain"] .block-container {
        max-width: 1440px;
        padding: 2rem 2.5rem 4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102d3a 0%, #173d48 100%);
    }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #eaf5f3;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(224, 242, 240, .18);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #bdd2d2;
    }

    h1, h2, h3 {
        color: var(--rx-ink);
        letter-spacing: -.025em;
    }

    h2 {
        margin-top: 1.3rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--rx-line);
        border-radius: 16px;
        background: rgba(255, 255, 255, .86);
        box-shadow: 0 8px 26px rgba(24, 57, 67, .045);
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed #91b9b7;
        border-radius: 14px;
        background: #f7fbfa;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--rx-teal);
        background: #f0f8f6;
    }

    .stButton > button {
        min-height: 2.75rem;
        border-radius: 10px;
        font-weight: 650;
        transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
    }

    .stButton > button[kind="primary"] {
        border-color: var(--rx-teal);
        background: var(--rx-teal);
        color: #ffffff;
        box-shadow: 0 5px 14px rgba(8, 127, 130, .18);
    }

    .stButton > button[kind="primary"]:hover {
        border-color: var(--rx-teal-dark);
        background: var(--rx-teal-dark);
        box-shadow: 0 7px 18px rgba(8, 127, 130, .24);
        transform: translateY(-1px);
    }

    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        border-radius: 10px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stAlert"] *,
    [data-testid="stMain"] [data-testid="stStatusWidget"] * {
        color: #173042 !important;
    }

    [data-testid="stMain"] input:not([type="checkbox"]):not([type="radio"]),
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        border-color: #91a7ad;
        background-color: #ffffff !important;
        color: #173042 !important;
    }

    [data-testid="stMain"] .stButton > button {
        color: #173042 !important;
    }

    [data-testid="stMain"] .stButton > button[kind="primary"],
    [data-testid="stMain"] .stButton > button[kind="primary"] * {
        color: #ffffff !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--rx-line);
        border-radius: 12px;
        overflow: hidden;
    }

    .rx-hero {
        position: relative;
        overflow: hidden;
        margin: .25rem 0 1.5rem;
        padding: clamp(1.6rem, 4vw, 3rem);
        border: 1px solid rgba(206, 229, 226, .9);
        border-radius: 24px;
        background: linear-gradient(118deg, #ffffff 0%, #f5fbf9 58%, #e8f5f1 100%);
        box-shadow: 0 16px 42px rgba(24, 57, 67, .07);
    }

    .rx-hero::after {
        position: absolute;
        top: -5.2rem;
        right: -3.2rem;
        width: 18rem;
        height: 18rem;
        border: 1px solid rgba(8, 127, 130, .12);
        border-radius: 50%;
        box-shadow: 0 0 0 2.6rem rgba(8, 127, 130, .035),
                    0 0 0 5.2rem rgba(8, 127, 130, .025);
        content: "";
        pointer-events: none;
    }

    .rx-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        margin-bottom: .85rem;
        padding: .38rem .72rem;
        border: 1px solid #cce5df;
        border-radius: 999px;
        background: #edf8f4;
        color: #176d69;
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .rx-hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #173746;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 750;
        letter-spacing: -.055em;
        line-height: 1.08;
    }

    .rx-hero h1 span {
        color: var(--rx-teal);
    }

    .rx-hero p {
        position: relative;
        z-index: 1;
        max-width: 790px;
        margin: 1rem 0 0;
        color: #536d79;
        font-size: 1rem;
        line-height: 1.7;
    }

    @media (max-width: 700px) {
        [data-testid="stMain"] .block-container {
            padding: 1.2rem 1rem 3rem;
        }

        .rx-hero {
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for OCR text
if "raw_ocr_text" not in st.session_state:
    st.session_state["raw_ocr_text"] = ""
if "input_ocr_text" not in st.session_state:
    st.session_state["input_ocr_text"] = st.session_state["raw_ocr_text"]
if "clipboard_image_reset_id" not in st.session_state:
    st.session_state["clipboard_image_reset_id"] = 0
if "review_id" not in st.session_state:
    st.session_state["review_id"] = None
if "normalization_ids" not in st.session_state:
    st.session_state["normalization_ids"] = {}
if "interaction_report_id" not in st.session_state:
    st.session_state["interaction_report_id"] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💊 Rx OCR")
    st.caption("Medical Prescription OCR & Drug Interaction Checker")
    st.divider()
    st.markdown(
        """
        **Workflow Steps**
        - 🏠 **Phase 1**: Scaffold & Health Probe
        - 📷 **Phase 2**: Image Upload/Paste & Raw OCR
        - 🔬 **Phase 3**: Candidate Extraction *(Done)*
        - 🧑‍⚕️ **Phase 4**: Mandatory Candidate Review *(Done)*
        - 🧾 **Phase 5**: Safe Medicine Normalization
        - ⚠️ **Phase 6**: Source-Backed Interaction Evidence *(Research only)*
        """
    )
    st.divider()
    st.caption(
        "🔒 PaddleOCR and local Gemma/Ollama run locally. Gemini uses Google; "
        "Gemma uses its API when configured and falls back to local Ollama."
    )


# ---------------------------------------------------------------------------
# Backend status helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5)
def _check_backend(url: str) -> dict[str, Any]:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _fetch_ocr_engines(
    url: str,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    try:
        r = requests.get(f"{url}/prescriptions/engines", timeout=3)
        r.raise_for_status()
        payload = r.json()
        engines = payload.get("engines")
        if not isinstance(engines, list):
            return [], False, "Backend engine response did not contain an engine list."
        return engines, bool(payload.get("gemini_second_opinion_configured")), None
    except (requests.RequestException, ValueError) as exc:
        return [], False, str(exc)


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.markdown(
    """
    <section class="rx-hero">
      <div class="rx-eyebrow">Prescription review workspace · Phase 6</div>
      <h1>Clearer prescription review,<br><span>with people in control.</span></h1>
      <p>
        Read prescription images, inspect unverified medicine candidates, and
        review each step before terminology or interaction evidence is considered.
        Images are processed in memory by default.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.error(
    "Research prototype only. Extracted medicine candidates may be wrong. "
    "Review status does not establish clinical correctness. Do not use this tool to make treatment decisions. "
    "Normalization and interaction evidence are not clinical verification.",
    icon="⚠️",
)

backend_status = _check_backend(BACKEND_URL)

if backend_status["ok"]:
    data = backend_status["data"]
    st.success(
        f"✅ Backend Connected (`{BACKEND_URL}`) — Service: `{data.get('service')}` | Version: `{data.get('version')}`"
    )
else:
    st.warning(
        f"⚠️ Backend unreachable at `{BACKEND_URL}`. Start it with:  \n"
        "`python -m uvicorn backend.app.main:app --reload --reload-dir backend --port 8000`"
    )

st.divider()

# ---------------------------------------------------------------------------
# Phase 2: Live OCR Testbed
# ---------------------------------------------------------------------------

st.header("📷 1. Prescription Input & OCR (Phase 2)")

col_left, col_right = st.columns([1, 1], gap="medium")

engines, gemini_configured, engine_status_error = (
    _fetch_ocr_engines(BACKEND_URL)
    if backend_status["ok"]
    else ([], False, backend_status.get("error", "Backend unavailable"))
)
if engine_status_error:
    st.warning(f"OCR engine readiness could not be retrieved: {engine_status_error}")

with col_left:
    st.subheader("Prescription Input")
    input_mode = st.radio(
        "Choose input type",
        options=["Upload image", "Paste image", "Paste text"],
        horizontal=True,
        key="prescription_input_mode",
    )
    image_bytes: bytes | None = None
    image_filename = "pasted-prescription.png"
    image_mime_type = "image/png"

    if input_mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Upload prescription image (PNG, JPG, WEBP — Max 10 MB):",
            type=["png", "jpg", "jpeg", "webp"],
            help="Images are processed in memory and not stored permanently by default.",
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_filename = uploaded_file.name
            image_mime_type = uploaded_file.type or "application/octet-stream"
            st.image(
                image_bytes,
                caption=f"Uploaded: {image_filename}",
                use_container_width=True,
            )
    elif input_mode == "Paste image":
        if st.button("Clear pasted image", key="clear_clipboard_image"):
            previous_component_key = (
                f"clipboard_image_{st.session_state['clipboard_image_reset_id']}"
            )
            if previous_component_key in st.session_state:
                del st.session_state[previous_component_key]
            st.session_state["clipboard_image_reset_id"] += 1
            st.rerun()

        clipboard_payload = _CLIPBOARD_IMAGE_INPUT(
            key=f"clipboard_image_{st.session_state['clipboard_image_reset_id']}",
            default=None,
        )
        if clipboard_payload is None:
            st.info("Click the paste area, then press Ctrl+V or Cmd+V.")
        elif not isinstance(clipboard_payload, dict):
            st.error("Clipboard image input returned an invalid value.")
        else:
            mime_type = clipboard_payload.get("mime_type")
            encoded_image = clipboard_payload.get("base64")
            if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                st.error("Paste a PNG, JPG, or WEBP image.")
            elif not isinstance(encoded_image, str):
                st.error("Clipboard image data is missing.")
            else:
                try:
                    decoded_image = base64.b64decode(encoded_image, validate=True)
                except (binascii.Error, ValueError):
                    st.error("Clipboard image data could not be decoded.")
                else:
                    if not decoded_image:
                        st.error("The pasted image is empty.")
                    elif len(decoded_image) > MAX_PASTED_IMAGE_BYTES:
                        st.error("The pasted image exceeds the 10 MB limit.")
                    else:
                        image_bytes = decoded_image
                        image_mime_type = mime_type
                        image_filename = (
                            "pasted-prescription.jpg"
                            if mime_type == "image/jpeg"
                            else "pasted-prescription.webp"
                            if mime_type == "image/webp"
                            else "pasted-prescription.png"
                        )
                        st.image(
                            image_bytes,
                            caption="Pasted prescription image",
                            use_container_width=True,
                        )
    else:
        pasted_prescription_text = st.text_area(
            "Paste prescription text",
            height=160,
            placeholder="Paste text from a prescription or an existing OCR result...",
            key="phase2_pasted_text",
        )
        if st.button("Use pasted text", key="use_pasted_rx_text"):
            if not pasted_prescription_text.strip():
                st.warning("Paste some prescription text first.")
            else:
                st.session_state["raw_ocr_text"] = pasted_prescription_text.strip()
                st.session_state["input_ocr_text"] = pasted_prescription_text.strip()
                st.success("Text is ready for medicine-candidate extraction below.")
        st.caption("Pasted text skips image OCR.")

    st.subheader("OCR Settings")
    st.caption("Select an OCR engine for the uploaded or pasted image.")

    engine_options: dict[str, str] = {}
    for eng in engines:
        status_icon = "🟢 Ready" if eng["is_available"] else "🔴 Setup Required"
        engine_options[eng["engine_name"]] = f"{eng['engine_name'].upper()} ({status_icon})"

    if not engine_options:
        engine_options = {
            "paddle": "PADDLE (⚪ Status Unavailable)",
        }

    preferred_ready_engine = next(
        (
            engine_name
            for engine_name in ("gemma", "paddle")
            if any(
                engine.get("engine_name") == engine_name and engine.get("is_available")
                for engine in engines
            )
        ),
        None,
    )
    if DEFAULT_OCR_ENGINE in {"paddle", "gemini"} and DEFAULT_OCR_ENGINE in engine_options:
        default_engine = DEFAULT_OCR_ENGINE
    else:
        default_engine = preferred_ready_engine or DEFAULT_OCR_ENGINE
    selected_engine_key = st.selectbox(
        "Choose OCR Engine:",
        options=list(engine_options.keys()),
        format_func=lambda k: engine_options.get(k, k),
        index=(
            list(engine_options.keys()).index(default_engine)
            if default_engine in engine_options
            else 0
        ),
    )

    active_engine_info = next((e for e in engines if e["engine_name"] == selected_engine_key), None)
    if active_engine_info:
        if not active_engine_info["is_available"]:
            st.info(f"ℹ️ **Setup needed**: {active_engine_info.get('instructions_if_unavailable')}")
        else:
            st.caption(f"ℹ️ {active_engine_info.get('description')}")
            if selected_engine_key == "gemini":
                model_name = active_engine_info.get("details", {}).get("model")
                if model_name:
                    st.caption(f"Configured Gemini model: `{model_name}`")
            elif selected_engine_key == "gemma":
                details = active_engine_info.get("details", {})
                provider = details.get("provider")
                model_name = details.get("api_model") or details.get("ollama_model")
                if provider:
                    st.caption(f"Gemma provider: `{provider}`")
                if model_name:
                    st.caption(f"Configured Gemma model: `{model_name}`")
    if selected_engine_key == "gemma":
        st.warning(
            "Gemma may misread handwriting. Verify the transcription, "
            "medicine names, doses, and instructions against the image."
        )
    elif selected_engine_key == "gemini":
        st.warning(
            "Gemini will receive this prescription image to transcribe it. "
            "Its transcription is unverified; review medicine candidates and instructions."
        )

    include_gemini_second_opinion = False
    if selected_engine_key == "paddle":
        include_gemini_second_opinion = st.checkbox(
            "Get a Gemini second opinion",
            value=False,
            disabled=not gemini_configured,
            help=(
                "When enabled, the prescription image and PaddleOCR text are sent to "
                "Google Gemini for comparison. Review both outputs yourself."
            ),
        )
    elif selected_engine_key == "gemini":
        st.caption("Gemini is selected as the OCR engine; a separate second opinion is not needed.")
    elif selected_engine_key == "gemma":
        st.caption(
            "Gemma reads the image and returns transcription plus medicine details. "
            "With no API key, it uses the configured local Ollama model."
        )
    include_medicine_extraction = st.checkbox(
        (
            "Use Gemma medicine findings and create a review session"
            if selected_engine_key == "gemma"
            else "Also find medicine names and create a review session"
        ),
        value=True,
        help=(
            (
                "Use the medicine names Gemma returned from the image. Only names also "
                "present in Gemma's transcription become review candidates."
                if selected_engine_key == "gemma"
                else "After OCR, run candidate extraction automatically and make the "
                "candidates available in the mandatory human-review section below."
            )
        ),
    )
    if gemini_configured:
        st.warning(
            "Gemini is an external service. Enabling this sends the prescription image "
            "and PaddleOCR text to Google; use only with appropriate permission."
        )
    else:
        st.caption("Configure GEMINI_API_KEY in the backend .env to enable the optional comparison.")

with col_right:
    st.subheader("OCR Output")

    if input_mode == "Paste text":
        st.info(
            "Pasted text does not run OCR. Use Extract Candidates in the next section "
            "to identify possible medicine names."
        )
    elif image_bytes is None:
        st.info("Choose Upload image or Paste image above to run OCR.")
    else:
        run_label = (
            "🚀 Run OCR + Find Medicine Names"
            if include_medicine_extraction
            else "🚀 Run OCR Engine"
        )
        if st.button(run_label, type="primary"):
            with st.spinner(f"Running {selected_engine_key.upper()} OCR engine in memory..."):
                try:
                    files = {
                        "file": (
                            image_filename,
                            image_bytes,
                            image_mime_type,
                        )
                    }
                    data = {
                        "engine": selected_engine_key,
                        "include_gemini_second_opinion": str(
                            selected_engine_key == "paddle"
                            and include_gemini_second_opinion
                        ).lower(),
                        "include_medicine_extraction": str(
                            include_medicine_extraction
                        ).lower(),
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/prescriptions/ocr",
                        files=files,
                        data=data,
                        # Allow two 600-second Gemma passes plus a response margin.
                        timeout=(
                            540
                            if include_gemini_second_opinion
                            else 1260
                            if selected_engine_key == "gemma"
                            else 240
                            if selected_engine_key == "gemini"
                            else 300
                        ),
                    )

                    if response.status_code == 200:
                        res_data = response.json()
                        result = res_data.get("result", {})
                        raw_text = result.get("raw_text", "")
                        st.session_state["raw_ocr_text"] = raw_text
                        st.session_state["input_ocr_text"] = raw_text

                        st.success(f"✅ OCR Inference Complete ({result.get('processing_time_ms', 0)} ms)")
                        st.text_area("Extracted Raw Text:", value=raw_text, height=180)
                        medicine_extraction = res_data.get("medicine_extraction")
                        if medicine_extraction:
                            st.session_state["review_id"] = medicine_extraction.get("review_id")
                            st.session_state["normalization_ids"] = {}
                            st.session_state["interaction_report_id"] = None
                            st.success(
                                "Medicine candidates extracted. Continue to the mandatory "
                                "human-review section below."
                            )
                        elif include_medicine_extraction and not raw_text.strip():
                            st.warning(
                                "OCR returned no readable text, so medicine candidates "
                                "could not be extracted."
                            )
                        if selected_engine_key == "gemma":
                            structured_data = result.get("structured_data") or {}
                            st.subheader("Gemma medicine details")
                            st.json(
                                {
                                    "medicines": structured_data.get("medicines", []),
                                    "medical_instructions": structured_data.get(
                                        "medical_instructions", []
                                    ),
                                }
                            )
                            st.warning(
                                "Gemma may misread handwriting. Verify the transcription, "
                                "medicine names, doses, and instructions against the image."
                            )
                        elif selected_engine_key == "gemini":
                            structured_data = result.get("structured_data") or {}
                            medicine_names = structured_data.get(
                                "possible_medicine_names", []
                            )
                            st.subheader("Possible medicine names from Gemini")
                            if medicine_names:
                                for medicine_name in medicine_names:
                                    st.text(f"- {medicine_name}")
                            else:
                                st.info(
                                    "Gemini did not identify a possible medicine name."
                                )
                            st.warning(
                                "These are Gemini suggestions, not verified medicine "
                                "identities. Check every name against the image and "
                                "complete the human-review workflow."
                            )
                            st.info(
                                "The transcription and name suggestions above both come "
                                "from Gemini. You can also select **Extract Candidates** "
                                "to run the app's separate text-based review workflow."
                            )

                        gemini_result = res_data.get("gemini_second_opinion")
                        if gemini_result:
                            st.subheader("Gemini second opinion")
                            if gemini_result.get("status") == "success":
                                st.caption(
                                    f"Model: {gemini_result.get('model')} · "
                                    f"{gemini_result.get('processing_time_ms')} ms"
                                )
                                st.text_area(
                                    "Independent transcription and disagreements:",
                                    value=gemini_result.get("text", ""),
                                    height=220,
                                    disabled=True,
                                )
                                st.warning(
                                    "Gemini may also be wrong. Compare against the image and "
                                    "complete the normal human-review steps."
                                )
                            else:
                                st.error(
                                    "Gemini second opinion failed: "
                                    f"{gemini_result.get('error', 'Unknown Gemini error')}. "
                                    "PaddleOCR output is shown separately."
                                )

                    elif response.status_code == 503:
                        err_payload = response.json().get("detail", {})
                        if isinstance(err_payload, dict):
                            st.error(
                                "🔴 OCR Engine Unavailable (503): "
                                f"{err_payload.get('message', 'Engine not ready')}"
                            )
                            st.markdown(
                                "**Instructions:**\n"
                                f"{err_payload.get('instructions', 'Install missing components.')}"
                            )
                            with st.expander("Diagnostic Details"):
                                st.json(err_payload.get("details", {}))
                        else:
                            st.error(f"Service unavailable (503): {err_payload}")

                    elif response.status_code in (400, 413):
                        st.error(f"⚠️ Validation Error ({response.status_code}): {response.json().get('detail')}")

                    elif response.status_code in (502, 504):
                        detail = response.json().get("detail", {})
                        if isinstance(detail, dict):
                            st.error(
                                f"OCR provider error ({response.status_code}): "
                                f"{detail.get('message', 'The OCR provider did not complete the request.')}"
                            )
                            provider_error = detail.get("error")
                            if isinstance(provider_error, str) and provider_error.strip():
                                st.caption(f"Provider details: {provider_error}")
                        else:
                            st.error(f"OCR provider error ({response.status_code}): {detail}")

                    else:
                        st.error(f"Server Error ({response.status_code}): {response.text}")

                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to communicate with OCR endpoint: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Phase 3: Medicine Candidate Extraction
# ---------------------------------------------------------------------------

st.header("🔬 2. Medicine Candidate Extraction (Phase 3)")

st.markdown(
    """
    Extracts potential medicine names, strengths, dosage forms, and administration instructions
    using prescription layout and generic attribute patterns; medicine names are not limited
    to a built-in drug list.
    """
)

sample_options = {
    "Select a synthetic prescription sample...": "",
    "Sample 1: Antibiotic & Pain Relief": (
        "Rx:\n"
        "1. Amoxicillin 500mg capsules PO TID x 7 days\n"
        "2. Paracetamol 650mg tab PRN for pain\n"
        "Disp: 21 caps\n"
        "Dr. Sample Physician, MD"
    ),
    "Sample 2: Chronic Disease Management": (
        "Clinic Health Center\n"
        "Patient: Test Patient\n"
        "1. Metformin 850 mg tablets twice daily with meals\n"
        "2. Atorvastatin 20mg tab once daily at bedtime\n"
        "3. Amlodipine 5mg tab daily\n"
        "Refills: 3"
    ),
    "Sample 3: Topical & Respiratory": (
        "Rx:\n"
        "- Salbutamol 100mcg inhaler 2 puffs PRN\n"
        "- Hydrocortisone 1% cream apply twice daily\n"
        "- Ciprofloxacin 0.5% eye drops 2 drops TID"
    ),
}

selected_sample_label = st.selectbox(
    "Or load a synthetic test prescription:",
    options=list(sample_options.keys()),
)

if sample_options[selected_sample_label]:
    st.session_state["raw_ocr_text"] = sample_options[selected_sample_label]
    st.session_state["input_ocr_text"] = sample_options[selected_sample_label]

input_ocr_text = st.text_area(
    "Prescription OCR Text (from Step 1 or pasted below):",
    height=160,
    placeholder="Paste raw OCR text here to test candidate extraction...",
    key="input_ocr_text",
)

col_act1, col_act2 = st.columns([1, 4])
with col_act1:
    extract_button = st.button("🔍 Extract Candidates", type="primary", use_container_width=True)

if extract_button:
    if not input_ocr_text.strip():
        st.warning("Please provide or upload prescription text before extracting.")
    else:
        with st.spinner("Analyzing text with transparent rule-based patterns..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/prescriptions/extract-medicines",
                    json={"ocr_text": input_ocr_text},
                    timeout=10,
                )

                if resp.status_code == 200:
                    ext_data = resp.json()
                    candidates = ext_data.get("candidates", [])
                    rx_lines = ext_data.get("prescription_lines", [])
                    unmatched_lines = ext_data.get("unmatched_lines", [])
                    st.session_state["review_id"] = ext_data.get("review_id")
                    st.session_state["normalization_ids"] = {}
                    st.session_state["interaction_report_id"] = None

                    st.success(f"Extraction Complete — Found {len(candidates)} candidate mention(s)")

                    st.warning(
                        "⚠️ **Mandatory Review Notice**: All candidates below are unverified rule-based extractions. "
                        "No interaction evidence source is configured by default. "
                        "Every candidate requires an explicit review action. The prototype does not verify reviewer qualifications."
                    )

                    tab_table, tab_lines, tab_unmatched = st.tabs([
                        "📋 Extracted Mentions Table",
                        "📑 Prescription Line Breakdown",
                        "🔍 Unmatched / Metadata Lines",
                    ])

                    with tab_table:
                        if candidates:
                            table_rows = []
                            for c in candidates:
                                table_rows.append({
                                    "Original Token in OCR": c.text,
                                    "Type": c.candidate_type.replace("_", " ").title(),
                                    "Start - End": f"{c.start_char} - {c.end_char}",
                                    "Line #": c.line_number,
                                    "Method": c.extraction_method.replace("_", " ").title(),
                                    "Review Status": c.review_status.replace("_", " ").title(),
                                    "Heuristic Score (uncalibrated)": (
                                        f"{c.heuristic_score:.2f}" if c.heuristic_score is not None else "Unavailable"
                                    ),
                                })
                            st.dataframe(table_rows, use_container_width=True)
                        else:
                            st.info("No medicine candidate mentions detected on the provided text.")

                    with tab_lines:
                        for rx in rx_lines:
                            if rx.get("is_likely_prescription_line"):
                                med = rx.get("medicine_name_candidate")
                                strength = rx.get("strength_candidate")
                                form = rx.get("dosage_form_candidate")
                                freq = rx.get("frequency_candidate")
                                unmatched = rx.get("unmatched_tokens", [])

                                with st.container(border=True):
                                    st.markdown(f"**Line {rx['line_number']}:** `{rx['raw_line']}`")
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("Medicine Candidate", med["text"] if med else "Not detected")
                                    c2.metric("Strength", strength["text"] if strength else "Not detected")
                                    c3.metric("Dosage Form", form["text"] if form else "Not detected")
                                    c4.metric("Frequency / Sig", freq["text"] if freq else "Not detected")
                                    if unmatched:
                                        st.caption(f"Unmatched tokens: {', '.join(unmatched)}")

                    with tab_unmatched:
                        if unmatched_lines:
                            st.markdown("Lines identified as non-prescription or header metadata:")
                            for line in unmatched_lines:
                                st.code(line, language="text")
                        else:
                            st.info("All lines had at least one candidate match.")

                elif resp.status_code in (400, 422):
                    st.error(f"Validation Error ({resp.status_code}): {resp.json().get('detail')}")
                else:
                    st.error(f"Server Error ({resp.status_code}): {resp.text}")

            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to communicate with extraction endpoint: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Phase 4: Mandatory candidate review
# ---------------------------------------------------------------------------

st.header("🧑‍⚕️ 3. Mandatory Human Review (Phase 4)")

review_id = st.session_state.get("review_id")
if not review_id:
    st.info("Extract candidates to create an in-memory review session.")
else:
    try:
        review_response = requests.get(
            f"{BACKEND_URL}/prescriptions/review/{review_id}",
            timeout=10,
        )
        if review_response.status_code != 200:
            st.error(
                f"Could not load review session ({review_response.status_code}): "
                f"{review_response.text}"
            )
        else:
            review_session = review_response.json()
            st.caption(
                f"Review session `{review_id}` — "
                f"{review_session['pending_count']} pending, "
                f"{review_session['confirmed_count'] + review_session['corrected_count']} confirmed/corrected, "
                f"{review_session['rejected_count']} rejected, "
                f"{review_session['uncertain_count']} uncertain."
            )

            pending_candidates = [
                candidate for candidate in review_session["candidates"].values()
                if candidate["review_status"] == "pending_review"
            ]
            completed_candidates = [
                candidate for candidate in review_session["candidates"].values()
                if candidate["review_status"] != "pending_review"
            ]

            st.subheader("Pending review")
            if not pending_candidates:
                st.success("There are no pending candidates in this review session.")

            for candidate in pending_candidates:
                candidate_id = candidate["candidate_id"]
                with st.container(border=True):
                    st.markdown(f"**Original candidate:** `{candidate['original_candidate_text']}`")
                    st.markdown(f"**Full evidence context:** `{candidate['evidence_context']}`")
                    st.write(
                        f"Type: `{candidate['candidate_type']}` · "
                        f"Method: `{candidate['extraction_method']}` · "
                        f"Line: `{candidate['line_number']}` · "
                        f"Offsets: `{candidate['start_char']}–{candidate['end_char']}`"
                    )
                    score = candidate.get("heuristic_score")
                    st.caption(
                        "Heuristic score: "
                        + (f"{score:.2f}" if score is not None else "unavailable")
                        + " — an uncalibrated rule weight, not a probability; it has not been validated "
                        "on a labeled dataset."
                    )
                    ocr_confidence = candidate.get("ocr_engine_confidence")
                    if candidate.get("ocr_confidence_available") and ocr_confidence is not None:
                        st.caption(f"OCR engine confidence: {ocr_confidence}")
                    else:
                        st.caption("OCR engine confidence: unavailable (not supplied to candidate extraction)")
                    st.caption(
                        "Evidence flags — prescription layout pattern: "
                        f"{candidate['has_contextual_pattern']}; possible ambiguity/spillover: "
                        f"{candidate['has_spillover_or_ambiguity']}. Current state: "
                        f"`{candidate['review_status']}`."
                    )

                    correction = st.text_input(
                        "Correction (only used when choosing Correct)",
                        key=f"review-correction-{candidate_id}",
                        max_chars=500,
                    )
                    action_columns = st.columns(4)
                    review_actions = [
                        ("Confirm as written", "confirm"),
                        ("Save correction", "correct"),
                        ("Reject", "reject"),
                        ("Mark uncertain", "mark_uncertain"),
                    ]
                    for column, (label, action) in zip(action_columns, review_actions):
                        if column.button(label, key=f"review-{action}-{candidate_id}"):
                            payload = {"action": action}
                            if action == "correct":
                                if not correction.strip():
                                    st.error("Enter a non-empty correction before saving it.")
                                    continue
                                payload["corrected_text"] = correction
                            try:
                                action_response = requests.post(
                                    f"{BACKEND_URL}/prescriptions/review/{review_id}/candidates/{candidate_id}",
                                    json=payload,
                                    timeout=10,
                                )
                                if action_response.status_code == 200:
                                    st.rerun()
                                else:
                                    st.error(
                                        f"Review action failed ({action_response.status_code}): "
                                        f"{action_response.text}"
                                    )
                            except requests.RequestException as exc:
                                st.error(f"Could not submit review action: {exc}")

            st.subheader("Completed review actions")
            if not completed_candidates:
                st.info("No candidates have been reviewed yet.")
            for candidate in completed_candidates:
                with st.container(border=True):
                    score = candidate.get("heuristic_score")
                    st.markdown(
                        f"**State:** `{candidate['review_status']}` · "
                        f"**Action:** `{candidate['review_action']}` · "
                        f"**Reviewed at:** `{candidate['reviewed_at']}`"
                    )
                    st.write(
                        f"Type: `{candidate['candidate_type']}` · "
                        f"Method: `{candidate['extraction_method']}` · "
                        f"Line: `{candidate['line_number']}` · "
                        f"Offsets: `{candidate['start_char']}–{candidate['end_char']}`"
                    )
                    st.caption(
                        "Heuristic score: "
                        + (f"{score:.2f}" if score is not None else "unavailable")
                        + " — an uncalibrated rule weight, not a probability."
                    )
                    ocr_confidence = candidate.get("ocr_engine_confidence")
                    if candidate.get("ocr_confidence_available") and ocr_confidence is not None:
                        st.caption(f"OCR engine confidence: {ocr_confidence}")
                    else:
                        st.caption("OCR engine confidence: unavailable")
                    st.caption(
                        "Evidence flags — prescription layout pattern: "
                        f"{candidate['has_contextual_pattern']}; possible ambiguity/spillover: "
                        f"{candidate['has_spillover_or_ambiguity']}."
                    )
                    if candidate["review_status"] == "corrected":
                        original, corrected = st.columns(2)
                        original.markdown(
                            f"**Original OCR candidate**  \n`{candidate['original_candidate_text']}`"
                        )
                        corrected.markdown(
                            f"**Reviewer-entered correction**  \n`{candidate['reviewer_corrected_text']}`"
                        )
                    else:
                        st.markdown(
                            f"**Original candidate:** `{candidate['original_candidate_text']}`"
                        )
                    st.caption(f"Full evidence context: {candidate['evidence_context']}")

                if (
                    candidate["review_status"] in {"confirmed", "corrected"}
                    and candidate["candidate_type"] == "possible_medicine_name"
                ):
                    normalization_ids = st.session_state["normalization_ids"]
                    normalization_id = normalization_ids.get(candidate_id)
                    if not normalization_id:
                        st.caption(
                            "This explicit lookup sends only the reviewed medicine text to the "
                            "U.S. National Library of Medicine RxNav service."
                        )
                        if st.button(
                            "Request terminology normalization",
                            key=f"normalize-{candidate_id}",
                        ):
                            try:
                                normalization_response = requests.post(
                                    f"{BACKEND_URL}/prescriptions/review/{review_id}/"
                                    f"candidates/{candidate_id}/normalization",
                                    timeout=10,
                                )
                                if normalization_response.status_code == 201:
                                    normalization_ids[candidate_id] = normalization_response.json()[
                                        "normalization_id"
                                    ]
                                    st.rerun()
                                else:
                                    st.error(
                                        f"Normalization request failed "
                                        f"({normalization_response.status_code}): "
                                        f"{normalization_response.text}"
                                    )
                            except requests.RequestException as exc:
                                st.error(f"Could not request normalization: {exc}")
                    else:
                        try:
                            normalization_response = requests.get(
                                f"{BACKEND_URL}/prescriptions/normalization/{normalization_id}",
                                timeout=10,
                            )
                            if normalization_response.status_code != 200:
                                st.error(
                                    f"Could not retrieve normalization result "
                                    f"({normalization_response.status_code}): "
                                    f"{normalization_response.text}"
                                )
                            else:
                                normalization = normalization_response.json()
                                st.markdown("**Separate normalization review**")
                                st.write(
                                    f"Input text: `{normalization['evidence']['normalization_input_text']}` · "
                                    f"Status: `{normalization['status']}`"
                                )
                                st.write(
                                    "Parsed attributes: "
                                    f"strength `{normalization['evidence']['attributes']['strength']}`, "
                                    f"form `{normalization['evidence']['attributes']['dosage_form']}`, "
                                    f"route `{normalization['evidence']['attributes']['route']}`, "
                                    f"frequency `{normalization['evidence']['attributes']['frequency']}`"
                                )
                                st.caption(
                                    f"Terminology source: {normalization['provider_name']} · "
                                    f"configured: {normalization['provider_configured']} · "
                                    f"version: {normalization['provider_version'] or 'unavailable'} · "
                                    f"release: {normalization['provider_release_date'] or 'unavailable'}"
                                )
                                st.info(normalization["resolution_reason"])

                                concepts = normalization["proposed_concepts"]
                                if concepts:
                                    st.markdown("**Proposed source-backed concepts**")
                                    st.caption(
                                        "RxNav ranking scores order these suggestions for this query. "
                                        "They are not calibrated probabilities or proof of medicine identity."
                                    )
                                    for concept in concepts:
                                        with st.container(border=True):
                                            st.markdown(
                                                f"**{concept['name']}** "
                                                f"(`{concept['source_identifier']}`)"
                                            )
                                            st.write(
                                                f"Ingredient: `{concept['ingredient']}` · "
                                                f"Strength: `{concept['strength']}` · "
                                                f"Form: `{concept['dosage_form']}` · "
                                                f"Route: `{concept['route']}`"
                                            )
                                            st.write(
                                                f"Match method: `{concept['match_method']}` · "
                                                f"Attribute comparison: `{concept['attribute_match']}`"
                                            )
                                            if concept.get("match_score") is not None:
                                                st.write(
                                                    f"RxNav native score: `{concept['match_score']}` · "
                                                    f"Rank: `{concept.get('match_rank') or 'unavailable'}`"
                                                )
                                            st.caption(
                                                f"Evidence: {concept['evidence']} · "
                                                f"Source: {concept['source_name']} · "
                                                f"Version: {concept['source_version'] or 'unavailable'} · "
                                                f"Release: {concept['source_release_date'] or 'unavailable'}"
                                            )
                                            for warning in concept["attribute_warnings"]:
                                                st.warning(warning)

                                if normalization["accepted_concept"]:
                                    st.success(
                                        "Accepted concept: "
                                        f"{normalization['accepted_concept']['name']} "
                                        f"({normalization['accepted_concept']['source_identifier']})"
                                    )
                                elif (
                                    normalization["status"] in {"pending_review", "ambiguous"}
                                    and normalization["decision_action"] is None
                                ):
                                    if concepts:
                                        concept_options = {
                                            concept["source_identifier"]: (
                                                f"{concept['name']} — {concept['source_identifier']} "
                                                f"({concept['match_method']})"
                                            )
                                            for concept in concepts
                                        }
                                        selected_concept = st.selectbox(
                                            "Select a concept to consider (no selection is accepted by default)",
                                            options=[""] + list(concept_options),
                                            format_func=lambda value: (
                                                "Choose a concept..." if not value
                                                else concept_options[value]
                                            ),
                                            key=f"normalization-selection-{normalization_id}",
                                        )
                                        selected = next(
                                            (
                                                concept
                                                for concept in concepts
                                                if concept["source_identifier"] == selected_concept
                                            ),
                                            None,
                                        )
                                        acknowledge_mismatches = False
                                        if selected and selected["attribute_match"] == "mismatch":
                                            acknowledge_mismatches = st.checkbox(
                                                "I acknowledge the displayed strength/form mismatch and still want to accept this proposal.",
                                                key=f"normalization-mismatch-{normalization_id}",
                                            )
                                        decision_columns = st.columns(3)
                                        decisions = [
                                            (decision_columns[0], "Accept selected concept", "accept"),
                                            (decision_columns[1], "Reject proposals", "reject"),
                                            (decision_columns[2], "Leave unmatched", "mark_unmatched"),
                                        ]
                                        if len(concepts) > 1:
                                            decisions.append(
                                                (
                                                    decision_columns[2],
                                                    "Keep ambiguous",
                                                    "mark_ambiguous",
                                                )
                                            )
                                        for column, label, action in decisions:
                                            if column.button(
                                                label,
                                                key=f"normalization-decision-{action}-{normalization_id}",
                                            ):
                                                if action == "accept" and not selected_concept:
                                                    st.error("Select a proposed concept before accepting.")
                                                    continue
                                                if (
                                                    action == "accept"
                                                    and selected
                                                    and selected["attribute_match"] == "mismatch"
                                                    and not acknowledge_mismatches
                                                ):
                                                    st.error(
                                                        "Acknowledge the displayed strength/form mismatch before accepting."
                                                    )
                                                    continue
                                                payload = {"action": action}
                                                if action == "accept":
                                                    payload["selected_concept_id"] = selected_concept
                                                    payload["acknowledge_attribute_mismatches"] = (
                                                        acknowledge_mismatches
                                                    )
                                                try:
                                                    decision_response = requests.post(
                                                        f"{BACKEND_URL}/prescriptions/normalization/"
                                                        f"{normalization_id}/decision",
                                                        json=payload,
                                                        timeout=10,
                                                    )
                                                    if decision_response.status_code == 200:
                                                        st.rerun()
                                                    else:
                                                        st.error(
                                                            f"Normalization decision failed "
                                                            f"({decision_response.status_code}): "
                                                            f"{decision_response.text}"
                                                        )
                                                except requests.RequestException as exc:
                                                    st.error(
                                                        f"Could not submit normalization decision: {exc}"
                                                    )
                                elif normalization["decision_action"]:
                                    st.caption(
                                        f"Normalization action recorded: "
                                        f"{normalization['decision_action']} at "
                                        f"{normalization['decision_at']}."
                                    )
                        except requests.RequestException as exc:
                            st.error(f"Could not retrieve normalization result: {exc}")

            st.info(
                "Only individually confirmed candidates (using original text) and corrected candidates "
                "(using the separately stored correction) pass the reusable workflow eligibility gate. "
                "Passing that gate is not proof of identity or clinical correctness."
            )
    except requests.RequestException as exc:
        st.error(f"Could not connect to the review API: {exc}")

# ---------------------------------------------------------------------------
# Phase 6: Source-backed interaction evidence report
# ---------------------------------------------------------------------------

st.divider()
st.header("⚠️ 4. Interaction Evidence Report (Phase 6)")
st.warning(
    "Only explicitly accepted Phase 5 concepts may be submitted. "
    "No interaction evidence source is configured in this prototype by default. "
    "A missing record does not establish that a medicine combination is safe."
)

normalization_ids = st.session_state.get("normalization_ids", {})
accepted_normalizations: dict[str, dict[str, Any]] = {}
for normalization_id in normalization_ids.values():
    try:
        normalization_response = requests.get(
            f"{BACKEND_URL}/prescriptions/normalization/{normalization_id}",
            timeout=10,
        )
    except requests.RequestException as exc:
        st.error(f"Could not check normalization record {normalization_id}: {exc}")
        continue

    if normalization_response.status_code != 200:
        st.error(
            f"Could not retrieve normalization record {normalization_id} "
            f"({normalization_response.status_code}): {normalization_response.text}"
        )
        continue

    normalization = normalization_response.json()
    if normalization["status"] == "accepted" and normalization["accepted_concept"]:
        concept = normalization["accepted_concept"]
        accepted_normalizations[normalization_id] = {
            "label": (
                f"{concept['name']} — {concept['source_identifier']} "
                f"(normalization {normalization_id})"
            ),
            "normalization_id": normalization_id,
        }

if not accepted_normalizations:
    st.info("No accepted Phase 5 normalization records are available in this review session.")
else:
    report_options = list(accepted_normalizations)
    selected_normalizations = st.multiselect(
        "Select accepted normalized concepts to include in this evidence lookup",
        options=report_options,
        format_func=lambda value: accepted_normalizations[value]["label"],
        key=f"interaction-normalizations-{st.session_state.get('review_id')}",
    )
    if st.button(
        "Check selected accepted concepts",
        type="primary",
        disabled=len(selected_normalizations) < 2,
    ):
        try:
            response = requests.post(
                f"{BACKEND_URL}/prescriptions/interactions/check",
                json={"normalization_ids": selected_normalizations},
                timeout=30,
            )
            if response.status_code == 201:
                st.session_state["interaction_report_id"] = response.json()["report_id"]
                st.rerun()
            else:
                st.error(
                    f"Interaction check request failed ({response.status_code}): "
                    f"{response.text}"
                )
        except requests.RequestException as exc:
            st.error(f"Could not submit interaction evidence request: {exc}")

report_id = st.session_state.get("interaction_report_id")
if report_id:
    try:
        report_response = requests.get(
            f"{BACKEND_URL}/prescriptions/interactions/reports/{report_id}",
            timeout=10,
        )
    except requests.RequestException as exc:
        st.error(f"Could not retrieve interaction report: {exc}")
    else:
        if report_response.status_code != 200:
            st.error(
                f"Could not retrieve interaction report ({report_response.status_code}): "
                f"{report_response.text}"
            )
        else:
            report = report_response.json()
            st.subheader(f"Report status: `{report['overall_status']}`")
            st.write(report["message"])
            st.caption(
                f"Report ID: `{report['report_id']}` · "
                f"Provider: `{report['provider_name']}` · "
                f"Configured: `{report['provider_configured']}` · "
                f"Version: `{report['provider_version'] or 'unavailable'}` · "
                f"Release: `{report['provider_release_date'] or 'unavailable'}`"
            )
            st.write(
                f"Pair coverage: {report['checked_pair_count']} lookup(s) attempted "
                f"of {report['expected_pair_count']} required pair(s). "
                f"Complete: {report['check_complete']}."
            )
            st.error(report["safety_notice"])

            st.markdown("**Accepted concepts included**")
            for concept in report["checked_concepts"]:
                st.write(
                    f"- `{concept['source_identifier']}` — {concept['source_name']} "
                    f"(terminology source: {concept['terminology_source']}; "
                    f"Phase 4: {concept['phase4_review_action']} at "
                    f"{concept['phase4_reviewed_at']}; Phase 5: "
                    f"{concept['phase5_review_action']} at {concept['phase5_decision_at']})"
                )

            if report["blocked_inputs"]:
                st.error("One or more requested records failed the Phase 5 acceptance gate.")
                st.dataframe(report["blocked_inputs"], use_container_width=True)

            st.markdown("**Pair-by-pair results**")
            if not report["pair_results"]:
                st.info("No medicine pairs were checked.")
            for pair in report["pair_results"]:
                concept_a = pair["concept_a"]
                concept_b = pair["concept_b"]
                lookup_state = (
                    "Lookup attempted"
                    if pair["status"] != "not_configured"
                    else "Not checked — no source configured"
                )
                with st.expander(
                    f"Pair {pair['pair_index']}: {concept_a['source_identifier']} × "
                    f"{concept_b['source_identifier']} — {pair['status']}"
                ):
                    st.write(
                        f"{concept_a['source_name']} (`{concept_a['source_identifier']}`) "
                        f"× {concept_b['source_name']} (`{concept_b['source_identifier']}`)"
                    )
                    st.write(
                        f"Status: `{pair['status']}` · {lookup_state} · "
                        f"Coverage complete: `{pair['coverage_complete']}`"
                    )
                    st.caption(
                        f"Evidence source: {pair['source_name']} · "
                        f"Version: {pair['source_version'] or 'unavailable'} · "
                        f"Release: {pair['release_date'] or 'unavailable'} · "
                        f"Retrieved: {pair['retrieved_at'] or 'unavailable'}"
                    )
                    if pair["message"]:
                        st.info(pair["message"])
                    for limitation in pair["limitations"]:
                        st.caption(f"Source limitation: {limitation}")
                    if pair["evidence"]:
                        for evidence in pair["evidence"]:
                            st.markdown(f"**Source evidence:** {evidence['description'] or 'unavailable'}")
                            st.write(
                                f"Source-provided severity/classification: "
                                f"{evidence['source_severity'] or 'unavailable'}"
                            )
                            st.caption(
                                f"Record: {evidence['source_record_identifier'] or 'unavailable'} · "
                                f"URL: {evidence['source_record_url'] or 'unavailable'} · "
                                f"Dataset version: {evidence['dataset_version'] or 'unavailable'} · "
                                f"Release: {evidence['release_date'] or 'unavailable'} · "
                                f"Retrieved: {evidence['retrieved_at'] or 'unavailable'}"
                            )
                            if evidence["relevant_context"]:
                                st.write(f"Source context: {evidence['relevant_context']}")
                            for limitation in evidence["limitations"]:
                                st.caption(f"Evidence limitation: {limitation}")
                    elif pair["status"] == "no_matching_record_found":
                        st.info(
                            "The configured source returned no matching record for this pair. "
                            "This is not evidence that the combination is safe."
                        )

            for limitation in report["limitations"]:
                st.caption(f"Report limitation: {limitation}")

# ---------------------------------------------------------------------------
# Privacy notice
# ---------------------------------------------------------------------------

st.info(
    "🔒 **Privacy** — Images are processed in memory and not stored permanently by default. "
    "PaddleOCR and local Ollama process data locally. If you select Gemini or configure "
    "cloud Gemma, the image or relevant chat data is sent to Google for that request. "
    "Persistent local storage is opt-in only (`PERSIST_IMAGES=true`)."
)
