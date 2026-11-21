"""prescription image search and medicine information."""

from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import dotenv_values

PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_dotenv_config = dotenv_values(PROJECT_ENV_FILE, interpolate=False)
BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or _dotenv_config.get("BACKEND_URL")
    or "http://localhost:8000"
).rstrip("/")
_GEMINI_API_KEY: str | None = (
    os.getenv("GEMINI_API_KEY") or _dotenv_config.get("GEMINI_API_KEY") or None
)
_GEMINI_MODEL: str = (
    os.getenv("GEMINI_MODEL")
    or _dotenv_config.get("GEMINI_MODEL")
    or "gemini-2.0-flash"
)
_OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL")
    or _dotenv_config.get("OLLAMA_BASE_URL")
    or "http://localhost:11434"
).rstrip("/")
_OLLAMA_GEMMA_MODEL = (
    os.getenv("OLLAMA_GEMMA_MODEL")
    or _dotenv_config.get("OLLAMA_GEMMA_MODEL")
    or "gemma3:4b"
)
del _dotenv_config

MAX_IMAGE_BYTES = 10 * 1024 * 1024
GEMINI_CHAT_TIMEOUT_SECONDS = 10
LOCAL_GEMMA_TIMEOUT_SECONDS = 45
_CLIPBOARD_IMAGE_INPUT = components.declare_component(
    "gemma_prescription_clipboard_image",
    path=Path(__file__).resolve().parent / "components" / "clipboard_image",
)

_MEDICINE_INFO_QUESTION = (
    "Summarize the labeled uses, age-group directions, warnings, side effects, "
    "and storage information for this medicine. Use only facts supported by "
    "the selected label. Do not give personalized dosing or treatment advice."
)


# ── Gemini inline chatbot ─────────────────────────────────────────────────────

_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)

_CHATBOT_SYSTEM_PROMPT = """\
You are a medicine-information assistant. The user has just uploaded a \
prescription image. The medicines extracted from that prescription are listed \
below as JSON. Answer ONLY questions that are directly about one or more of \
those medicines. If the question is unrelated to any of these medicines, \
politely decline and remind the user to ask about the listed medicines only.

Do NOT diagnose, recommend starting or stopping a medicine, calculate a \
personal dose, or give any advice specific to an individual patient. \
Direct the user to a doctor or pharmacist for personalised guidance.

When asked about each or all listed medicines, answer for each one in a short \
list, with at most one brief sentence per medicine. Do not just repeat the \
medicine names or ask the user to rephrase an in-scope question. If an extracted \
name is unclear, say it needs verification instead of guessing what medicine it means.

EXTRACTED MEDICINES:
{medicines_json}
"""


_LOCAL_GEMMA_LABEL = "_Gemini is busy; answered by local Gemma._"


def _strip_local_gemma_label(answer: str) -> str:
    """Keep provider labels out of the model's conversation context."""
    return answer.replace(_LOCAL_GEMMA_LABEL, "").strip()


def _ask_gemini(
    medicines: list[dict[str, Any]],
    history: list[dict[str, str]],
    question: str,
) -> str:
    """Call Gemini REST API; return answer text or an error string."""
    if not _GEMINI_API_KEY:
        return (
            "⚠️ GEMINI_API_KEY is not set in your .env file. "
            "Add it to enable the chatbot."
        )

    medicines_json = json.dumps(
        [
            {k: v for k, v in m.items() if k not in {"confidence_status", "confidence_source"}}
            for m in medicines
        ],
        indent=2,
        ensure_ascii=False,
    )
    system_text = _CHATBOT_SYSTEM_PROMPT.format(medicines_json=medicines_json)

    # Build Gemini contents array (multi-turn)
    contents: list[dict[str, Any]] = []
    for turn in history[-3:]:
        contents.append({"role": "user", "parts": [{"text": turn["user"]}]})
        contents.append(
            {
                "role": "model",
                "parts": [{"text": _strip_local_gemma_label(turn["assistant"])}],
            }
        )
    contents.append({"role": "user", "parts": [{"text": question}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }

    url = _GEMINI_REST_URL.format(model=_GEMINI_MODEL, key=_GEMINI_API_KEY)
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=(5, GEMINI_CHAT_TIMEOUT_SECONDS - 5),
        )
    except requests.Timeout:
        return _ask_local_gemma(
            system_text,
            history,
            question,
            reason=f"Gemini timed out after about {GEMINI_CHAT_TIMEOUT_SECONDS} seconds",
        )
    except requests.RequestException as exc:
        return _ask_local_gemma(
            system_text,
            history,
            question,
            reason=f"Gemini request failed: {exc}",
        )

    if not resp.ok:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:300])
        except ValueError:
            detail = resp.text[:300]
        if resp.status_code in {408, 429, 500, 502, 503, 504}:
            return _ask_local_gemma(
                system_text,
                history,
                question,
                reason=f"Gemini returned HTTP {resp.status_code}: {detail}",
            )
        return f"⚠️ Gemini API error (HTTP {resp.status_code}): {detail}"

    try:
        return (
            resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return "⚠️ Gemini returned an unexpected response format."


def _ask_local_gemma(
    system_text: str,
    history: list[dict[str, str]],
    question: str,
    *,
    reason: str,
) -> str:
    """Use the configured local Ollama Gemma model when Gemini is unavailable."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_text}]
    for turn in history[-2:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append(
            {
                "role": "assistant",
                "content": _strip_local_gemma_label(turn["assistant"]),
            }
        )
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            f"{_OLLAMA_BASE_URL}/api/chat",
            json={
                "model": _OLLAMA_GEMMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0.2, "num_predict": 160},
            },
            timeout=LOCAL_GEMMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        answer = (
            payload.get("message", {}).get("content")
            if isinstance(payload, dict)
            else None
        )
    except requests.Timeout:
        return (
            f"⚠️ {reason}. Local Gemma did not finish within "
            f"{LOCAL_GEMMA_TIMEOUT_SECONDS} seconds. Try a shorter question."
        )
    except (requests.RequestException, ValueError) as exc:
        return (
            f"⚠️ {reason}. Local Gemma fallback also failed: {exc}. "
            "Check that Ollama is running and the configured Gemma model is installed."
        )

    if not isinstance(answer, str) or not answer.strip():
        return f"⚠️ {reason}. Local Gemma returned an empty answer."
    return f"{_LOCAL_GEMMA_LABEL}\n\n{answer.strip()}"


def _http_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail: Any = payload.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str) and message.strip():
                return message
        if isinstance(detail, str) and detail.strip():
            return detail
    return response.text.strip() or f"Request failed with HTTP {response.status_code}."


def _decode_clipboard_image(payload: Any) -> tuple[bytes | None, str | None, str | None]:
    if payload is None:
        return None, None, None
    if not isinstance(payload, dict):
        return None, None, "Clipboard image input returned an invalid value."

    mime_type = payload.get("mime_type")
    encoded_image = payload.get("base64")
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        return None, None, "Paste a PNG, JPG, or WEBP image."
    if not isinstance(encoded_image, str):
        return None, None, "Clipboard image data is missing."

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError):
        return None, None, "Clipboard image data could not be decoded."
    if not image_bytes:
        return None, None, "The pasted image is empty."
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None, None, "The pasted image exceeds the 10 MB limit."

    extension = {
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/png": "png",
    }[mime_type]
    return image_bytes, f"pasted-prescription.{extension}", None


st.set_page_config(
    page_title="Medical prescription Ocr",
    page_icon="💊",
    layout="wide",
)

st.title("Medicine Prescription Search")
st.sidebar.page_link(
    "pages/1_Medicine_Chatbot.py",
    label="💬 Medicine-only Chatbot",
)

if "clipboard_image_reset_id" not in st.session_state:
    st.session_state["clipboard_image_reset_id"] = 0
if "prescription_search_results" not in st.session_state:
    st.session_state["prescription_search_results"] = []
if "gemini_chat_history" not in st.session_state:
    st.session_state["gemini_chat_history"] = []

upload_column, paste_column = st.columns(2, gap="large")

with upload_column:
    st.subheader("Upload prescription image")
    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or WEBP image (max 10 MB)",
        type=["png", "jpg", "jpeg", "webp"],
        key="prescription_upload",
    )
    uploaded_image: tuple[str, bytes, str] | None = None
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        if len(uploaded_bytes) > MAX_IMAGE_BYTES:
            st.error("The uploaded image exceeds the 10 MB limit.")
        else:
            uploaded_image = (
                uploaded_file.name,
                uploaded_bytes,
                uploaded_file.type or "application/octet-stream",
            )
            st.image(uploaded_bytes, caption=uploaded_file.name, use_container_width=True)

with paste_column:
    st.subheader("Paste prescription image")
    if st.button("Clear pasted image", key="clear_clipboard_image"):
        previous_component_key = (
            f"clipboard_image_{st.session_state['clipboard_image_reset_id']}"
        )
        if previous_component_key in st.session_state:
            del st.session_state[previous_component_key]
        st.session_state["clipboard_image_reset_id"] += 1
        st.rerun()

    st.caption("Click the paste area, then press Ctrl+V or Cmd+V.")
    clipboard_payload = _CLIPBOARD_IMAGE_INPUT(
        key=f"clipboard_image_{st.session_state['clipboard_image_reset_id']}",
        default=None,
    )
    pasted_bytes, pasted_filename, paste_error = _decode_clipboard_image(clipboard_payload)
    pasted_image: tuple[str, bytes, str] | None = None
    if paste_error:
        st.error(paste_error)
    elif pasted_bytes and pasted_filename:
        mime_type = {
            ".jpg": "image/jpeg",
            ".webp": "image/webp",
            ".png": "image/png",
        }[Path(pasted_filename).suffix]
        pasted_image = (pasted_filename, pasted_bytes, mime_type)
        st.image(pasted_bytes, caption="Pasted prescription image", use_container_width=True)



if st.button("Search medicines", type="primary", use_container_width=True):
    images: list[tuple[str, bytes, str]] = []
    if uploaded_image:
        images.append(uploaded_image)
    if pasted_image:
        images.append(pasted_image)

    if not images:
        st.warning("Upload or paste a prescription image first.")
    else:
        for state_key in list(st.session_state.keys()):
            if state_key.startswith(
                ("medicine_labels_", "medicine_answer_", "medicine_label_choice_")
            ):
                del st.session_state[state_key]
        st.session_state["gemini_chat_history"] = []
        results: list[dict[str, Any]] = []
        for filename, image_bytes, mime_type in images:
            with st.spinner(f"Extracting medicines from {filename}… this may take 1–2 minutes, please wait."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/prescriptions/ocr",
                        files={"file": (filename, image_bytes, mime_type)},
                        data={"engine": "gemma"},
                        timeout=660,
                    )
                except requests.RequestException as exc:
                    results.append(
                        {
                            "filename": filename,
                            "error": f"Could not reach the Gemma OCR backend: {exc}",
                        }
                    )
                    continue

            if response.status_code != 200:
                results.append(
                    {
                        "filename": filename,
                        "error": _http_error_message(response),
                    }
                )
                continue

            try:
                payload = response.json()
            except ValueError:
                results.append(
                    {
                        "filename": filename,
                        "error": "The OCR backend returned an invalid response.",
                    }
                )
                continue

            result = payload.get("result") if isinstance(payload, dict) else None
            if not isinstance(result, dict):
                results.append(
                    {
                        "filename": filename,
                        "error": "The OCR backend response did not include a result.",
                    }
                )
                continue
            structured_data = result.get("structured_data")
            if not isinstance(structured_data, dict):
                results.append(
                    {
                        "filename": filename,
                        "error": "Gemma's OCR response did not include structured medicine data.",
                    }
                )
                continue
            medicines = structured_data.get("medicines")
            if not isinstance(medicines, list):
                results.append(
                    {
                        "filename": filename,
                        "error": "Gemma's OCR response did not include a medicine list.",
                    }
                )
                continue
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            results.append(
                {
                    "filename": filename,
                    "raw_text": result.get("raw_text", ""),
                    "medicines": medicines,
                    "model": metadata.get("model"),
                    "provider": metadata.get("provider"),
                }
            )
        st.session_state["prescription_search_results"] = results

for image_index, image_result in enumerate(
    st.session_state["prescription_search_results"]
):
    st.subheader(f"Results from {image_result['filename']}")
    if image_result.get("error"):
        st.error(image_result["error"])
        continue

    medicines = image_result.get("medicines", [])
    if not medicines:
        st.info("Gemma did not identify a readable medicine name in this image.")
    else:
        for medicine_index, medicine in enumerate(medicines):
            if not isinstance(medicine, dict):
                continue
            name = medicine.get("medicine_name")
            display_name = (
                name.strip()
                if isinstance(name, str) and name.strip()
                else "Name unclear — verify against the image"
            )
            medicine_key = f"{image_index}_{medicine_index}"
            label_results_key = f"medicine_labels_{medicine_key}"
            answer_key = f"medicine_answer_{medicine_key}"
            answer_setid_key = f"{answer_key}_setid"
            label_search_done_key = f"{label_results_key}_searched"
            label_choice_key = f"medicine_label_choice_{medicine_key}"

            with st.container(border=True):
                st.subheader(f"Medicine {medicine_index + 1}")
                st.write(f"Name: {display_name}")
                st.write(f"Dosage: {medicine.get('dosage') or 'Not detected'}")
                st.write(f"Frequency: {medicine.get('frequency') or 'Not detected'}")
                st.write(f"Duration: {medicine.get('duration') or 'Not detected'}")
                st.write(
                    f"Instructions: {medicine.get('instructions') or 'Not detected'}"
                )
                confidence_val = medicine.get("confidence")
                if isinstance(confidence_val, str):
                    try:
                        confidence_val = float(confidence_val.rstrip("%").strip())
                    except ValueError:
                        confidence_val = None
                if isinstance(confidence_val, (int, float)):
                    conf_src = medicine.get("confidence_source") or "Gemma OCR"
                    st.write(f"Confidence: **{confidence_val:.1f}%** ({conf_src})")
                else:
                    st.write("Confidence: Not available")

                search_disabled = not (
                    isinstance(name, str)
                    and 2 <= len(name.strip()) <= 120
                )
                if st.button(
                    "Search medicine information",
                    key=f"search_medicine_{medicine_key}",
                    disabled=search_disabled,
                ):
                    st.session_state.pop(label_results_key, None)
                    st.session_state.pop(label_choice_key, None)
                    st.session_state.pop(answer_key, None)
                    st.session_state.pop(answer_setid_key, None)
                    st.session_state[label_search_done_key] = False
                    try:
                        label_response = requests.get(
                            f"{BACKEND_URL}/medicine-chat/labels",
                            params={"drug_name": name.strip()},
                            timeout=15,
                        )
                    except requests.RequestException as exc:
                        st.error(f"Could not search DailyMed labels: {exc}")
                    else:
                        if label_response.status_code != 200:
                            st.error(_http_error_message(label_response))
                        else:
                            try:
                                label_payload = label_response.json()
                            except ValueError:
                                st.error("DailyMed returned an invalid search response.")
                            else:
                                labels = (
                                    label_payload.get("labels")
                                    if isinstance(label_payload, dict)
                                    else None
                                )
                                if not isinstance(labels, list):
                                    st.error(
                                        "DailyMed returned an invalid label-search response."
                                    )
                                else:
                                    st.session_state[label_results_key] = labels
                                    st.session_state.pop(answer_key, None)
                                    st.session_state[label_search_done_key] = True

                labels = st.session_state.get(label_results_key, [])
                if labels:
                    st.caption(
                        "Select the exact product label; strength and formulation can differ."
                    )
                    selected_label_index = st.selectbox(
                        "DailyMed product label",
                        options=list(range(len(labels))),
                        format_func=lambda index: (
                            labels[index].get("title", "Untitled label")
                            if isinstance(labels[index], dict)
                            else "Invalid label result"
                        ),
                        key=label_choice_key,
                    )
                    selected_label = labels[selected_label_index]
                    if isinstance(selected_label, dict):
                        setid = selected_label.get("setid")
                        if isinstance(setid, str) and st.button(
                            "Get label information with Gemma",
                            key=f"get_medicine_info_{medicine_key}",
                        ):
                            try:
                                answer_response = requests.post(
                                    f"{BACKEND_URL}/medicine-chat/ask",
                                    json={
                                        "setid": setid,
                                        "question": _MEDICINE_INFO_QUESTION,
                                    },
                                    timeout=660,
                                )
                            except requests.RequestException as exc:
                                st.error(f"Could not get label information: {exc}")
                            else:
                                if answer_response.status_code != 200:
                                    st.error(_http_error_message(answer_response))
                                else:
                                    try:
                                        st.session_state[answer_key] = answer_response.json()
                                        st.session_state[answer_setid_key] = setid
                                    except ValueError:
                                        st.error(
                                            "Gemma returned an invalid medicine-information response."
                                        )

                elif st.session_state.get(label_search_done_key):
                    st.info("No matching DailyMed labels were found for this name.")

                selected_answer_setid = (
                    labels[st.session_state[label_choice_key]].get("setid")
                    if labels
                    and isinstance(st.session_state.get(label_choice_key), int)
                    and isinstance(
                        labels[st.session_state[label_choice_key]], dict
                    )
                    else None
                )
                answer = (
                    st.session_state.get(answer_key)
                    if st.session_state.get(answer_setid_key) == selected_answer_setid
                    else None
                )
                if isinstance(answer, dict):
                    answer_text = answer.get("answer")
                    if isinstance(answer_text, str) and answer_text.strip():
                        st.markdown("**Label information**")
                        st.write(answer_text)
                        source_url = answer.get("source_url")
                        source_title = answer.get("source_title")
                        if isinstance(source_url, str) and isinstance(source_title, str):
                            st.markdown(f"Source: [{source_title}]({source_url})")
                        sections = answer.get("sections")
                        if isinstance(sections, list):
                            st.caption(f"Label sections: {', '.join(map(str, sections))}")
                        if isinstance(answer.get("disclaimer"), str):
                            st.caption(answer["disclaimer"])

    raw_text = image_result.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        with st.expander(f"View transcription from {image_result['filename']}"):
            st.text(raw_text)

    st.warning(
        "The given transcription and medicine details are unverified. Confirm every "
        "name and instruction against the original prescription and official label."
    )

# ── Gemini Medicine Chatbot ───────────────────────────────────────────────────
all_results = st.session_state.get("prescription_search_results", [])
all_medicines: list[dict[str, Any]] = []
for res in all_results:
    if isinstance(res.get("medicines"), list):
        all_medicines.extend(res["medicines"])

if all_medicines:
    st.divider()
    st.subheader("💬 Ask Gemini about your prescription medicines")
    st.caption(
        "Gemini will answer questions **only** about the medicines extracted from "
        "this prescription. It cannot diagnose or give personal dosing advice — "
        "always confirm with a doctor or pharmacist."
    )
    if not _GEMINI_API_KEY:
        st.warning(
            "GEMINI_API_KEY is not set in your .env file. "
            "Add it to enable the chatbot."
        )
    else:
        # Show medicine names as quick context pills
        medicine_names = [
            m.get("medicine_name") or "Unclear"
            for m in all_medicines
            if isinstance(m, dict)
        ]
        st.caption(
            "Medicines in context: "
            + " · ".join(f"**{n}**" for n in medicine_names)
        )

        # Suggested questions
        suggested = [
            "What is each medicine used for?",
            "What are the common side effects?",
            "Are there any important warnings?",
            "What is the typical dosage for these medicines?",
            "Can any of these interact with each other?",
        ]
        st.markdown("**Suggested questions:**")
        cols = st.columns(len(suggested))
        for col, suggestion in zip(cols, suggested):
            if col.button(suggestion, key=f"suggest_{suggestion[:20]}"):
                st.session_state["gemini_pending_question"] = suggestion

        # Replay chat history
        for turn in st.session_state["gemini_chat_history"]:
            with st.chat_message("user"):
                st.write(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["assistant"])

        # Handle a suggested question pre-fill OR free-text input
        pending = st.session_state.pop("gemini_pending_question", None)
        user_input = st.chat_input(
            "Ask anything about these medicines…",
            key="gemini_chat_input",
        )
        question_to_ask = pending or user_input

        if question_to_ask:
            with st.chat_message("user"):
                st.write(question_to_ask)
            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking…"):
                    answer = _ask_gemini(
                        all_medicines,
                        st.session_state["gemini_chat_history"],
                        question_to_ask,
                    )
                st.markdown(answer)
            st.session_state["gemini_chat_history"].append(
                {"user": question_to_ask, "assistant": answer}
            )
            st.caption(
                "⚠️ For informational purposes only. Not a diagnosis or personal "
                "treatment recommendation. Confirm with a doctor or pharmacist."
            )

        if st.session_state["gemini_chat_history"]:
            if st.button("🗑️ Clear chat history", key="clear_gemini_chat"):
                st.session_state["gemini_chat_history"] = []
                st.rerun()
