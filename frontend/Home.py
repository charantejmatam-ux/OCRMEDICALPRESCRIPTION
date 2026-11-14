"""prescription image search and medicine information."""

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

PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_dotenv_config = dotenv_values(PROJECT_ENV_FILE, interpolate=False)
BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or _dotenv_config.get("BACKEND_URL")
    or "http://localhost:8000"
).rstrip("/")
del _dotenv_config

MAX_IMAGE_BYTES = 10 * 1024 * 1024
_CLIPBOARD_IMAGE_INPUT = components.declare_component(
    "gemma_prescription_clipboard_image",
    path=Path(__file__).resolve().parent / "components" / "clipboard_image",
)

_MEDICINE_INFO_QUESTION = (
    "Summarize the labeled uses, age-group directions, warnings, side effects, "
    "and storage information for this medicine. Use only facts supported by "
    "the selected label. Do not give personalized dosing or treatment advice."
)


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

if "clipboard_image_reset_id" not in st.session_state:
    st.session_state["clipboard_image_reset_id"] = 0
if "prescription_search_results" not in st.session_state:
    st.session_state["prescription_search_results"] = []

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
        results: list[dict[str, Any]] = []
        for filename, image_bytes, mime_type in images:
            with st.spinner(
                f"Extracting text and identifying medicines from {filename}..."
            ):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/prescriptions/ocr",
                        files={"file": (filename, image_bytes, mime_type)},
                        data={"engine": "gemma"},
                        # Allow two 600-second Gemma passes plus a response margin.
                        timeout=1260,
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

    raw_text = image_result.get("raw_text")
    with st.expander("Extracted prescription text"):
        if isinstance(raw_text, str) and raw_text.strip():
            st.text(raw_text)
        else:
            st.info("Gemma did not return readable text for this image.")

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
