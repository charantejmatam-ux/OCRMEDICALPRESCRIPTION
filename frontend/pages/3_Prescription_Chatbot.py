"""Chat about medicines detected in the current prescription."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import dotenv_values

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_dotenv_config = dotenv_values(PROJECT_ENV_FILE, interpolate=False)
BACKEND_URL = (
    os.getenv("BACKEND_URL") or _dotenv_config.get("BACKEND_URL") or "http://localhost:8000"
).rstrip("/")
del _dotenv_config

st.set_page_config(page_title="Prescription Chatbot", page_icon="💬", layout="wide")
st.title("Prescription-Aware Chatbot")
st.write(
    "Ask follow-up questions about the medicines detected in your current "
    "prescription. You can ask what each medicine is used for, how it works, "
    "its side effects and precautions, age-group information, label directions "
    "such as how often it is taken, and interactions mentioned in its label."
)
st.info(
    "Medicine names and instructions come from OCR and may be wrong. Check them "
    "against the original prescription. This chatbot cannot determine personal "
    "dosing or replace a doctor or pharmacist. For a possible urgent reaction or "
    "overdose, contact emergency services or poison control."
)
st.caption(
    "DailyMed labels can describe uses, how a medicine works, side effects, "
    "precautions, and interactions. Interaction answers reflect what each "
    "selected label states; no mention of an interaction does not prove a "
    "combination is safe."
)

prescription_results = st.session_state.get("prescription_search_results", [])
medicines: list[dict[str, Any]] = []
for image_index, image_result in enumerate(prescription_results):
    if not isinstance(image_result, dict) or image_result.get("error"):
        continue
    for medicine_index, medicine in enumerate(image_result.get("medicines", [])):
        if not isinstance(medicine, dict):
            continue
        name = medicine.get("medicine_name")
        if not isinstance(name, str) or not name.strip():
            continue
        medicines.append(
            {
                "key": f"{image_index}_{medicine_index}",
                "name": name.strip(),
                "filename": image_result.get("filename", "prescription"),
                "dosage": medicine.get("dosage"),
                "frequency": medicine.get("frequency"),
                "duration": medicine.get("duration"),
            }
        )

if not medicines:
    st.warning(
        "No detected medicines are available in this session. Return to the "
        "prescription page, upload or paste an image, and select **Search medicines** first."
    )
    st.stop()

prescription_fingerprint = tuple(
    (medicine["filename"], medicine["name"].casefold()) for medicine in medicines
)
if st.session_state.get("rx_chat_prescription_fingerprint") != prescription_fingerprint:
    for key in list(st.session_state):
        if key.startswith(("rx_chat_labels_", "rx_chat_label_choice_")):
            del st.session_state[key]
    st.session_state["rx_chat_messages"] = []
    st.session_state["rx_chat_prescription_fingerprint"] = prescription_fingerprint

st.subheader("Medicines detected")
for medicine in medicines:
    details = [medicine["name"]]
    if medicine.get("dosage"):
        details.append(str(medicine["dosage"]))
    if medicine.get("frequency"):
        details.append(str(medicine["frequency"]))
    if medicine.get("duration"):
        details.append(str(medicine["duration"]))
    st.markdown(f"- {' · '.join(details)} — *{medicine['filename']}*")

if st.button("Find official DailyMed labels", type="primary"):
    with st.spinner("Searching DailyMed for each detected medicine..."):
        for medicine in medicines:
            cache_key = f"rx_chat_labels_{medicine['key']}"
            try:
                response = requests.get(
                    f"{BACKEND_URL}/medicine-chat/labels",
                    params={"drug_name": medicine["name"]},
                    timeout=15,
                )
                if response.status_code == 200:
                    st.session_state[cache_key] = response.json().get("labels", [])
                else:
                    try:
                        detail = response.json().get("detail")
                    except ValueError:
                        detail = response.text
                    st.session_state[cache_key] = {"error": str(detail)}
            except requests.RequestException as exc:
                st.session_state[cache_key] = {"error": str(exc)}

labels_by_medicine: list[tuple[dict[str, Any], dict[str, Any]]] = []
for medicine in medicines:
    cache_key = f"rx_chat_labels_{medicine['key']}"
    cached = st.session_state.get(cache_key)
    if cached is None:
        continue
    if isinstance(cached, dict) and cached.get("error"):
        st.error(f"{medicine['name']}: {cached['error']}")
        continue
    if not cached:
        st.warning(
            f"No DailyMed label found for **{medicine['name']}**. Check the "
            "detected name or confirm it with a pharmacist."
        )
        continue
    selected_index = st.selectbox(
        f"Product label for {medicine['name']} (check strength and formulation)",
        options=list(range(len(cached))),
        format_func=lambda index, choices=cached: (
            choices[index].get("title", "Untitled label")
            + (
                f" — published {choices[index]['published_date']}"
                if choices[index].get("published_date")
                else ""
            )
        ),
        key=f"rx_chat_label_choice_{medicine['key']}",
    )
    selected_label = cached[selected_index]
    st.link_button(
        f"Open {medicine['name']} official label",
        f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={selected_label['setid']}",
    )
    labels_by_medicine.append((medicine, selected_label))

if labels_by_medicine:
    names = ", ".join(medicine["name"] for medicine, _ in labels_by_medicine)[:80]
    with st.form("prescription_chat_question"):
        question = st.text_area(
            "Ask about this prescription",
            max_chars=200,
            placeholder=(
                "What are these medicines used for? How do they work? What side "
                "effects or precautions are listed? Do their labels discuss interactions?"
            ),
        )
        submitted = st.form_submit_button("Ask about these medicines")

    if submitted:
        if not question.strip():
            st.warning("Enter a medicine question.")
        else:
            items: list[dict[str, Any]] = []
            previous_messages = st.session_state.get("rx_chat_messages", [])
            with st.spinner("Checking the selected official medicine labels..."):
                for medicine, label in labels_by_medicine:
                    previous_exchange = None
                    for previous_message in reversed(previous_messages):
                        for previous_item in previous_message.get("items", []):
                            if (
                                previous_item.get("medicine") == medicine["name"]
                                and isinstance(previous_item.get("answer"), dict)
                            ):
                                previous_exchange = (
                                    str(previous_message.get("question", ""))[:50],
                                    str(previous_item["answer"].get("answer", ""))[:100],
                                )
                                break
                        if previous_exchange:
                            break

                    recent_context = ""
                    if previous_exchange:
                        recent_context = (
                            f"\nRecent Q: {previous_exchange[0]}; label answer: "
                            f"{previous_exchange[1]}"
                        )
                    contextual_question = (
                        f"Medicines: {names}.{recent_context}\n"
                        f"Question: {question.strip()}\n"
                        "Answer only from this selected label. Explain supported uses, "
                        "mechanism, age directions, frequency, side effects, precautions, "
                        "and interactions. Do not infer personal prescription reasons or "
                        "dosing. For interactions, state only label-supported facts and "
                        "never call an unmentioned combination safe."
                    )
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/medicine-chat/ask",
                            json={"setid": label["setid"], "question": contextual_question},
                            timeout=660,
                        )
                        if response.status_code == 200:
                            items.append(
                                {"medicine": medicine["name"], "answer": response.json()}
                            )
                        else:
                            try:
                                detail = response.json().get("detail")
                            except ValueError:
                                detail = response.text
                            items.append(
                                {"medicine": medicine["name"], "error": str(detail)}
                            )
                    except requests.RequestException as exc:
                        items.append({"medicine": medicine["name"], "error": str(exc)})
            st.session_state.setdefault("rx_chat_messages", []).append(
                {"question": question.strip(), "items": items}
            )

for message in st.session_state.get("rx_chat_messages", []):
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        for item in message["items"]:
            st.markdown(f"#### {item['medicine']}")
            if item.get("error"):
                st.error(item["error"])
                continue
            answer = item["answer"]
            st.markdown(answer.get("answer", "No answer was returned."))
            source_title = answer.get("source_title", "DailyMed product label")
            source_url = answer.get("source_url", "https://dailymed.nlm.nih.gov/")
            st.markdown(f"**Source:** [{source_title}]({source_url})")
            sections = answer.get("sections", [])
            st.caption(f"Label sections: {', '.join(sections) or 'none listed'}")
            if answer.get("disclaimer"):
                st.caption(answer["disclaimer"])

st.caption(
    "Reference: [U.S. National Library of Medicine DailyMed](https://dailymed.nlm.nih.gov/). "
    "The exact product label matters; confirm any treatment decision with a clinician or pharmacist."
)
