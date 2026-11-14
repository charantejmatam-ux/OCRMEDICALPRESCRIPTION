from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.medicine_chat import (
    DailyMedLabel,
    LabelSection,
    answer_medicine_question,
    extract_relevant_label_sections,
    search_dailymed_labels,
)
from gemma_integration import GemmaSettings

_SETID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_daily_med_search_returns_normalized_label_records(monkeypatch) -> None:
    response = Mock()
    response.json.return_value = {
        "data": [
            {
                "setid": _SETID.upper(),
                "title": "EXAMPLE MEDICINE TABLET",
                "published_date": "Jan 01, 2026",
            },
            {"setid": "not-a-uuid", "title": "Invalid label"},
        ]
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.get", Mock(return_value=response))

    labels = search_dailymed_labels("example")

    assert labels == [
        DailyMedLabel(
            setid=_SETID,
            title="EXAMPLE MEDICINE TABLET",
            published_date="Jan 01, 2026",
        )
    ]


def test_extract_relevant_label_sections_ignores_other_sections() -> None:
    xml = f"""<document xmlns="urn:test">
      <section><title>1 INDICATIONS AND USAGE</title>
        <text><paragraph>Used for a test condition.</paragraph></text>
      </section>
      <section><title>2 DOSAGE AND ADMINISTRATION</title>
        <text><paragraph>Adults take the labeled amount.</paragraph></text>
      </section>
      <section><title>MARKETING INFORMATION</title>
        <text><paragraph>Manufacturer details.</paragraph></text>
      </section>
    </document>"""

    sections = extract_relevant_label_sections(xml)

    assert [section.title for section in sections] == [
        "1 INDICATIONS AND USAGE",
        "2 DOSAGE AND ADMINISTRATION",
    ]
    assert "Adults take the labeled amount." in sections[1].text


def test_extract_relevant_sections_includes_otc_uses_and_directions() -> None:
    xml = """<document>
      <section><title>Uses</title><text>For a labeled use.</text></section>
      <section><title>Directions</title><text>Follow the age-group directions.</text></section>
      <section><title>Ask a doctor before use if</title>
        <text>Medical conditions may matter.</text></section>
    </document>"""

    sections = extract_relevant_label_sections(xml)

    assert [section.title for section in sections] == [
        "Uses",
        "Directions",
        "Ask a doctor before use if",
    ]


def test_chat_rejects_non_medicine_question_before_network_calls(monkeypatch) -> None:
    post = Mock()
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.post", post)

    with pytest.raises(ValueError, match="only answers questions about medicine"):
        answer_medicine_question(
            setid=_SETID,
            question="How do I fix a flat tire?",
        )

    post.assert_not_called()


def test_chat_uses_only_local_gemma_and_selected_label_evidence(monkeypatch) -> None:
    settings = GemmaSettings(
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma3:4b",
    )
    monkeypatch.setattr(
        "backend.app.services.medicine_chat.GemmaSettings.from_env",
        lambda: settings,
    )
    readiness_response = Mock()
    readiness_response.json.return_value = {"models": [{"name": "gemma3:4b"}]}
    readiness_response.raise_for_status.return_value = None
    label_response = Mock()
    label_response.iter_content.return_value = [
        b"""<document><title>Drug Facts</title>
        <component><structuredBody><component><section>
        <subject><manufacturedProduct><manufacturedProduct>
        <name>Example Medicine Tablet</name>
        </manufacturedProduct></manufacturedProduct></subject>
        </section></component></structuredBody></component>
        <section><title>1 INDICATIONS AND USAGE</title>
        <text><paragraph>For the treatment of a test condition.</paragraph></text>
        </section></document>"""
    ]
    label_response.encoding = "utf-8"
    label_response.raise_for_status.return_value = None
    answer_response = Mock()
    answer_response.json.return_value = {
        "message": {"content": "The label says it is used for a test condition."}
    }
    answer_response.raise_for_status.return_value = None
    get = Mock(side_effect=[readiness_response, label_response])
    post = Mock(return_value=answer_response)
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.get", get)
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.post", post)

    result = answer_medicine_question(
        setid=_SETID,
        question="What is this medicine used for?",
    )

    assert result.answer == "The label says it is used for a test condition."
    assert result.model == "gemma3:4b"
    assert result.source_title == "Example Medicine Tablet"
    assert result.sections == ["1 INDICATIONS AND USAGE"]
    messages = post.call_args.kwargs["json"]["messages"]
    assert "medicine-information assistant" in messages[0]["content"]
    assert "For the treatment of a test condition." in messages[1]["content"]
    assert post.call_args.args[0] == "http://localhost:11434/api/chat"


def test_chat_uses_cloud_gemma_with_only_selected_label_evidence(monkeypatch) -> None:
    settings = GemmaSettings(
        api_key="test-key",
        api_model="gemma-api-test",
    )
    monkeypatch.setattr(
        "backend.app.services.medicine_chat.GemmaSettings.from_env",
        lambda: settings,
    )
    label_response = Mock()
    label_response.iter_content.return_value = [
        b"""<document><title>Drug Facts</title>
        <section><title>1 INDICATIONS AND USAGE</title>
        <text><paragraph>For the treatment of a test condition.</paragraph></text>
        </section></document>"""
    ]
    label_response.encoding = "utf-8"
    label_response.raise_for_status.return_value = None
    label_response.close.return_value = None
    cloud_response = Mock()
    cloud_response.status_code = 200
    cloud_response.ok = True
    cloud_response.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "The label says it treats a test condition."}]}}
        ]
    }
    get = Mock(return_value=label_response)
    post = Mock(return_value=cloud_response)
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.get", get)
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.post", post)

    result = answer_medicine_question(
        setid=_SETID,
        question="What is this medicine used for?",
    )

    assert result.answer == "The label says it treats a test condition."
    assert result.model == "gemma-api-test"
    assert post.call_args.args[0].endswith("/models/gemma-api-test:generateContent")
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "test-key"}
    prompt = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
    assert "medicine-information assistant" in prompt
    assert "For the treatment of a test condition." in prompt
    assert "What is this medicine used for?" in prompt
    get.assert_called_once()


def test_chat_falls_back_to_local_gemma_when_cloud_is_rate_limited(monkeypatch) -> None:
    settings = GemmaSettings(
        api_key="test-key",
        api_model="gemma-api-test",
        ollama_model="gemma3:4b",
    )
    monkeypatch.setattr(
        "backend.app.services.medicine_chat.GemmaSettings.from_env",
        lambda: settings,
    )
    label_response = Mock()
    label_response.iter_content.return_value = [
        b"""<document><title>Drug Facts</title>
        <section><title>Uses</title><text>For a labeled use.</text></section>
        </document>"""
    ]
    label_response.encoding = "utf-8"
    label_response.raise_for_status.return_value = None
    label_response.close.return_value = None
    readiness_response = Mock()
    readiness_response.json.return_value = {"models": [{"name": "gemma3:4b"}]}
    readiness_response.raise_for_status.return_value = None
    cloud_response = Mock()
    cloud_response.status_code = 429
    local_response = Mock()
    local_response.json.return_value = {
        "message": {"content": "The label says it is for a labeled use."}
    }
    local_response.raise_for_status.return_value = None
    get = Mock(side_effect=[label_response, readiness_response])
    post = Mock(side_effect=[cloud_response, local_response])
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.get", get)
    monkeypatch.setattr("backend.app.services.medicine_chat.requests.post", post)

    result = answer_medicine_question(
        setid=_SETID,
        question="What is this medicine used for?",
    )

    assert result.answer == "The label says it is for a labeled use."
    assert result.model == "gemma3:4b"
    assert post.call_args_list[0].args[0].endswith(
        "/models/gemma-api-test:generateContent"
    )
    assert post.call_args_list[1].args[0] == "http://localhost:11434/api/chat"


def test_chat_endpoint_rejects_unrelated_questions(client: TestClient) -> None:
    response = client.post(
        "/medicine-chat/ask",
        json={
            "setid": _SETID,
            "question": "Tell me a joke about bicycles.",
        },
    )

    assert response.status_code == 400
    assert "only answers questions about medicine" in response.json()["detail"]


def test_chat_endpoint_search_labels(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr(
        "backend.app.routers.medicine_chat.search_dailymed_labels",
        lambda _: [DailyMedLabel(_SETID, "EXAMPLE MEDICINE", "Jan 01, 2026")],
    )

    response = client.get("/medicine-chat/labels", params={"drug_name": "example"})

    assert response.status_code == 200
    assert response.json()["labels"][0]["setid"] == _SETID
