# Medicine Information Chatbot (standalone page disabled)

The standalone Streamlit page **Medicine Chatbot** is currently disabled and
excluded from the web app's page navigation. Its previous implementation is
retained, commented out, under `frontend/disabled/Medicine_Chatbot.py`.

The medicine-chat backend remains available to other app features. When enabled,
the chatbot searches official U.S.
DailyMed product labels and answers medicine questions using the selected
label. Search always uses DailyMed; `GEMMA_API_KEY` does not change or improve
the search itself. When the key is configured, Gemma's Google API generates
the answer from the selected label. Without the key, the chatbot uses local
Ollama Gemma. Temporary cloud-service or quota failures also fall back to local
Ollama.

The chatbot covers information stated in the selected label, such as approved
uses, labeled directions for age groups, warnings, and side effects. It does
not calculate a personal dose, decide whether a dose applies to a particular
person, diagnose, or replace advice from a doctor or pharmacist. Questions
outside medicine are rejected, and unsupported questions are directed to the
label or a pharmacist/prescriber.

## Optional cloud setup

1. Create a Google AI Studio API key at <https://aistudio.google.com/apikey>.
2. Put the key in the project-root `.env` file as `GEMMA_API_KEY=your-key`.
   Keep it private; do not commit `.env` or put the key in frontend code.
3. Keep `OCR_ENGINE=gemma` to use cloud Gemma for prescription image extraction.
   The same key enables cloud Gemma answers in the medicine chatbot.
4. Restart the backend after changing `.env`.

When cloud Gemma is used, prescription images go to Google for extraction. For
chat, the question and relevant sections from the selected public DailyMed
label go to Google. DailyMed label search itself remains unchanged. If you do
not want data sent to Google, leave the key empty and run local Ollama Gemma.

## Local setup

Without a key, start Ollama and install the configured model (default:
`gemma3:4b`), then start the FastAPI and Streamlit apps as described in
`README.md`. The chatbot answers from official DailyMed label evidence and
sends the question and label evidence only to the configured local Ollama
service.
