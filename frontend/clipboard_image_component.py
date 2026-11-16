from pathlib import Path

import streamlit.components.v1 as components

CLIPBOARD_IMAGE_INPUT = components.declare_component(
    "rx_clipboard_image_input",
    path=Path(__file__).resolve().parent / "components" / "clipboard_image",
)
