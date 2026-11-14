"""Run the standalone Gemma prescription extractor from a terminal."""

from __future__ import annotations

import argparse
import json
import sys

from gemma_integration.prescription import (
    GemmaIntegrationError,
    process_prescription_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract prescription text and medicine details with Gemma."
    )
    parser.add_argument("image", help="Path to a prescription image")
    arguments = parser.parse_args()
    try:
        result = process_prescription_image(arguments.image)
    except (GemmaIntegrationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
