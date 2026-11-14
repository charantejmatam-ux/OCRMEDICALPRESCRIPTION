# Isolated TrOCR prescription test

This folder runs the Hugging Face model
[`shubham879/trocr-prescription`](https://huggingface.co/shubham879/trocr-prescription)
directly in its own environment, using the CPU or an available GPU. It does not use or change the project's backend, frontend,
or other OCR engines. The output is the model's raw transcription only.

## Install

From the project root, create a separate environment for this test:

```powershell
python -m venv prescription_ocr_test\.venv
prescription_ocr_test\.venv\Scripts\python.exe -m pip install --upgrade pip
prescription_ocr_test\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
prescription_ocr_test\.venv\Scripts\python.exe -m pip install -r prescription_ocr_test\requirements.txt
```

The CPU PyTorch install and Python dependencies do not download this model's
weights. The runner checks `prescription_ocr_test\model` first and will stop with
instructions if the model is missing. Keeping the model files in this folder
also avoids Hugging Face cache symlink permission errors on Windows.

## Run using cached files

From the project root, first confirm the isolated environment and dependencies:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe --version
prescription_ocr_test\.venv\Scripts\python.exe -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
prescription_ocr_test\.venv\Scripts\python.exe -m pip check
```

Then run the model on the included line image:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe prescription_ocr_test\run_ocr.py .\images\sample_line.jpeg
```

To try a different image, replace the image path. Quote paths that contain spaces:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe prescription_ocr_test\run_ocr.py ".\images\handwritten-prescription-1568x1254.jpg"
```

## Try line segmentation on a full prescription

The separate `prescription_ocr.py` script detects horizontal text regions and
transcribes each one. It defaults to the included prescription image:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe -m pip install -r prescription_ocr_test\requirements.txt
prescription_ocr_test\.venv\Scripts\python.exe prescription_ocr_test\prescription_ocr.py
```

Pass another image path as the first argument to test a different file:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe prescription_ocr_test\prescription_ocr.py ".\images\sample_line.jpeg"
```

To inspect segmentation before running OCR, save an annotated copy with green
boxes and red region numbers. This does not load the model:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe prescription_ocr_test\prescription_ocr.py ".\images\sample_line.jpeg" --debug-boxes
```

The annotated image is saved beside the input as `sample_line_debug_boxes.jpg`.

This is experimental line segmentation; neighboring lines may be combined or
missed. Check every transcription against the image, especially medicine names
and instructions.

## Explicitly download model files

Model downloading is disabled by default. The model files are already stored in
`prescription_ocr_test\model`; only run the following if you intentionally want
to fetch any missing files from Hugging Face:

```powershell
prescription_ocr_test\.venv\Scripts\python.exe prescription_ocr_test\run_ocr.py .\images\sample_line.jpeg --download
```

The model files are stored in `prescription_ocr_test\model`. Subsequent runs use
that local copy without downloading. Use a cropped, line-level image for best
results; handwriting may be transcribed incorrectly and must be checked
against the image.
