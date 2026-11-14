---
language: en
license: apache-2.0
tags:
- trocr
- handwritten-text-recognition
- medical-ocr
- prescription-ocr
- vision-encoder-decoder
- transformers
pipeline_tag: image-to-text
library_name: transformers
base_model: microsoft/trocr-base-handwritten
---

# Model Card for trocr-prescription

## Model Details

### Model Description

This model is a fine-tuned version of **microsoft/trocr-base-handwritten** for handwritten medical prescription text recognition.

It is designed to extract handwritten prescription text from medical images using transformer-based OCR.

- **Developed by:** shubham879  
- **Model type:** VisionEncoderDecoderModel  
- **Language(s):** English  
- **License:** Apache-2.0  
- **Finetuned from model:** microsoft/trocr-base-handwritten  

### Model Sources

- **Repository:** https://huggingface.co/shubham879/trocr-prescription  
- **Demo:** https://prescription-error-detection.vercel.app/

---

## Uses

### Direct Use

This model is intended for:

- handwritten prescription OCR  
- medical text extraction  
- OCR preprocessing pipelines  

### Downstream Use

Can be integrated into:

- prescription error detection systems  
- medical document digitization  
- healthcare OCR pipelines  

### Out-of-Scope Use

Not recommended for:

- printed document OCR  
- multilingual handwriting  
- severely distorted or noisy scans  

---

## Bias, Risks, and Limitations

- Accuracy decreases on unclear handwriting  
- Sensitive to poor image quality  
- Drug names with unusual handwriting may be misread  

### Recommendations

Use cropped line-level prescription images for best performance.

---

## How to Get Started with the Model

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("shubham879/trocr-prescription")
model = VisionEncoderDecoderModel.from_pretrained("shubham879/trocr-prescription")

image = Image.open("sample.png").convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)

text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(text)
```

## Training Details

### Training Data

The model was fine-tuned using the **Doctor’s Handwritten Prescription BD dataset**, which contains handwritten medical prescription images.

### Training Procedure

#### Preprocessing

- Images resized using TrOCRProcessor  
- RGB conversion applied  
- Tokenization handled by TrOCRProcessor  

#### Training Hyperparameters

- **Training regime:** fp32  

---

## Evaluation

### Metrics

- Character Error Rate (CER)  
- Sequence Accuracy  

### Results

- **CER:** 9%  
- **Accuracy:** 73%  

The model performs well on handwritten prescription text at line level with low character-level error.