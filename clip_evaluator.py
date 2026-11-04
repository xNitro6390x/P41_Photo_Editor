
import torch
from transformers import CLIPProcessor, CLIPModel

CLIP_MODEL = "openai/clip-vit-base-patch32"

clip_model = None
clip_processor = None

def load_clip():
    global clip_model, clip_processor

    if clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

        clip_model.to(device)
        clip_model.eval()

    return clip_model, clip_processor


def calculate_clip_score(image, prompt):
    model, processor = load_clip()

    device = next(model.parameters()).device

    inputs = processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        similarity = (image_features @ text_features.T).item()

    score = ((similarity + 1) / 2) * 100

    return round(score, 2)
