
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

MODEL_ID = "timbrooks/instruct-pix2pix"

pipe = None

def load_model():
    global pipe

    if pipe is not None:
        return pipe

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


def edit_image(
    image,
    prompt,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
    steps=20
):
    model = load_model()

    image = image.convert("RGB")

    # Resize image while maintaining aspect ratio
    image.thumbnail((512, 512))

    width = (image.width // 8) * 8
    height = (image.height // 8) * 8

    image = image.resize((width, height))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = torch.Generator(device=device).manual_seed(42)

    with torch.inference_mode():
        result = model(
            prompt=prompt,
            image=image,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator
        )

    return result.images[0]
