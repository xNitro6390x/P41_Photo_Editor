
from PIL import Image, ImageDraw, ImageFont

def add_watermark(image, text="AI EDITED - P41"):
    image = image.convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = image.width - text_width - 20
    y = image.height - text_height - 20

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 190),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180)
    )

    result = Image.alpha_composite(image, overlay)

    return result.convert("RGB")
