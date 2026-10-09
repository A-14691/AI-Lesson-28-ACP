import io
import os

import requests
from PIL import Image, ImageDraw

from config import HF_API_KEY

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

IMAGE_PATH = "input.jpg"
MASK_PATH = None
MASK_BOX = None
OUTPUT_PATH = "restored_image.png"
PROMPT = (
    "Restore the damaged area of this vintage photograph with realistic inpainting, "
    "preserving the original texture, details, and historical character while seamlessly "
    "repairing cracks, tears, and missing sections."
)


def create_mask_from_box(image_size, box):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    return mask


def prepare_image_and_mask(image_path, mask_path=None, mask_box=None):
    image = Image.open(image_path).convert("RGB")

    if mask_path:
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size)
        return image, mask

    if mask_box is None:
        width, height = image.size
        mask_box = (
            int(width * 0.18),
            int(height * 0.18),
            int(width * 0.82),
            int(height * 0.82),
        )

    mask = create_mask_from_box(image.size, mask_box)
    return image, mask


def run_inpainting(image, mask, prompt):
    image_buffer = io.BytesIO()
    mask_buffer = io.BytesIO()

    image.save(image_buffer, format="PNG")
    mask.save(mask_buffer, format="PNG")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    files = {
        "image": ("input.png", image_buffer.getvalue(), "image/png"),
        "mask_image": ("mask.png", mask_buffer.getvalue(), "image/png"),
    }
    data = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, files=files, data=data, timeout=180)

    if response.status_code != 200:
        try:
            error_details = response.json()
            message = error_details.get("error", error_details)
        except ValueError:
            message = response.text
        raise RuntimeError(f"Inpainting request failed: {message}")

    try:
        result = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unable to read inpainted image response: {exc}") from exc

    return result


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    image, mask = prepare_image_and_mask(IMAGE_PATH, MASK_PATH, MASK_BOX)
    restored = run_inpainting(image, mask, PROMPT)
    restored.save(OUTPUT_PATH)
    print(f"Saved restored image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
