import base64
from io import BytesIO
from PIL import Image
from xai_sdk import Client as GrokClient
from google.genai import Client as GeminiClient
from dreamsculpt_be.config import GEMINI_API_KEY, GROK_API_KEY


def get_client(external_model: str):
    if external_model == 'GROK':
        return GrokClient(api_key=GROK_API_KEY)
    elif external_model == 'GEMINI':
        return GeminiClient(api_key=GEMINI_API_KEY)
    else:
        raise KeyError(f"Invalid external model provider, received: {external_model}")


def base64_encode_image(pil_image: Image.Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def base64_decode_image(b64_encoded_image: str) -> Image.Image:
    if "," in b64_encoded_image:
        b64_encoded_image = b64_encoded_image.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_encoded_image)
    image = Image.open(BytesIO(image_bytes))
    return image.convert("RGB")
