import base64
from io import BytesIO
from PIL import Image


def base64_encode_image(pil_image: Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
