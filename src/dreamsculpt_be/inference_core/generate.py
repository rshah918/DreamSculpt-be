from PIL import Image
from io import BytesIO
from typing import List
from time import sleep, time
import torch
from diffusers import FluxKontextPipeline
from fastapi.exceptions import HTTPException
from google.genai import Client
from google.genai.types import GenerateContentConfig, ImageConfig, GenerateContentResponse
from dreamsculpt_be.config import ASPECT_RATIO, RESOLUTION
import dreamsculpt_be.inference_core.scheduler as scheduler
from concurrent.futures import ThreadPoolExecutor

def mock_generate(
    text_prompts, batch: List[Image.Image], input_height_width=200, output_height_width=512
) -> List[Image.Image]:
    sleep(2)
    return batch


def generate(
    pipeline: FluxKontextPipeline,
    text_prompts,
    batch: List[Image.Image],
    input_height_width=200,
    output_height_width=512,
) -> List[Image.Image]:
    batch = [
        image.resize(
            (input_height_width, input_height_width),
            Image.LANCZOS,
        )
        for image in batch
    ]

    generation_start = time()
    images = pipeline(
        prompt=text_prompts,
        max_area=output_height_width**2,
        _auto_resize=False,
        image=batch,
        num_inference_steps=15,
        guidance_scale=5,
        height=output_height_width,
        width=output_height_width,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images

    generation_end = time()
    print(f"Image generated in {generation_end - generation_start} seconds")
    return images

def gemini_generate_batch(client: Client, text_prompts: List[str], image_prompts: List[Image.Image]) -> List[Image.Image]:
    if scheduler.remaining_generations < len(image_prompts):
        raise HTTPException(status_code=429, detail="Gemini generation limit reached!")
    
    # Gemini API only allows single image generation. Spawn threads to process batch in parallel
    with ThreadPoolExecutor(max_workers=len(text_prompts)) as executor:
        results = list(
            executor.map(
                lambda args: gemini_generate(client, *args),
                zip(text_prompts, image_prompts)
            )
        )
    scheduler.remaining_generations -= len(image_prompts)
    return results

def gemini_generate(client: Client, text_prompt: str, image_prompt: str) -> Image.Image:
    config: GenerateContentConfig = GenerateContentConfig(response_modalities=['IMAGE'], image_config=ImageConfig(aspect_ratio=ASPECT_RATIO, image_size=RESOLUTION))

    response: GenerateContentResponse = client.models.generate_content(model="gemini-2.5-flash-image", contents=[text_prompt, image_prompt], config=config)
    for part in response.parts:
        if part.inline_data is not None:
            # convert google's image object to PIl image
            image_bytes = part.inline_data.data
            return Image.open(BytesIO(image_bytes)).convert("RGB")

