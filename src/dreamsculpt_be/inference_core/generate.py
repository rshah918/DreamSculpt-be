from PIL import Image
from typing import List
from time import sleep, time
from dreamsculpt_be.config import model_prompt
import torch
from diffusers import FluxKontextPipeline


def mock_generate(batch: List[Image.Image], input_height_width=200, output_height_width=512) -> List[Image.Image]:
    sleep(2)
    return batch


def generate(
    pipeline: FluxKontextPipeline, batch: List[Image.Image], input_height_width=200, output_height_width=512
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
        prompt=[model_prompt] * len(batch),
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
