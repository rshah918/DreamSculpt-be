from multiprocessing.connection import Connection
from dreamsculpt_be.inference_core.generate import generate
from dreamsculpt_be.config import max_batch_size
from dreamsculpt_be.config import model_path
from dreamsculpt_be.utils.utils import base64_encode_image, base64_decode_image
from PIL.Image import Image
from typing import List, Tuple
from queue import Queue
import torch
from time import time
from diffusers import (
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    FluxKontextPipeline,
    AutoencoderKL,
)
from transformers import BitsAndBytesConfig, T5EncoderModel
from threading import Thread


def load_model() -> FluxKontextPipeline:
    model_load_start = time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="text_encoder_2",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Auto-offload if needed (falls back to CPU for overflow)
    )

    transformer = FluxTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config="black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
    )
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", subfolder="vae", torch_dtype=torch.bfloat16
    )

    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        vae=vae,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipeline.transformer = torch.compile(pipeline.transformer)
    # pipeline.text_encoder_2 = torch.compile(pipeline.text_encoder_2)
    # pipeline.vae.to(memory_format=torch.channels_last)
    # pipeline.vae = torch.compile(pipeline.vae)
    # pipeline.enable_model_cpu_offload(device="cuda")

    model_load_end = time()
    print(f"Model Loaded in {model_load_end - model_load_start} seconds")
    return pipeline


def ipc_receiver(child_conn: Connection, request_queue: Queue):
    # Listen to the IPC pipe and queue up incoming requests. This is done in a seperate thread to avoid blocking dispatch
    while True:
        while child_conn.poll():
            request = child_conn.recv()
            request_queue.put(request)


def scheduler_loop(child_conn: Connection) -> str:
    # load model
    pipeline = load_model()
    request_queue = Queue()
    Thread(target=ipc_receiver, args=[child_conn, request_queue], daemon=True).start()
    while True:
        if not request_queue.empty():
            # Calculate batch size
            batch_size: int = 1
            if request_queue.qsize() > 1:
                batch_size = min(request_queue.qsize(), max_batch_size)
            # Parse request batch
            request_batch: List[Tuple[str, str]] = [request_queue.get() for i in range(batch_size)]
            request_ids: List[str] = [request[0] for request in request_batch]
            image_prompts: List[Image] = [base64_decode_image(request[1]) for request in request_batch]
            # Generate
            generated_pil_images: List[Image] = generate(pipeline, batch=image_prompts)
            # Send generated images back to parent process
            generated_images: List[str] = [base64_encode_image(image) for image in generated_pil_images]
            for result in zip(request_ids, generated_images):
                child_conn.send(result)
