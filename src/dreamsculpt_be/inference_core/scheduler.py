import multiprocessing
from typing import List, Tuple
from queue import Queue
import torch
from time import time
from dreamsculpt_be.inference_core.generate import mock_generate, gemini_generate_batch
from dreamsculpt_be.config import MAX_BATCH_SIZE, MODEL_PATH, USE_EXTERNAL_MODEL
from dreamsculpt_be.utils.utils import base64_encode_image, base64_decode_image
from PIL import Image
from diffusers import (
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    FluxKontextPipeline,
    AutoencoderKL,
)
from transformers import BitsAndBytesConfig, T5EncoderModel
from threading import Thread
from google.genai import Client


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
        MODEL_PATH,
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


def ipc_receiver(ipc_request_queue: multiprocessing.Queue, session_queue: Queue, request_map: dict[str: Tuple[str, str, str]]):
    # Listen to the IPC queue and queue up incoming requests. This is done in a seperate thread to avoid blocking dispatch
    while True:
        request: Tuple[str, str, str, str] = ipc_request_queue.get()
        session_id: str = str(request[0])
        if session_id not in request_map.keys():
            request_map[session_id] = request[1:]
            session_queue.put(session_id)
        else:
            request_map[session_id] = request[1:]


def scheduler_loop(ipc_request_queue: multiprocessing.Queue, ipc_result_queue: multiprocessing.Queue) -> str:
    # load model / initialize gemini client
    if USE_EXTERNAL_MODEL:
        client = Client()
    else:
        pipeline = load_model()
    request_map: dict[str: Tuple[str, str, str]] = {} # session_id: [request_id, image_prompt, text_prompt]
    session_queue = Queue()
    Thread(target=ipc_receiver, args=[ipc_request_queue, session_queue, request_map], daemon=True).start()
    while True:
        if not session_queue.empty():
            # Calculate batch size
            batch_size: int = 1
            if session_queue.qsize() > 1:
                batch_size = min(session_queue.qsize(), MAX_BATCH_SIZE)
            # Parse request batch
            session_batch: List[str] = [str(session_queue.get()) for _ in range(batch_size)] # 
            request_batch = [request_map[session_id] for session_id in session_batch]
            # Clean up request tracker
            for session_id, request_id in zip(session_batch, request_batch):
                if request_map[session_id] == request_id:
                    del request_map[session_id]

            request_ids: List[str] = [request[0] for request in request_batch]
            image_prompts: List[Image.Image] = [base64_decode_image(request[1]) for request in request_batch]
            text_prompts: List[str] = [request[2] for request in request_batch]
            # Generate
            generated_pil_images: List[Image.Image] = gemini_generate_batch(client, text_prompts, image_prompts) if USE_EXTERNAL_MODEL else mock_generate(text_prompts, batch=image_prompts)
            # Send generated images back to parent process
            generated_images: List[str] = [base64_encode_image(image) for image in generated_pil_images]
            for result in zip(request_ids, generated_images):
                ipc_result_queue.put(result)
