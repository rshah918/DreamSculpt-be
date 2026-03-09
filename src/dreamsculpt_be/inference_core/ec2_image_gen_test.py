from diffusers import (
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    FluxKontextPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image
import torch
from time import time
from transformers import BitsAndBytesConfig, T5EncoderModel
from PIL import Image
from torchinfo import summary


prompt = "Transform this rough sketch into an awe-inspiring, photorealistic image as if it exists in the real world with natural textures, lifelike materials, and fine detail. Add realistic depth, dramatic lighting, and atmospheric effects such as reflections, sky, and shadows. The final result should look like a stunning photograph, true to the layout of the sketch but elevated into a vivid, breathtaking real-world scene"
ckpt_path = "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q5_0.gguf"

model_load_start = time()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

text_encoder_2 = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    subfolder="text_encoder_2",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",  # Auto-offload if needed (falls back to CPU for overflow)
)

transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    config="black-forest-labs/FLUX.1-Kontext-dev",
    subfolder="transformer",
)
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="vae", torch_dtype=torch.float16)

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    transformer=transformer,
    vae=vae,
    text_encoder_2=text_encoder_2,
    torch_dtype=torch.float16,
).to("cuda")


# Resolution (adjust for your needs; must be divisible by 16)
img_height = 512
img_width = 512
pack_h = img_height // 16  # 32
pack_w = img_width // 16  # 32
img_seq_len = pack_h * pack_w  # 1024 image tokens

text_seq_len = 256  # example max text tokens

# Device and dtype matching your GGUF quantized model
device = transformer.device
dtype = torch.float16

dummy_img_ids = torch.zeros(1, img_seq_len, 3, dtype=dtype).to(device)

# txt_ids: (batch, text_seq_len, 3)
dummy_txt_ids = torch.zeros(1, text_seq_len, 3, dtype=dtype).to(device)

pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune")
# pipeline.text_encoder_2 = torch.compile(pipeline.text_encoder_2, mode="max-autotune")
# pipeline.vae.to(memory_format=torch.channels_last)
# pipeline.vae = torch.compile(pipeline.vae, mode="max-autotune")
# pipeline.enable_model_cpu_offload(device="cuda")
# pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
#     pipeline.scheduler.config,
#     timestep_spacing="trailing",
# )

model_load_end = time()
print(f"Model Loaded in {model_load_end - model_load_start} seconds")


def generate_batch(batch_size: int = 1, input_height_width=256, output_height_width=512):
    image = load_image("./test_input.png").resize(
        (input_height_width, input_height_width), 
        Image.LANCZOS,
    )
    image.save("200x200_input.png")

    generation_start = time()
    images = pipeline(
        prompt=[prompt] * batch_size,
        max_area=output_height_width**2,
        _auto_resize=False,
        image=[image] * batch_size,
        num_inference_steps=15,
        guidance_scale=5,
        height=output_height_width,
        width=output_height_width,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    generation_end = time()
    print(f"Image generated in {generation_end - generation_start} seconds")
    images.save("gguf_image.png")


# generate_batch(1)  # dummy generation to compile graph
# generate_batch(1)
# generate_batch(2)  # dummy generation to compile graph
# generate_batch(2)
generate_batch(1, 200)  # dummy generation to compile graph
generate_batch(1, 200)
generate_batch(1, 200)
# generate_batch(2, 200)  # dummy generation to compile graph
# generate_batch(2, 200)


"""
 Benchmark Results (no vae, only text prompt, first run in graph mode):
    | Quant | GPU          | Resolution | Steps | Time (s) | Notes                  | VRAM              |
    | ----- | ------------ | ---------- | ----- | -------- | ---------------------- | ----------------- |
    | Q2_K  | g4dn.2x (T4) | 256×256    | 15    | 87.5     | 13s load, poor quality | 14.1 GB / 15.3 GB |
    | Q2_K  | g5.2x (A10G) | 256×256    | 15    | 11.84    | 7.15s load             | 14.4 GB / 23 GB   |
    | Q2_K  | g5.2x        | 256×256    | 5     | 4.55     | good speed             | 18.7 GB / 23 GB   |
    | Q5_0  | g5.2x        | 256×256    | 15    | 21.79    | mid quality            | 18.7 GB / 23 GB   |
    | Q5_0  | g5.2x        | 256×256    | 30    | 41.45    | slightly less mid      | 18.7 GB / 23 GB   |
    | Q5_0  | g5.2x        | 512×512    | 15    | 25.27    | SWEET SPOT for 512x512 | 19.0 GB / 23 GB   |
    | Q5_0  | g5.2x        | 512×512    | 30    | 49.56    | high quality           | 19.0 GB / 23 GB   |
    | Q5_0  | g5.2x        | 256×256    | 8     | 11.76    | SWEET SPOT for 256x256 | 14.1 GB / 23 GB   |
    | Q8_0  | g5.2x        | —          | —     | OOM      |                        |                   |

Benchmark Results: (5 generations with vae, text + image input, eager mode)
    | Quant | GPU          | Resolution | Steps | Time (s) | Notes                  | VRAM              |
    | ----- | ------------ | ---------- | ----- | -------- | ---------------------- | ----------------- |
    | Q5_0  | g5.2x        | 256×256    | 15    | 12.48    | SWEET SPOT for 256x256 |                   |
    | Q5_0  | g5.2x        | 512×512    | 15    | 15.33    | SWEET SPOT for 512x512 |   20.32 /  24 GB  |
Notes: 
    - Tesla T4 is simply too weak. I could stream the output of each timestep back to the UI to hide the latency, but this just barely supports one image at a time. Tomorrow, try with g5

    - Testing compilation speedup: 8 steps @ 256x256 on g5.2xlarge
        - eager mode: 11.759 seconds
        - with only vae block compiled: 
            - first run: 132.27 seconds
            - second run: 27.18 seconds
            - third run: 18.067 seconds
            - fourth run: 19.052 seconds
            - fifth run: 18.00 seconds
        - with both transformer and vae blocks compiled:
            - first run: 421.07 seconds
            - second run: 88.01 seconds
            - third run: 92.53 seconds
            - fourth run: 87.15 seconds
            - fifth run: 87.08 seconds
        - no compilation, but vae channels_last_format set
            - first run: 11.59 seconds
            - second run: 11.59 seconds
        - I think I may have messed up. The cached compiled tensor graphs might not be persisting between runs. Need to
            do multiple invocations in the same script

    - Verdict: go with g5.2xlarge, 15 steps, 512x512. Stream each generation to feel faster
    Next Step:
        - Hardcode sketch input, update prompt, and validate generation pipeline
        - Update server code with target configs
        - Test deployment

    11/13/2025
        - Validated generation pipeline, and confirmed that previous graph vs eager mode expirement was incorrect. I am seeing a massive speedup with graph mode.
        - Generation quality is trash, need to tweak guidance_scale. Tried 1.0, need to try 7.0 next
        - Look into possibly using Schnell instead of Dev

    11/25/2025
        - Okay lets first try 7.0 guidance scale
            - trying 100.0 guidance scale, 7.0 just returned the same image
            - 4/5 guidance scale:
                - set strength to 0.99, generated a high quality random image
                - strength 0.5, generated the same image as input
                - strength 0.8, generated some garbage
                    - Trying 30 steps: higher quality irrelevant image
        - Alright lets switch to flux kontext. 
            - 0.8 strength 7 guidance scale: garbage
            - 0.99 strength 7 guidance scale: high quality mountain
            - 0.99 strength 20 guidance scale: higher quality mountain
            - 0.99 strength 50 guidance scale: mountain
            - 0.5 strength 4.5 guidance scale: same as input image
                - 30 inference steps: same as input image
            - 0.7 strength 4.5 guidance: garbage
            - 1.0 strength 4.5 guidnace: mountain
            - 0.85 strength, 10 guidance: garbage
                - shorter prompt: garbage
            - 6_K quant
                - 20817MiB 
                - 0.99 strength, 4.5 guidance: high quality knight
                - 0.9 strength, 4.5 guidance: image of computer error screen
                - 0.85 strength, 4.5 guidance: garbage
                - 0.85 strength, 10 guidance: image of computer error screen
                - 0.7 strength, 10 guidance: garbage
                - 0.4 strength, 30 steps, 10 guidace: 
        - I was close to giving up but Draw Things works fine, something is wrong with the inference code. Im going to switch to the FluxKontext 
            pipeline. Also, if Im using a quant theres no way my GPU usage should be 20GB.
            -HOLY SHIT IT WORKED THE IMAGE IS GREAT
        1024x1024 image:
            - 19043MiB /  23028MiB when using 4 bit quantized t5 encoder
            - 12699MiB /  23028MiB when setting model_cpu_offload()

    11/26/2025

FLUX.1-Kontext-dev GGUF – Optimization Benchmarks (11/26/2025)
GPU: AWS g5.2xlarge (A10G 24 GB) | Resolution: 512×512 (unless noted) | Steps: 15 | Img2Img mode

| Quant | Configuration                                                      | VRAM Peak | First Run   | Warm Runs   | Notes                                      |
|-------|--------------------------------------------------------------------|-----------|-------------|-------------|--------------------------------------------|
| Q6_K  | No optimizations                                                   | 20.16 GB  | -           | 67.14 s     | Baseline                                   |
| Q6_K  | enable_model_cpu_offload() only                                    | 12.92 GB  | -           | 76.26 s     | Saves VRAM, slower                         |
| Q6_K  | compile(transformer) only                                          | 20.17 GB  | ~340 s      | 46.3 s      |                                            |
| Q6_K  | compile(transformer + vae + t5)                                    | 19.99 GB  | 166 s       | 46.2 s      |                                            |
| Q6_K  | compile + cpu_offload                                              | 11.80 GB  | 263 s       | 61.9 s      | Too much transfer overhead                 |
|       |                                                                    |           |             |             |                                            |
| Q5_0  | compile(transformer + vae + t5)                                    | 18.24 GB  | 259 s       | 45.7 s      | 1024×1024 input baseline                   |
| Q5_0  | compile + vae.channels_last                                        | 17.70 GB  | 161 s       | 45.9 s      | 0.2 seconds longer w/ channels_last format?|
| Q5_0  | compile + channels_last + max_autotune                             | 19.17 GB  | 169 s       | 45.6 s      | No real gain                               |
| Q5_0  | WINNER: compile + channels_last + 512×512 input                    | 15.36 GB  | ~321 s      | 11.20 s     | BEST SPEED/QUALITY AT 512X512 Input Shape  |
| Q5_0  | compile + channels_last + 512×512 + cpu_offload                    | 9.21 GB   | 133 s       | 19.25 s     | Lowest VRAM, a slower than i would like.   |
| Q5_0  | compile + cpu_offload (no channels_last)                           |  10.01 GB | 393 s       | 51.9 s      | Slower than compile-only                   |
| Q5_0  | compile + cpu_offload (no channels_last) + 256×256                 |  16.17 GB | 358.3 s     | 5.5 s       | BEST CONFIGURATION AT 256X256 Input shape  |
|       |                                                                    |           |             |             |                                            |
| Q4_0  | compile + channels_last                                            | 16.56 GB  | 256 s       | 45.9 s      | POOR QUALITY – artifacts, do not use       |
                
Conclusion:
    - Shrinking input image size makes a huge deal!!!
    - use Q5_0 quant
    - Compile VAE, transfomer, and T5 (max_autotune doesnt help much)
    - Input image size to 512x512, compile model, set vae channels_last format
        - Memory usage:  15711MiB /  23028MiB
        - Single Image generation time: 11.20 seconds (321.42 seconds for first run)
        - with enable_model_cpu_offload() also set:
            - Memory usage: 9423MiB /  23028MiB
            - Single Image generation time: 19.25 seconds (132.79 seconds for first run)
11/28/2025
    - Input image size to 256x256, compile model, set vae channels_last format
        - Memory usage: 15421MiB /  23028MiB
        - Single Image generation time:  5.54 seconds (358.31 seconds for first run)
        - with enable_model_cpu_offload() also set:
            - Memory usage: 9167MiB /  23028MiB
            - Single Image generation time: 12.3 seconds ( 128.7 seconds for first run) 
    - Lets track generation time and memory as a function of batch size
    - without cpu_offload or model compilation:
        Batch=1: 15803MiB /  23028MiB, 23s 
        Batch=2: 15849MiB /  23028MiB, 27s 
        Batch=3: 16041MiB /  23028MiB, 32.1s
        Batch=4: 16555MiB /  23028MiB, 36.8s
        Marginal VRAM increases with batch_size: 46, 192, 514 MiB per additional image
        Max images in flight: ~10
    - Tried to use FlowMatchEulerDiscreteScheduler but the image was a tad grainier than I would like

Learnings:
    - I managed to bring single image generation time down to ~5 seconds!!!! Shrinking the input image halved latency without affecting quality.
    - Batching increases latency to unacceptable levels. I assumed latency would not increase because all images would be processes in parallel
    - Marginal VRAM increases quadratically with each additional image?? 
    - Use multi-processing to fire off each image 1 at a time. 
Next step:
    - Implement multi-processing to stream requests 1 and a time, track marginal VRAM per # of in-flight requests

12/22/2025

with model compilation, no cpu_offload:
    256x256 output image:
        Batch=1, 256x256 input: 15735MiB /  23028MiB, 5.5s 
        Batch=2, 256x256 input: 15867MiB /  23028MiB, 9.5s 
        Batch=1, 200x200 input: 15867MiB /  23028MiB, 4.95s
        Batch=2, 200x200 input: 15867MiB /  23028MiB, 7.9s
    512x512 output image:
        Batch=1, 256x256 input: 16337MiB /  23028MiB, 8.82s 
        Batch=2, 256x256 input: 17495MiB /  23028MiB, 15.2s 
        Batch=1, 200x200 input: 17495MiB /  23028MiB, 8.7s
        Batch=2, 200x200 input: 17495MiB /  23028MiB, 15.5s

Learnings:
    - I wrongly assumed that the model is a pipeline of layers, so I can have multiple image generations in flight sequentially. 
    - I can revert to batching by shrinking canvas dim to 200x200, while generation dim is 512x512
    - 512x512 output results in poor prompt adherence. Raising guidance_scale 3.5->5 fixed it.
  
1/4/2026

Switching to float16 lowered inference from 8.7 to 8.3 seconds 
max-autotune (just transfomer) lowered inference time from 8.3 to 7.57 seconds
    - compialtion takes ~7 minutes

    


"""
