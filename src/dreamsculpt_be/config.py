import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Literal

env_path = Path(__file__).parent / "env" / ".env.local"
load_dotenv(dotenv_path=env_path)

MODEL_PATH = "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q5_0.gguf"
MODEL_PROMPT = "Transform this rough sketch into an awe-inspiring, photorealistic image. Render every element as if it exists in the real world with natural textures, lifelike materials, and fine detail. Add realistic depth, dramatic lighting, and atmospheric effects such as reflections, sky, and shadows. The final result should look like a stunning photograph, true to the layout of the sketch but elevated into a vivid, breathtaking real-world scene"
GENERATION_HEIGHT = 512
GENERATION_WIDTH = 512
NUM_INFERENCE_STEPS = 15
GUIDANCE_SCALE = 5.0
MAX_BATCH_SIZE = 2

#Generic External Model Configs
EXTERNAL_MODEL: None | Literal["GROK"] | Literal["GEMINI"] = "GROK"
ASPECT_RATIO = "1:1"
RESOLUTION = "1K"
GENERATIONS_REMAINING = 100

# Gemini Configs
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", "MISSING_API_KEY")

# Grok Configs
GROK_MODEL = "grok-imagine-image"
GROK_API_KEY = os.getenv("XAI_API_KEY", "MISSING_API_KEY")

# Logging
USER_LOGS_S3_BUCKET = "dreamsculpt-logs"