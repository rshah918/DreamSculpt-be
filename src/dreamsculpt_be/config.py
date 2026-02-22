import os

MODEL_PATH = "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q5_0.gguf"
MODEL_PROMPT = "Transform this rough sketch into an awe-inspiring, photorealistic image. Render every element as if it exists in the real world with natural textures, lifelike materials, and fine detail. Add realistic depth, dramatic lighting, and atmospheric effects such as reflections, sky, and shadows. The final result should look like a stunning photograph, true to the layout of the sketch but elevated into a vivid, breathtaking real-world scene"
GENERATION_HEIGHT = 512
GENERATION_WIDTH = 512
NUM_INFERENCE_STEPS = 15
GUIDANCE_SCALE = 5.0
MAX_BATCH_SIZE = 2
USE_EXTERNAL_MODEL = True

# Gemini Configs
ASPECT_RATIO = "1:1"
RESOLUTION = "1K"
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", "MISSING_API_KEY") 