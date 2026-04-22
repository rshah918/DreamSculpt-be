from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    image_prompt: str = Field(description="base64 encoded image string")
    text_prompt: str = Field(description="text prompt for image generation", max_length=999)
