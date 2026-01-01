from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    image_prompt: str = Field(description="base64 encoded image string")
