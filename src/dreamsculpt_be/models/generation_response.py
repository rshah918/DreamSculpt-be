from pydantic import BaseModel

class GenerationResponse(BaseModel):
    generated_image: str

