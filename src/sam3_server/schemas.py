from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    filepaths: List[str]
    prompt: str