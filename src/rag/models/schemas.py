from pydantic import BaseModel, Field

class AskIn(BaseModel):
    doc_id: str = Field(..., description="Returned by /upload-pdf")
    question: str = Field(..., min_length=3)