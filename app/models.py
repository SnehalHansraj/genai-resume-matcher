from pydantic import BaseModel

class MatchResponse(BaseModel):
    resume_id: str
    similarity_score: float
    explanation: str
