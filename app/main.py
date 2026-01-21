from fastapi import FastAPI
from pydantic import BaseModel
from app.embedding import get_embedding
from app.matcher import ResumeMatcher
from app.models import MatchResponse
from app.llm_explainer import generate_explanation

app = FastAPI(title="GenAI Resume Matcher")

# Dummy resume database - removed and replaced with an empty dict 
#In-memory storage (temporary) 
resumes = {}
resume_embeddings = []
matcher = None

# Create embeddings once at startup
resume_embeddings = [get_embedding(text) for text in resumes.values()]


@app.get("/")
def home():
    return {"message": "Resume Matcher API is running 🚀"}

@app.get("/match", response_model=MatchResponse)
def match(job_text: str):
    if matcher is None:
        return {
            "resume_id": "",
            "similarity_score": 0.0,
            "explanation": "No resumes available. Please add resumes first."
        }

    job_embedding = get_embedding(job_text)
    idx, score = matcher.match(job_embedding)

    resume_id = list(resumes.keys())[idx]
    explanation = generate_explanation(resumes[resume_id], job_text)

    return MatchResponse(
        resume_id=resume_id,
        similarity_score=round(score, 3),
        explanation=explanation
    )


class ResumeInput(BaseModel):
    resume_id: str
    text: str 

@app.post("/add_resume")
def add_resume(resume: ResumeInput):
    global matcher

    resumes[resume.resume_id] = resume.text
    # Update embeddings
    resume_embeddings.append(get_embedding(resume.text))
    
    matcher = ResumeMatcher(resume_embeddings)

    return {
        "message": f"Resume {resume.resume_id} added successfully.",
        "total_resumes": len(resumes)
    }