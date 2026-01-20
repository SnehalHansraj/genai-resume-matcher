from fastapi import FastAPI
from app.embedding import get_embedding
from app.matcher import ResumeMatcher
from app.models import MatchResponse
from app.llm_explainer import generate_explanation

app = FastAPI(title="GenAI Resume Matcher")

# Dummy resume database
resumes = {
    "resume_1": "Python developer with experience in ML, NLP, FastAPI and SQL.",
    "resume_2": "Java backend developer with Spring Boot and REST API experience."
}

# Create embeddings once at startup
resume_embeddings = [get_embedding(text) for text in resumes.values()]
matcher = ResumeMatcher(resume_embeddings)

@app.get("/")
def home():
    return {"message": "Resume Matcher API is running 🚀"}

@app.get("/match", response_model=MatchResponse)
def match(job_text: str):
    job_embedding = get_embedding(job_text)
    idx, score = matcher.match(job_embedding)

    resume_id = list(resumes.keys())[idx]
    explanation = generate_explanation(resumes[resume_id], job_text)

    return MatchResponse(
        resume_id=resume_id,
        similarity_score=round(score, 3),
        explanation=explanation
    )
