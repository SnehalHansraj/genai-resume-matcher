# GenAI Resume Matcher

A FastAPI-based backend application that matches resumes to job descriptions
using semantic embeddings and generates human-readable explanations using
Generative AI.

## Features
- Semantic resume–job matching using transformer embeddings
- Similarity search with FAISS
- Explainable AI using a GenAI-based explanation layer
- REST API built with FastAPI

## Tech Stack
- Python
- FastAPI
- SentenceTransformers
- FAISS
- OpenAI API (with fallback handling)

## How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
