import os
from dotenv import load_dotenv
import openai

load_dotenv()

def generate_explanation(resume_text: str, job_text: str) -> str:
    """
    Safely generate an explanation using OpenAI.
    If OpenAI fails, return a fallback explanation.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    # Fallback if API key is missing
    if not api_key:
        return (
            "The candidate appears to match the job based on relevant skills "
            "and experience mentioned in the resume."
        )

    openai.api_key = api_key

    try:
        prompt = f"""
You are an AI recruitment assistant.

JOB DESCRIPTION:
{job_text}

RESUME:
{resume_text}

Explain in 3–4 lines why the candidate matches or does not match the job.
Focus on skills and experience.
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        # VERY IMPORTANT: never crash the server
        print("GenAI error:", e)

        return (
            "The candidate matches the job based on overall technical background "
            "and alignment with the role requirements."
        )
    