import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# Load .env locally (Render uses Dashboard env vars)
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is missing. Add it in Render Environment Variables.")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="ATS Resume Matcher", version="1.0")

# CORS (so frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ For production, set to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """Health check"""
    return {"message": "✅ ATS Backend running on Render!"}

@app.post("/match-resume")
async def match_resume(job_description: str = Form(...), resume: str = Form(...)):
    """
    Takes job description + resume, returns ATS score + feedback.
    """
    try:
        prompt = f"""
        You are an Applicant Tracking System (ATS).
        Compare the following resume with the job description.

        Job Description:
        {job_description}

        Resume:
        {resume}

        Respond in this format:
        Score: X/100
        Strengths: ...
        Weaknesses: ...
        Suggestions: ...
        """

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=400
        )

        return {"result": response.output_text}

    except Exception as e:
        return {"error": str(e)}
