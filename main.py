import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match-resume")
async def match_resume(job_description: str = Form(...), resume: str = Form(...)):
    """
    Takes job description + resume text and returns ATS score + feedback
    """
    try:
        prompt = f"""
        Compare the following resume against the job description.
        Job Description: {job_description}
        Resume: {resume}

        Give an ATS score (0-100) and highlight key strengths and weaknesses.
        Format response as:
        Score: X/100
        Strengths: ...
        Weaknesses: ...
        Suggestions: ...
        """

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=500
        )

        return {"result": response.output_text}
    except Exception as e:
        return {"error": str(e)}
