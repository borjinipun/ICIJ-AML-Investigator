from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import investigate

app = FastAPI(title="AML GenAI Investigator")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/investigate")
def investigate_api(req: QueryRequest):
    return investigate(req.question)