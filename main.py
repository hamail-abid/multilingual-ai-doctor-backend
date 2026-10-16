from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import query_rag

app = FastAPI(title="Multilingual AI Doctor API")

class ChatRequest(BaseModel):
    question: str
    language: str = "Urdu"

@app.get("/")
def home():
    return {"message": "Multilingual AI Doctor Backend Server is Running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        response_text = query_rag(request.question, language=request.language)
        return {
            "status": "success",
            "question": request.question,
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))