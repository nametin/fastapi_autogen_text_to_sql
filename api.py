from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modules.GroqModule import GroqModule
import os 
from helpers import Helpers


app = FastAPI()

class SQLRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/text-to-sql/")
async def text_to_sql(request: SQLRequest):
    # sample question
    # "What is the issue date of the volume with the minimum weeks on top?.""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        print("API KEY: \n",groq_api_key)
        groq_module = GroqModule(groq_api_key)
        sql_query = groq_module.text_to_sql(request.question)
        
        return {"sql_query": sql_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
