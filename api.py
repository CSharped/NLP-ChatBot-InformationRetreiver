from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load data and model
model = SentenceTransformer('all-MiniLM-L6-v2')
chats_df = pd.read_csv('./data/news_CSV.csv', encoding='latin1')
questions = chats_df['context'].tolist()
question_embeddings = np.load('./embeddings/question_embeddings.npy')

# Create FastAPI app
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/questions")
async def get_top_questions():
    """Get top 10 questions from the dataset"""
    try:
        top_questions = questions[:10]
        return {"questions": top_questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def get_answer(request: QuestionRequest):
    """Get answer for a question"""
    try:
        # Get question embedding and find most similar question
        query_embedding = model.encode([request.question])
        similarities = cosine_similarity(query_embedding, question_embeddings)
        most_similar_idx = np.argmax(similarities)
        similarity_score = float(similarities[0][most_similar_idx])
        
        # Get the answer
        similar_question = questions[most_similar_idx]
        answer = chats_df[chats_df['context'] == similar_question]['response']
        
        # If similarity is too low, return a default message
        if similarity_score < 0.70:
            answer = "Sorry, I don't have a good answer for that question."
        
        return {
            "question": request.question,
            "answer": answer,
            "similar_question": similar_question,
            "similarity_score": similarity_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)