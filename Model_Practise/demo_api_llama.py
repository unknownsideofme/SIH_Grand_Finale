from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaLLM  # Updated import for Ollama
import uvicorn  # Added uvicorn import
import os
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

# Initialize FastAPI
app = FastAPI()

# Paths for FAISS index and metadata
faiss_index_path = "faiss_index"
metadata_path = "faiss_metadata.pkl"

# Load FAISS index with OpenAI Embeddings
db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Load metadata from pickle
with open(metadata_path, "rb") as file:
    db.docstore = pickle.load(file)

# Create the retriever and chain
context = db
prompt_template = """You are a title verification assistant for the Press Registrar General of India. Your task is to evaluate new title submissions based on similarity with existing titles, compliance with disallowed words, prefixes/suffixes, and other guidelines.

**Requirements**:
1. Calculate and return the similarity score between the input title and a list of provided existing titles. The similarity should account for:
   - Phonetic similarity (e.g., Soundex or Metaphone).
   - Common prefixes/suffixes (e.g., "The," "India," "News").
   - Spelling variations or slight modifications.
   - Semantic similarity, including translations or similar meanings in other languages.
2. If the input title violates any of the following guidelines, provide a clear reason for rejection:
   - Contains disallowed words (e.g., Police, Crime, Corruption, CBI, Army).
   - Combines existing titles (e.g., "Hindu" and "Indian Express" forming "Hindu Indian Express").
   - Adds periodicity (e.g., "Daily," "Weekly," "Monthly") to an existing title.
3. Provide a probability score for verification using the formula:  
   `Verification Probability = 100% - Similarity Score`.
4. Include actionable feedback for users to modify and resubmit their titles if rejected.

**Example Input**:  
- Input Title: "Daily Jagran News"  
- Existing Titles: ["Jagran News", "Daily Samachar", "Morning Express"]  

**Example Output**:  
- Similarity Score: 85%  
- Verification Probability: 15%  
- Rejection Reasons:  
  1. Similar to "Jagran News" (phonetic similarity).  
  2. Contains a disallowed prefix ("Daily").  
- Feedback: Remove the prefix "Daily" and ensure the title is unique.

Now, evaluate the following:

**Input Title**: {input}  
**Existing Titles**: {context}  
**Disallowed Words**: ["Police", "Crime", "Corruption", "CBI", "Army"]  
**Disallowed Prefixes/Suffixes**: ["Daily", "Weekly", "Monthly", "The", "India", "News"]
Now, evaluate the following and return your analysis in JSON format:
{{
    "similarity_score": "<similarity_score>",
    "verification_probability": "<verification_probability>",
    "rejection_reasons": "<rejection_reasons>",
    "suggestions": "<suggestions>"
}}
"""

# Create the PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "title_to_verify"],
    template=prompt_template
)

# Initialize the LLM model (Ollama)
llm = OllamaLLM(model="llama2")

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Retriever chain
retriever = db.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

# Pydantic Models for Request and Response
class TitleRequest(BaseModel):
    title: str

class VerificationResponse(BaseModel):
    similarity_score: float
    verification_probability: float
    rejection_reasons: list[str]
    suggestions: list[str]

@app.post("/verify", response_model=VerificationResponse)
async def verify_title(request: TitleRequest):
    title = request.title
    
    # Call the retriever chain to get a detailed response
    response = retriever_chain.invoke({"input": title})
    
    # Debug the raw response to see its structure
    print(response)
    
    # Parse the response for individual elements
    similarity_score = response.get("similarity_score", 0.0)
    verification_probability = response.get("verification_probability", 0.0)
    rejection_reasons = response.get("rejection_reasons", [])
    suggestions = response.get("suggestions", [])

    # Return structured response
    return VerificationResponse(
        similarity_score=similarity_score,
        verification_probability=verification_probability,
        rejection_reasons=rejection_reasons,
        suggestions=suggestions
    )

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
