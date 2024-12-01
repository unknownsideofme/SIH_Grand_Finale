from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import uvicorn
from langchain.schema import Document

import os
import json
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

langsmith_api_key = os.getenv("langsmith_api_key")

# Set additional environment variables programmatically
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "SLIFTEX"

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
if rejection reason is empty then return empty object
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
llm = ChatOpenAI(model = "gpt-3.5-turbo", api_key=api_key)

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
    
    # Process the response from the retriever chain
    if not response:
        raise HTTPException(status_code=400, detail="No results found.")
    
    serializable_response = []
    for doc in response:
        serializable_response.append(DocumentModel(content=doc.content, metadata=doc.metadata).dict())

    # Convert the serializable response to JSON format
    response_json = json.dumps(serializable_response, indent=4)
    print(response_json)

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
