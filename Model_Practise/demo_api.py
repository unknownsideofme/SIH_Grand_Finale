import os
import pickle
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from langchain_ollama import OllamaLLM  # New import for Ollama
from langchain.prompts import PromptTemplate
import uvicorn
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Ensure API key is set either via environment or code
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")  # Make sure your OPENAI_API_KEY environment variable is set

if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

# Set the API key using OpenAI package
import openai
openai.api_key = api_key  # Set the API key directly for OpenAI package

# Initialize FastAPI
app = FastAPI()

# Paths for FAISS index and metadata
faiss_index_path = "faiss_index"
metadata_path = "faiss_metadata.pkl"

# Load FAISS index with OpenAI Embeddings
db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Load metadata from pickle file
with open(metadata_path, "rb") as file:
    db.docstore = pickle.load(file)

# CORS configuration to allow all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Define the prompt template for title verification
prompt_template = """
You are a title verification assistant for the Press Registrar General of India. Your task is to evaluate new title submissions based on similarity with existing titles, compliance with disallowed words, prefixes/suffixes, and other guidelines.

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
"""

# Create the PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=prompt_template
)

# Initialize the LLM model (Ollama)
llm = OllamaLLM(model="llama2")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Retriever chain
retriever = db.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)
#Define a Pydantic model for the request body
class TitleRequest(BaseModel):
    title: str

@app.post("/verify")
async def verify_title(request: TitleRequest):
    title = request.title
    # Invoke the retriever chain with input title
    response = retriever_chain.invoke({"input": title})
    return response



if __name__ == "__main__":
    # Specify the port or use default
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
