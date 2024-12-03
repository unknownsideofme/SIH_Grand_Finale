from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import uvicorn
from pprint import pprint
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import pickle
from levens_dis import calc_levens_score
from phon_score import calc_phonatic_score
from semantic_score import calc_semantic_score



'''def remove_newlines(obj):
    if isinstance(obj, str):  # If it's a string, replace '\\n' with ''
        return obj.replace("\\n", "").replace("\\", "").replace(" ", "")
    elif isinstance(obj, list):  # If it's a list, process each element
        return [remove_newlines(item) for item in obj]
    elif isinstance(obj, dict):  # If it's a dictionary, process each key-value pair
        return {key: remove_newlines(value) for key, value in obj.items()}
    return obj  # For other data types, return as-is'''
    
    
    
    
    
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
    
    
context = db


# Initialize the LLM model (OpenAI GPT-3.5-turbo)
llm = ChatOpenAI(model = "gpt-3.5-turbo", api_key=api_key)

# Disallowed Words

disallowed_words = ["Police", "Crime", "Corruption", "CBI", "Army"]

def check_disallowed_words(title: str) -> bool:
    """
    Check if the title contains any disallowed words.

    Args:
    title (str): The title to check.

    Returns:
    bool: True if the title contains disallowed words, False otherwise.
    """
    for word in disallowed_words:
        if word.lower() in title.lower():
            return True
    return False


#class for the input title from API request 
class TitleInput(BaseModel):
    title: str


@app.post("/api/similarity")
def calculate_similarity(title_input: TitleInput):
    title = title_input.title
    
    flag = check_disallowed_words(title)
    if 
     
    

