from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
import uvicorn
import json
import os
from dotenv import load_dotenv
import pickle
from levens_dis import calc_levens_score
from phon_score import calc_phonatic_score
from semantic_score import calc_semantic_score
from prefix_suffix import Pref_suff
from suggestions import calc_suggestions
import re
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

langsmith_api_key = os.getenv("langsmith_api_key")

pinecone_api = os.getenv("PINECONE_API_KEY")


# Set additional environment variables programmatically
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "SLIFTEX"


 #Initialize the LLM model (OpenAI GPT-3.5-turbo)
llm = ChatOpenAI(model = "gpt-40-mini", api_key=api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api)

# Connect to the existing index
index_name = "sliftex"  # Replace with your Pinecone index name
index = pc.Index(index_name)



embeddings = OllamaEmbeddings(model ="mxbai-embed-large")
query = "The VicharDhara Express"
query_embedding = embeddings.embed_query(query)

# Perform the search using keyword arguments
results = index.query("Sri Krishna")

print(results)

