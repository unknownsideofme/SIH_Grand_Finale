from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
import pickle
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize FastAPI app
app = FastAPI()

# Load FAISS index
faiss_index_path = "faiss_index"
metadata_path = "faiss_metadata.pkl"
db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
with open(metadata_path, "rb") as file:
    db.docstore = pickle.load(file)

# Define LLM and Prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

prompt_template = """

"""
