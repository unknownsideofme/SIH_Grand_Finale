from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from pprint import pprint
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

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

def Pref_suff(context,  title, llm, db ):
   
    # Define the prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a title verification assistant for the Press Registrar General of India.

    1. You are given a list of existing titles and an input title.
    2. Your task is to find out in how many existing titles the prefixes and suffixes of the input title is used.
    3. Return the a percentage that represents the frequency of the prefix or suffix in the titles.
    4. "frequency%"= (number of titles with the prefix or suffix / total number of titles) * 100 .
    5. The words like "The" "A" "An" are considered as prefixes.
    6. The words like "Ltd" "Inc" "Co" are considered as suffixes.
    7. In case of exeamples like "The New York Times" "The" is the prefix and "Times" is the suffix.
    8. In case of examples like "The New York Express" since Express has been repeated in the database as a suffix, it should be considered as a suffix.
    9. Returen only the precentage of the words they are repeated as prefixes and suffixes.
    10. Dont return the actual no of titles
    Output Format:
    {{
        "prefixes": {{
            "prefix1": %,
            "prefix2": %,
            ...
        }},
        "suffixes": {{
            "suffix1": %,
            "suffix2": %,
            ...
        }}
    }}
    input : {input}
    context : {context}
    """)
    retriever = db.as_retriever()
    # Create document chain and retrieval chain
    document_chain_semantic = create_stuff_documents_chain(llm, prompt)
    retrieval_chain_semantic = create_retrieval_chain(retriever, document_chain_semantic)
    
    # Invoke the retrieval chain with extracted context
    res = retrieval_chain_semantic.invoke({"input": title, "context": context})
    return res['answer']


