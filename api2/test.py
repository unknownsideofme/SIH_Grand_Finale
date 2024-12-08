from dotenv import load_dotenv
import os
from langchain_ollama.embeddings import OllamaEmbeddings
from metaphone import doublemetaphone
from langchain_groq import ChatGroq
#from semantic_search import semantic_search
from phonatics_search import phonatic_search

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

langsmith_api_key = os.getenv("langsmith_api_key")


# Set additional environment variables programmatically
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "SLIFTEX"


#llm
llm = ChatGroq(api_key=api_key, model="llama3-groq-70b-8192-tool-use-preview")
# Embed the query
embed_model = OllamaEmbeddings(model="mxbai-embed-large")
title = "The Shramik Vichar Shakti"
embedded_title = embed_model.embed_query(title)
title_metaphone = doublemetaphone(title)[0]
embedded_metaphone = embed_model.embed_query(title_metaphone)

print(phonatic_search(embedded_metaphone, title, llm))