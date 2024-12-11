from dotenv import load_dotenv
import os
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from api2.semantic_search import semantic_search
from api2.phonatics_search import phonatic_search
from api2.suggestions import calc_suggestions
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
 

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

openai_api_key = os.getenv("OPENAI_API_KEY")

llm2 = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

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




# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#disallowed words

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
        if word.lower() in title:
            return True
    return False

#remove /n
def remove_newlines(obj):
    if isinstance(obj, str):
        try:
            # Try to parse the string as JSON (if it's JSON-like)
            parsed_obj = json.loads(obj)
            # If it can be parsed, recursively clean the nested structure
            return remove_newlines(parsed_obj)
        except json.JSONDecodeError:
            # If it cannot be parsed (it's just a regular string), remove newlines and spaces
            return obj.replace("\n", "").replace("  ", "")
    
    elif isinstance(obj, list):  # If it's a list, process each element
        return [remove_newlines(item) for item in obj]
    
    elif isinstance(obj, dict):  # If it's a dictionary, process each key-value pair
        return {key: remove_newlines(value) for key, value in obj.items()}
    
    return obj  # For other data types, return as-is

#class for the input title from API request 
class TitleInput(BaseModel):
    title: str
    
def get_top_15(json_data):
    """
    Extracts the top 15 elements from the JSON data based on the similarity score.

    Args:
        json_data (str or dict): JSON string or dictionary containing the "similar titles".

    Returns:
        str: JSON string with the top 15 titles.
    """
    # Check if the input is a string and convert it to a dictionary
    if isinstance(json_data, str):
        data = json.loads(json_data)
    elif isinstance(json_data, dict):
        data = json_data
    else:
        raise TypeError("The input must be a JSON string or a dictionary.")
    
    # Get the "similar titles" dictionary
    similar_titles = data.get("similar titles", {})
    
    # Sort the titles based on the score in descending order
    sorted_titles = sorted(similar_titles.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Take the top 15 titles
    top_15_titles = dict(sorted_titles[:15])
    
    # Return the top 15 in JSON format
    return json.dumps({"similar titles": top_15_titles}, indent=4)



@app.post("/sliftex/similarity")

def search_title(title_input: TitleInput):
    title= title_input.title
    response_semantic = semantic_search(title)
    top_15_semantic = get_top_15(response_semantic)
    
    response_phonatic = phonatic_search(title)
    top_15_phonatic = get_top_15(response_phonatic)
    res = json.dumps( {"semantic_search": top_15_semantic, "phonatic_search": top_15_phonatic})
    res = remove_newlines(res)
    suggest = calc_suggestions(res, title, llm)
    suggest = remove_newlines(json.dumps(suggest))
    res = json.dumps( {"semantic_search": response_semantic, "phonatic_search": response_phonatic, "suggestions": suggest})
    res = remove_newlines(res)
    return  res

if __name__ == "__main__":
    uvicorn.run(app)
    

@app.post("/sliftex/update")
def update():
    return "Update successful"