from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama.embeddings import OllamaEmbeddings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

# Load environment variables
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

index_semantic = "sliftexsearch"
indexsemantic = pc.Index(index_semantic)

#embed the query
embed_model = OllamaEmbeddings(model="mxbai-embed-large")


# Function to clean text by removing stopwords and punctuation
stop_words = set(stopwords.words('english'))

def clean_text(text):
    
    # Tokenize text into words
    tokens = word_tokenize(text)
    # Convert to lowercase and remove punctuation and stopwords
    cleaned = [
        word.lower() for word in tokens
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return "".join(cleaned) 


def convert_pinecone_response_to_json(pinecone_response):
    """
    Converts the Pinecone response to the desired JSON format, using the metadata 'text' as titles,
    and sorts the results by score in descending order.
    
    Args:
        pinecone_response (dict): The JSON object returned by Pinecone.

    Returns:
        dict: Reformatted and sorted JSON structure.
    """
    formatted_response = {"similar titles": {}}
    
    # Assuming 'matches' contains the relevant data
    matches = pinecone_response.get('matches', [])  # Safely get 'matches' or an empty list
    
    # Extract and sort entries by score in descending order
    sorted_matches = sorted(
        matches, 
        key=lambda x: x.get('score', 0.0),  # Use score as the sorting key
        reverse=True  # Sort in descending order
    )
    
    for entry in sorted_matches:
        metadata = entry.get('metadata', {})
        title = metadata.get('text', '')  # Get the 'text' from metadata
        score = entry.get('score', 0.0)  # Replace 'score' with the correct key for the score
        
        # Capitalize the first word after a space
        formatted_title = ' '.join([word.capitalize() for word in title.split()])
        
        # Add the title and score to the formatted response
        formatted_response["similar titles"][formatted_title] = {
            "score": score
        }
    
    return formatted_response









# Initialize OpenAI model and embeddings (you can replace it with any other LLM you're using)

def semantic_search( title):
    # Generate the query embedding for the input title
    title = clean_text(title)
    embedded_title = embed_model.embed_query(title)
    # Perform the semantic search in Pinecone using the embedding
    results = indexsemantic.query(
        vector=embedded_title,  # The embedding for the query
        top_k=50,  # Number of results to return
        include_metadata=True  # Whether to include metadata in the results
    )

    # Convert the Pinecone response to the desired JSON format
    results = convert_pinecone_response_to_json(results) 
    
    return results


