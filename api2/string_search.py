from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
pinecone_key = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

index_similarity = "sliftex"
indexsimilarity = pc.Index(index_similarity)

def string_search(title):
    results = indexsimilarity.query(
        vector=title,  # The embedding for the query
        top_k=10,  # Number of results to return
        include_metadata=True  # Whether to include metadata in the results
    )
    
    return results