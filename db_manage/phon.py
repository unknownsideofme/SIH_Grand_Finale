from pinecone import Pinecone
import os
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from metaphone import doublemetaphone

load_dotenv()

# Load environment variables
pinecone_key = os.getenv("PINECONE_API_KEY")


#query data
query = "Shramik"
query_meta,query_metb = doublemetaphone(query)
from langchain_ollama.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model ="mxbai-embed-large")
embedded_title = embeddings.embed_query(query_meta)


# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

index_semantic = "phonaticsearch"
indexsemantic = pc.Index(index_semantic)

# Initialize OpenAI model and embeddings (you can replace it with any other LLM you're using)

def semantic_search(embedded_title):
    # Generate the query embedding for the input title
    

    # Perform the semantic search in Pinecone using the embedding
    results = indexsemantic.query(
        vector=embedded_title,  # The embedding for the query
        top_k=20,  # Number of results to return
        include_metadata=True  # Whether to include metadata in the results
    )
    return results


print(semantic_search(embedded_title))
    
