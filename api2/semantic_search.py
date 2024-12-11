from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama.embeddings import OllamaEmbeddings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever



# Load environment variables
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

index_semantic = "sliftex"
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


def convert_documents_to_json(documents):
    """
    Converts a list of Document objects to the desired JSON format, using `page_content` as titles,
    and sorts the results by score in descending order.

    Args:
        documents (list): List of Document objects, each containing metadata and page_content.

    Returns:
        dict: Reformatted and sorted JSON structure.
    """
    formatted_response = {"similar titles": {}}

    # Sort documents by the 'score' in descending order
    sorted_documents = sorted(
        documents,
        key=lambda doc: doc.metadata.get('score', 0.0),  # Safely get the score from metadata
        reverse=True
    )

    for doc in sorted_documents:
        title = doc.page_content  # Extract title from page_content
        score = doc.metadata.get('score', 0.0)  # Extract score from metadata

        # Capitalize the first word after a space
        formatted_title = ' '.join([word.capitalize() for word in title.split()])

        # Add the title and score to the formatted response
        formatted_response["similar titles"][formatted_title] = {
            "score": score
        }

    return formatted_response







#creating pinecone retriever
encoded_docs = BM25Encoder().load("./api2/document.json")
retriever = PineconeHybridSearchRetriever(index=indexsemantic, sparse_encoder=encoded_docs, embeddings=embed_model)




# Initialize OpenAI model and embeddings (you can replace it with any other LLM you're using)

def semantic_search( title):
    # Generate the query embedding for the input title
    title = clean_text(title)
    # Perform the semantic search in Pinecone using the embedding
    results = retriever.invoke(title)
    print(results)
    # Convert the Pinecone response to the desired JSON format
    results = convert_documents_to_json(results) 
    
    return results


res = semantic_search(" The Hindu")