import pandas as pd
import pickle
import os

# Data Preprocessing
# Step 1: Open the file in read-binary mode
with open('Model_Practise/data.pkl', 'rb') as file:
    # Step 2: Load the data from the file
    data = pickle.load(file)

df = pd.DataFrame(data)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Rename column
df.rename(columns={'Title Name': 'title'}, inplace=True)

# Convert all names to lowercase
df['title'] = df['title'].str.lower()

# Data Ingestion
from langchain_community.document_loaders import DataFrameLoader
loader = DataFrameLoader(df)
documents = loader.load()

# Transform
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)

# Vector Embedding with Hugging Face and Pinecone
from langchain_community.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
import pinecone

# Initialize Hugging Face SentenceTransformer model (MS MARCO)
hf_model = SentenceTransformer('msmarco-distilbert-base-v4')  # Ensure the model is downloaded or available

# Initialize Pinecone
pinecone.init(
    api_key="YOUR_PINECONE_API_KEY",  # Replace with your Pinecone API key
    environment="YOUR_PINECONE_ENVIRONMENT"  # Replace with your Pinecone environment
)

# Create a new Pinecone index or connect to an existing one
index_name = "langchain-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=hf_model.get_sentence_embedding_dimension())

# Connect to the Pinecone index
index = pinecone.Index(index_name)

# Create embeddings and add to Pinecone
texts = [doc.page_content for doc in split_docs]
metadatas = [doc.metadata for doc in split_docs]
embeddings = hf_model.encode(texts, convert_to_tensor=True)

# Convert embeddings to lists and upload to Pinecone
vectors = [(str(i), embedding.tolist(), metadata) for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas))]
index.upsert(vectors)

print("Documents successfully stored in Pinecone!")
