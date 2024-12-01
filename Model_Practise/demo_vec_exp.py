import pandas as pd
import numpy as np
import pickle 
import os
import pickle


#Data Preprocessing

# Step 1: Open the file in read-binary mode
with open('Model_Practise/data.pkl', 'rb') as file:
    # Step 2: Load the data from the file
    data = pickle.load(file)

df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)

df.rename(columns={'Title Name': 'text'}, inplace=True)

#Data Ingestion
from langchain.document_loaders import DataFrameLoader
loader = DataFrameLoader(df)
documents = loader.load()

#Transform

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)  


#Vector Embedding
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
db = FAISS.from_documents(split_docs[:10], OpenAIEmbeddings()) 


# Save FAISS vector store
faiss_index_path = "faiss_index"
metadata_path = "faiss_metadata.pkl"

# Save FAISS index to disk
db.save_local(faiss_index_path)

# Save metadata to pickle
with open(metadata_path, "wb") as file:
    pickle.dump(db.docstore, file)

print("FAISS index and metadata successfully stored!")









