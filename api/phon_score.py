from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from pprint import pprint
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import pickle



def calc_phonatic_score (context , title , llm , db  ):
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a title verification assistant for the Press Registrar General of India. 
        You are given a list of existing titles and an input fed by the user.
        1. Your task is to remove the prefixes and suffixes from the input title first.
        2. Then you need to use metaphone phonetic algorithm to calculate the similarity between the input title and the existing titles.
        3. You need to return all the titles that are phonatically similar to the input title.
        4. Make sure to return the value of the similarity between the input title and the existing titles that we get using the metaphone phonetic algorithm.
        5. Strip the prefixes and suffixes from the input title before calculating the phonetic similarity.
        Output Format:
        {{
            {{
                "similar titles": {{
                    "title1": {{
                        "distance": 0.5,
                        "score": 50
                    }},
                    "title2": {{
                        "distance": 0.2,
                        "score": 80
                    }}
                    ...
                }} 
            }}
        }}
        input: {input}
        existing titles: {context}
        
        
        """
    )
    retriever = db.as_retriever()   
    document_chain_phonatic = create_stuff_documents_chain(llm, prompt)
    retreival_chain_phonatic = create_retrieval_chain(retriever, document_chain_phonatic)
    res = retreival_chain_phonatic.invoke({"input": title, "context": context})
    return res['answer']