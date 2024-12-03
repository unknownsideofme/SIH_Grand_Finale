from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from pprint import pprint
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import pickle



def calc_levens_score (context , title , llm , db  ):
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a title verification assistant for the Press Registrar General of India. 
        You are given a list of existing titles and an input fed by the user.
        1. Your task is to remove the prefixes and suffixes from the input title first.
        2. Then you need to calculate the Levenshtein distance between the input title and the existing titles only for those titles that match the given title string-wise.
        3. There can be multiple similar titles in the existing titles list.
        4. make sure to remove the prefix and suffix of the existinf titles also . 
        5. Then provide a score out of 100 on the basis of the Levenshtein distance.
        6. Dont return \n  in the output.
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


    document_chain_string = create_stuff_documents_chain(llm, prompt)

    retriever = db.as_retriever()

    retreival_chain_string = create_retrieval_chain(retriever, document_chain_string)
    
    res = retreival_chain_string.invoke({"input": title, "context": context})
    
    return res['answer']

