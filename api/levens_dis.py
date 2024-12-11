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
        2. Then you need to calculate the Levenshtein distance between the input title and the existing titles, but only for those titles that match the given title string-wise. 
        3. There can be multiple similar titles in the existing titles list.
        4. Make sure to remove the prefix and suffix from the existing titles as well.
        5. Calculate the Levenshtein distance consistently using the stripped versions of the titles. Include **all matching results**, even those with low scores.
        6. Provide a score out of 100 based on the Levenshtein distance.
        7. Normalize both the input and existing titles to lowercase before calculating the distance to avoid case-related inconsistencies.
        8. Strip the prefixes and suffixes from the input title before calculating the Levenshtein distance.
        9. Include all matching results in the output, as there are times when you miss some results, which can confuse the user.
        10. Ensure no newline (`\n`) characters appear in the output.
        11. For reproducibility, ensure that the Levenshtein score calculation method is consistent and deterministic across multiple runs.

        Your output should be in JSON format, with all matching results, including low scores. Make sure the scores and results are accurate and exhaustive.

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

