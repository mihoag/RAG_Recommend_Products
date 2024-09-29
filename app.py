import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List



# Path to the FAISS database
DB_FAISS_PATH = 'vectorstore_products/db_faiss'

# Load the FAISS database, allowing dangerous deserialization
db = FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
                      allow_dangerous_deserialization=True)

# Function to query the FAISS database and retrieve relevant products
def retrieve_products(user_description, top_k=5):
    # Convert user's input to embeddings and retrieve top matches
    results = db.similarity_search(user_description, k=top_k)
    return results


# FastAPI app instance
app = FastAPI()
class ProductQuery(BaseModel):
    description: str
    top_k: int = 5  # Default to return top 5 products

# Response schema
class ProductResponse(BaseModel):
    name: str
    category: str
    brand: str
    image: str
    uri: str
    short_description: str

# API route to query products based on user description
@app.post("/recommend-products/", response_model=List[ProductResponse])
async def recommend_products(query: ProductQuery):
    try:
        # Perform similarity search with FAISS based on user's description
        results = retrieve_products(query.description, query.top_k)
        print(results)
        # Extract fixed fields (name, category, brand) from the results
        extracted_results = [
            {
                "name": result.metadata['Name'],  # Accessing the 'Name' field
                "category": result.metadata['Category'],  # Accessing the 'Category' field
                "brand": result.metadata['Brand'],  # Accessing the 'Brand' field
                "image": result.metadata['MainImage'],  # Accessing the 'MainImage' field
                "uri": result.metadata['Uri'],  # Accessing the 'Uri' field
                "short_description": result.metadata['ShortDescription']# Access
            }
            for result in results
        ]
        return extracted_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





