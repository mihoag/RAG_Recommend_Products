import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader

# Path to the CSV file
csv_file_path = "data/products_v2.csv"


# Load the CSV data
try:
    data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
except FileNotFoundError:
    print(f"Error: File {csv_file_path} not found.")
    exit()

# Check if the CSV file contains expected columns
expected_columns = ['Name', 'Short Description', 'Full Description', 'ID', 'Category', 'Brand', "URI" , "Main Image", "Detail Product"]
for col in expected_columns:
    if col not in data.columns:
        print(f"Error: Column '{col}' not found in CSV.")
        exit()

# Function to prepare the documents from the CSV data
def prepare_documents(data):
    documents = []
    for index, row in data.iterrows():
        # Combine relevant fields for embedding
        content = f"Product Name: {row['Name']}\nShort Description: {row['Short Description']}\nFull Description: {row['Full Description']}\nDetail Product: {row['Detail Product']}\nCategory: {row['Category']}\nBrands: {row['Brand']}"
       
          # Store necessary fields in metadata
        metadata = {
            "ID": row['ID'],
            "Name": row['Name'],
            "Category": row['Category'],
            "Brand": row['Brand'],
            "Uri" : row['URI'],
            "MainImage" : row["Main Image"],
            "ShortDescription": row['Short Description']
        }
        # Add metadata (e.g., ID, Category, Brand) for retrieval
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents

# Prepare the documents
documents = prepare_documents(data)

# Split the documents into manageable chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Create embeddings using a HuggingFace pre-trained model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Create the FAISS vector store from the chunks
try:
    db = FAISS.from_documents(chunks, embedding_model)
except Exception as e:
    print(f"Error while creating FAISS database: {e}")
    exit()

# Save the FAISS vector database to a local directory
DB_FAISS_PATH = 'vectorstore_products/db_faiss'
db.save_local(DB_FAISS_PATH)

print("FAISS vector database created and saved successfully!")