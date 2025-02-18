# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import requests
# from bs4 import BeautifulSoup
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.llms import OpenAI
# import os

# # Initialize FastAPI app
# app = FastAPI()

# # Load API Keys (Set these in your environment variables)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Initialize embedding model
# embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # Initialize FAISS vector store
# vector_store = FAISS(embedding_model)

# # Define request models
# class URLInput(BaseModel):
#     url: str

# class QueryInput(BaseModel):
#     query: str

# # Function to scrape web page
# def scrape_url(url):
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to fetch URL")
    
#     soup = BeautifulSoup(response.text, "html.parser")
#     return ' '.join([p.text for p in soup.find_all("p")])

# # Route to scrape content and store embeddings
# @app.post("/scrape/")
# def scrape_and_store(url_input: URLInput):
#     content = scrape_url(url_input.url)

#     # Split text into smaller chunks
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_text(content)

#     # Store embeddings
#     vector_store.add_texts(chunks)

#     return {"message": "Content scraped and stored successfully"}

# # Route to query RAG chatbot
# @app.post("/query/")
# def query_rag(query_input: QueryInput):
#     # Retrieve most relevant document
#     retrieved_docs = vector_store.similarity_search(query_input.query, k=3)
#     context = " ".join([doc.page_content for doc in retrieved_docs])

#     # Generate response using Google Gemini
#     gemini_response = generate_gemini_response(query_input.query, context)
    
#     return {"response": gemini_response}

# # Function to generate response using Google Gemini API
# def generate_gemini_response(query, context):
#     headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
#     data = {"query": f"{query}\n\nContext: {context}"}
#     response = requests.post("https://api.gemini.com/v1/chat/completions", json=data, headers=headers)

#     if response.status_code != 200:
#         raise HTTPException(status_code=500, detail="Gemini API call failed")

#     return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

# # Run the app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Configure Google Gemini API
genai.configure(api_key="AIzaSyAnCmWau4DCGmWyfko4DYO7XL7zxaVKvAY")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")

# Initialize embedding model (SBERT for simplicity)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index for vector storage
d = 384  # Embedding dimension for MiniLM
index = faiss.IndexFlatL2(d)
documents = []  # Store documents

def scrape_content(url):
    """Scrape text content from a webpage."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except Exception as e:
        return str(e)

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.json
    urls = data.get("urls", [])
    
    for url in urls:
        text = scrape_content(url)
        if text:
            embedding = embedder.encode([text])[0]
            documents.append(text)
            index.add(np.array([embedding]).astype('float32'))
    
    return jsonify({"message": "Scraping completed.", "num_docs": len(documents)})

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")
    
    query_embedding = embedder.encode([user_query])[0]
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=1)
    
    if I[0][0] != -1:
        retrieved_content = documents[I[0][0]]
    else:
        retrieved_content = "No relevant content found."
    
    prompt = f"Answer the following query using this context: {retrieved_content}\n\nQuery: {user_query}"
    response = model.generate_content(prompt)
    
    return jsonify({"response": response.text})

if __name__ == "__main__":
    app.run(debug=True)
