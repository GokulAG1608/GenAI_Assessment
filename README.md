# RAG Chatbot using Google Gemini

This is a Retrieval-Augmented Generation (RAG) chatbot that scrapes web content, stores it in a vector database, and generates responses using Google Gemini.

---

## **1. Installation & Running Locally**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- pip
- Postman (optional, for testing API endpoints)
- Google Gemini API key (Get it from [Google AI Studio](https://ai.google.dev))

##**Tech Stack Used**

Backend: Flask (Python)
Scraping: BeautifulSoup, Requests
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Vector Database: FAISS (Facebook AI Similarity Search)
LLM: Google Gemini API
Deployment (Optional): Vercel, AWS, or Render
