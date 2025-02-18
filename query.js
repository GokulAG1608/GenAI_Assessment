import { NextApiRequest, NextApiResponse } from "next";
import * as faiss from "faiss-node";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { SentenceTransformer } from "@sentence-transformers/sentence-transformers";

// Initialize Gemini AI API
const genAI = new GoogleGenerativeAI("AIzaSyAnCmWau4DCGmWyfko4DYO7XL7zxaVKvAY");
const model = genAI.getGenerativeModel({ model: "gemini-pro" });

// Initialize FAISS index
const d = 384;
const index = new faiss.IndexFlatL2(d);
const documents = [];

// Load embedding model
const embedder = new SentenceTransformer("all-MiniLM-L6-v2");

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Only POST requests allowed" });
  }

  const { query } = req.body;
  if (!query) {
    return res.status(400).json({ message: "Query is required" });
  }

  const queryEmbedding = await embedder.encode([query]);
  const [D, I] = index.search(queryEmbedding[0], 1);

  let retrievedContent = "No relevant content found.";
  if (I[0] !== -1) {
    retrievedContent = documents[I[0]];
  }

  const prompt = `Answer the following query using this context: ${retrievedContent}\n\nQuery: ${query}`;
  const response = await model.generateContent(prompt);

  res.json({ response: response.candidates[0].content });
}
