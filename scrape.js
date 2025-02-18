import { NextApiRequest, NextApiResponse } from "next";
import axios from "axios";
import * as cheerio from "cheerio";
import * as faiss from "faiss-node";
import { SentenceTransformer } from "@sentence-transformers/sentence-transformers";

// Initialize FAISS index
const d = 384; // Embedding dimension for MiniLM
const index = new faiss.IndexFlatL2(d);
const documents = [];

// Load embedding model
const embedder = new SentenceTransformer("all-MiniLM-L6-v2");

// Function to scrape webpage content
const scrapeContent = async (url) => {
  try {
    const response = await axios.get(url);
    const $ = cheerio.load(response.data);
    const paragraphs = $("p")
      .map((_, p) => $(p).text())
      .get();
    return paragraphs.join(" ");
  } catch (error) {
    return `Error scraping ${url}: ${error.message}`;
  }
};

// API route handler
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Only POST requests allowed" });
  }

  const { urls } = req.body;
  if (!urls || urls.length === 0) {
    return res.status(400).json({ message: "No URLs provided" });
  }

  for (const url of urls) {
    const text = await scrapeContent(url);
    if (text) {
      const embedding = await embedder.encode([text]);
      documents.push(text);
      index.add(embedding[0]);
    }
  }

  res.json({ message: "Scraping completed.", num_docs: documents.length });
}
