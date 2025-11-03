# -*- coding: utf-8 -*-
"""
Allen Jones with Gemini
2025.9.29

Main method to take in user query, gather relevant data from DB and send that
to Gemini for write up.
"""

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- 1. Setup ---
# Load environment variables and configure the Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the ChromaDB client and the embedding model
client = chromadb.PersistentClient(path="db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Get the existing collections
bible_collection = client.get_collection(name="hebrew_bible")
bp_collection = client.get_collection(name="bible_project")

print("✅ Setup complete. Ready to receive queries.")

# --- 2. Retrieval Function ---
def retrieve_context(verse_reference, n_results=5):
    """
    Retrieves context for a given verse from the ChromaDB collections.
    """
    # Step 1: Get the original verse text by its ID
    original_verse_data = bible_collection.get(ids=[verse_reference])
    if not original_verse_data['documents']:
        return {"error": "Verse not found."}
    original_verse_text = original_verse_data['documents'][0]

    # Step 2: Query for relevant BibleProject chunks
    bp_results = bp_collection.query(
        query_texts=[original_verse_text],
        n_results=n_results
    )
    bp_context = "\n".join(bp_results['documents'][0])

    # Step 3: Query for potential re-use in the Bible
    # We ask for n_results + 1 because the most similar result will be the verse itself
    reuse_results = bible_collection.query(
        query_texts=[original_verse_text],
        n_results=n_results + 1 
    )

    # Filter out the original verse from the results to find others
    reuse_context = []
    for ref, text in zip(reuse_results['ids'][0], reuse_results['documents'][0]):
        if ref != verse_reference:
            reuse_context.append(f"- {ref}: {text}")

    return {
        "original_verse_text": original_verse_text,
        "bp_context": bp_context,
        "reuse_context": "\n".join(reuse_context)
    }


# --- 3. Generation Function ---
def generate_answer(verse_reference):
    """
    Retrieves context and generates a final answer using the LLM.
    """
    # First, get our raw materials from the database
    context = retrieve_context(verse_reference)

    if "error" in context:
        return context["error"]

    # This is our prompt template. We'll fill it with the retrieved context.
    prompt_template = f"""
You are a helpful biblical studies assistant. Your goal is to provide clear and 
insightful analysis on a given Bible verse based on the provided context.

Here is the verse and the context I have gathered:

### Original Verse:
{context['original_verse_text']}

### BibleProject Commentary Context:
{context['bp_context']}

### Potential Biblical Re-use Context:
{context['reuse_context']}

---
**Your Task:**
Based on all the context above, please provide a two-part analysis for the 
verse "{verse_reference}":
1.  **BibleProject Insights:** Summarize any relevant commentary from the 
BibleProject transcripts. If no specific commentary is found, state that.
2.  **Potential Re-use:** List the verses that show potential literary re-use 
and briefly explain the possible connection for each one.
"""

    # Generate the content using the Gemini model
    model = genai.GenerativeModel('models/gemini-pro-latest')
    response = model.generate_content(prompt_template)
    
    return response.text

# --- 4. Interactive Console Loop ---
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a verse to analyze (e.g., Genesis 1:1) or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        
        print("\n🧠 Thinking...")
        answer = generate_answer(user_input)
        
        print("\n--- Analysis ---")
        print(answer)
        print("-" * 20)



