# -*- coding: utf-8 -*-
"""
Allen Jones w/ Gemini assist
2025.9.29
Setting up a ChromaDB for BP podcast transcripts and copy of the 
WLC text. These will be the RAG repository for an AI agent that
will take a verse of the Bible, search BP's transcript and WLC
for related material.
"""



import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# --- 1. Load Environment Variables ---
load_dotenv() 
print("API Key Loaded:", os.getenv("GOOGLE_API_KEY") is not None)

# --- 2. Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path="db")
print("ChromaDB client created.")

# --- 3. Initialize the Embedding Model ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

# --- 4. Clear and Recreate ChromaDB Collections ---
# This ensures we start with a clean slate every time.
try:
    client.delete_collection(name="hebrew_bible")
    client.delete_collection(name="bible_project")
    print("Old collections deleted.")
except Exception as e:
    print("No old collections to delete, or another error occurred:", e)

bible_collection = client.create_collection(name="hebrew_bible")
bp_collection = client.create_collection(name="bible_project")
print("Fresh ChromaDB collections are ready.")

# --- 5. Process the Hebrew Bible HTML Files ---
print("\n--- Starting Hebrew Bible Processing ---")
wlc_folder_path = "Tanach.xml/Books" 
bible_verses = []
processed_references = set()

if not os.path.exists(wlc_folder_path):
    print(f"ERROR: Directory not found at {wlc_folder_path}")
else:
    for filename in sorted(os.listdir(wlc_folder_path)):
        if filename.endswith((".html", ".xml")):
            if ".DH." in filename:
                continue

            filepath = os.path.join(wlc_folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'lxml-xml')
                filename_without_ext = os.path.splitext(filename)[0]
                book_name = filename_without_ext.split('-')[1] if '-' in filename_without_ext else filename_without_ext

                for chapter in soup.find_all('c'):
                    chapter_num = chapter['n']
                    for verse in chapter.find_all('v'):
                        verse_num = verse['n']
                        reference = f"{book_name} {chapter_num}:{verse_num}"
                        if reference not in processed_references:
                            words = [word.text for word in verse.find_all('w')]
                            hebrew_text = " ".join(words)
                            if hebrew_text:
                                bible_verses.append({"reference": reference, "text": hebrew_text})
                                processed_references.add(reference)
    print(f"Successfully processed {len(bible_verses)} unique verses.")

# --- 6. Process the BibleProject Transcript Files ---
print("\n--- Starting BibleProject Transcript Processing ---")
transcripts_folder_path = "transcripts"
bp_chunks = []
if not os.path.exists(transcripts_folder_path):
    print(f"ERROR: Directory not found at {transcripts_folder_path}")
else:
    for filename in sorted(os.listdir(transcripts_folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(transcripts_folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                full_text = f.read()
                lines = full_text.splitlines()
                current_paragraph = ""
                for line in lines:
                    line = line.strip()
                    if not line and current_paragraph:
                        bp_chunks.append({"text": current_paragraph, "source": filename})
                        current_paragraph = ""
                    elif line:
                        current_paragraph += " " + line
                if current_paragraph:
                    bp_chunks.append({"text": current_paragraph.strip(), "source": filename})
    for i, chunk in enumerate(bp_chunks):
        source_name = os.path.splitext(chunk["source"])[0]
        chunk["id"] = f"{source_name}_chunk_{i}"
    print(f"Successfully processed {len(bp_chunks)} chunks from transcripts.")

# --- 7. Generate Embeddings and Populate ChromaDB (with Batching) ---
print("\n--- Starting to Populate ChromaDB ---")
print("This may take a few minutes...")

def add_to_collection_in_batches(collection, items, batch_size=4000):
    """Adds items to a ChromaDB collection in batches."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Prepare the data for this batch
        ids = [item['reference'] if 'reference' in item else item['id'] for item in batch]
        documents = [item['text'] for item in batch]
        metadatas = [{'reference': item['reference']} if 'reference' in item else {'source': item['source']} for item in batch]
        
        print(f"Adding batch of {len(batch)} items to {collection.name}...")
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        
    print(f"Finished populating {collection.name}.")

if bible_verses:
    add_to_collection_in_batches(bible_collection, bible_verses)

if bp_chunks:
    add_to_collection_in_batches(bp_collection, bp_chunks)

# --- 8. Final Verification ---
print("\n--- Database Build Complete ---")
print(f"Total items in Bible collection: {bible_collection.count()}")
print(f"Total items in BibleProject collection: {bp_collection.count()}")




