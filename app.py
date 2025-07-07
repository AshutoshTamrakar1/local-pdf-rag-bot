# pdf_chatbot_llama3.py

import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.llms import Ollama

# 1. Extract text from PDF
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# 2. Chunk text into manageable pieces
def chunk_text(text, max_length=500):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence.strip() + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip() + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# 3. Build FAISS index
def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# 4. Retrieve top-k relevant chunks
def find_relevant_chunks(question, chunks, index, model, top_k=3):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    return [chunks[i] for i in I[0]]

# 5. Generate answer using LLaMA 3 via Ollama
def generate_answer(context, question):
    llm = Ollama(model="llama3", temperature=0.1)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    return llm(prompt)

# 6. Main chatbot loop
def main():
    pdf_path = "path to ur pdf"  # Replace with your actual PDF file
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print("🔍 Creating embeddings and FAISS index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_faiss_index(chunks, model)

    print("🤖 Chatbot is ready! Ask questions about your PDF (type 'exit' to quit).")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        relevant_chunks = find_relevant_chunks(query, chunks, index, model)
        context = "\n".join(relevant_chunks)
        answer = generate_answer(context, query)
        print("\nBot:", answer.strip())

if __name__ == "__main__":
    main()