#!/usr/bin/env python3
import os
import sys
import json
import urllib.request
import re

class SimpleRAG:
    def __init__(self, server_url="http://localhost:8080/generate"):
        self.server_url = server_url
        self.documents = []
        self.chunks = []

    def load_documents(self, doc_path):
        print(f"Loading document: {doc_path}...")
        if not os.path.exists(doc_path):
            print(f"Error: {doc_path} not found.")
            return
            
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        self.documents.append(content)
        raw_chunks = re.split(r'\n\n|\.\s', content)
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk) > 10:
                self.chunks.append(chunk)
        print(f"Loaded {len(self.chunks)} chunks.")

    def compute_similarity(self, query, chunk):
        query_words = set(re.findall(r'\w+', query.lower()))
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        if not query_words or not chunk_words:
            return 0.0
        intersection = query_words.intersection(chunk_words)
        return len(intersection) / (len(query_words) + len(chunk_words) - len(intersection))

    def retrieve(self, query, top_k=2):
        print(f"Retrieving chunks for query: '{query}'...")
        scores = []
        for chunk in self.chunks:
            score = self.compute_similarity(query, chunk)
            scores.append((score, chunk))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        retrieved = [chunk for score, chunk in scores[:top_k] if score > 0]
        
        if not retrieved and self.chunks:
            retrieved = [self.chunks[0]]
            
        return retrieved

    def query_llm(self, prompt):
        data = json.dumps({"prompt": prompt, "max_tokens": 40}).encode('utf-8')
        req = urllib.request.Request(
            self.server_url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data.get("generated_text", "No response")
        except Exception as e:
            return f"Error contacting server: {e}"

    def ask(self, query):
        contexts = self.retrieve(query)
        context_str = "\n".join(contexts)
        prompt = f"context: {context_str} query: {query}"
        print(f"Formulated prompt: {prompt}")
        
        response = self.query_llm(prompt)
        print(f"LLM Answer: {response}")
        return response

if __name__ == "__main__":
    kb_file = "kb.txt"
    if not os.path.exists(kb_file):
        with open(kb_file, 'w', encoding='utf-8') as f:
            f.write("The local server runs llama model with arm sme sve2 acceleration for high throughput inference.\n\n")
            f.write("SME means Scalable Matrix Extension. SVE2 means Scalable Vector Extension version 2.\n")
            f.write("The server runs on port 8080 with 8 threads by default.\n")
            
    rag = SimpleRAG()
    rag.load_documents(kb_file)
    
    query = "sme sve2 acceleration"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    rag.ask(query)