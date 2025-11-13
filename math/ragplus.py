# Math Domain - RAG+ Dual Corpus Retrieval System

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict
import time
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("MATH DOMAIN - RAG+ DUAL CORPUS RETRIEVAL SYSTEM")
print("=" * 80)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
KNOWLEDGE_INDEX = "math-knowledge-corpus"
APPLICATION_INDEX = "math-application-corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize components
print("\n[1/3] Initializing components...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    knowledge_index = pc.Index(KNOWLEDGE_INDEX)
    application_index = pc.Index(APPLICATION_INDEX)
    print(f"✓ Connected to Pinecone indexes")
    
    from groq import Groq
    llm = Groq(api_key=GROQ_API_KEY)
    print(f"✓ Gemini LLM initialized")
    
except Exception as e:
    print(f"✗ Initialization error: {e}")
    exit()


class MathRAGPlusSystem:
    """Math Domain RAG+ system with dual corpus retrieval"""
    
    def __init__(self, embedder, knowledge_idx, application_idx, llm):
        self.embedder = embedder
        self.knowledge_index = knowledge_idx
        self.application_index = application_idx
        self.llm = llm
        self.query_history = []
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant knowledge from math concepts corpus"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.knowledge_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        knowledge_docs = []
        for match in results['matches']:
            knowledge_docs.append({
                'id': match['id'],
                'score': match['score'],
                'section_reference': match['metadata']['section_reference'],
                'statutory_text': match['metadata']['statutory_text'],
                'original_question': match['metadata']['original_question'],
                'context': match['metadata'].get('context', ''),
                'type': 'KNOWLEDGE'
            })
        
        return knowledge_docs
    
    def retrieve_applications(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant problem applications"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.application_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        application_docs = []
        for match in results['matches']:
            application_docs.append({
                'id': match['id'],
                'score': match['score'],
                'case_name': match['metadata']['case_name'],
                'section_applied': match['metadata']['section_applied'],
                'year': match['metadata']['year'],
                'court': match['metadata']['court'],
                'summary': match['metadata']['case_summary'],
                'judgment_url': match['metadata']['judgment_url'],
                'type': 'APPLICATION'
            })
        
        return application_docs
    
    def hybrid_retrieve(self, query: str, knowledge_k: int = 3, application_k: int = 3) -> Dict:
        """Perform hybrid retrieval from both corpora"""
        print(f"\n🔍 Retrieving for: '{query}'")
        
        knowledge_docs = self.retrieve_knowledge(query, knowledge_k)
        application_docs = self.retrieve_applications(query, application_k)
        
        print(f"  ✓ Knowledge docs: {len(knowledge_docs)}")
        print(f"  ✓ Application docs: {len(application_docs)}")
        
        return {
            'knowledge': knowledge_docs,
            'applications': application_docs,
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
    
    def format_context(self, retrieval_results: Dict) -> str:
        """Format retrieved documents into context for LLM"""
        context_parts = []
        
        if retrieval_results['knowledge']:
            context_parts.append("=== MATH KNOWLEDGE ===")
            for i, doc in enumerate(retrieval_results['knowledge'], 1):
                context_parts.append(f"\n[K{i}] Topic: {doc['section_reference']}")
                context_parts.append(f"Concept: {doc['statutory_text']}")
                context_parts.append(f"Relevance: {doc['score']:.3f}")
        
        if retrieval_results['applications']:
            context_parts.append("\n\n=== PROBLEM APPLICATIONS ===")
            for i, doc in enumerate(retrieval_results['applications'], 1):
                context_parts.append(f"\n[A{i}] Problem: {doc['case_name']}")
                context_parts.append(f"Topic Applied: {doc['section_applied']}")
                context_parts.append(f"Solution: {doc['summary']}")
                context_parts.append(f"Relevance: {doc['score']:.3f}")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate comprehensive answer using LLM"""
        prompt = f"""You are a math expert AI assistant. Answer the user's question using the provided math knowledge and problem applications.

INSTRUCTIONS:
- Provide accurate, comprehensive mathematical information
- Reference specific concepts and problems when relevant
- Explain both the theory and its practical application
- Use clear, educational language
- Show step-by-step solutions when appropriate
- If information is insufficient, state limitations clearly

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, question: str, knowledge_k: int = 3, application_k: int = 3) -> Dict:
        """Complete RAG+ query pipeline"""
        start_time = time.time()
        
        retrieval_results = self.hybrid_retrieve(question, knowledge_k, application_k)
        context = self.format_context(retrieval_results)
        
        print("  🤖 Generating answer...")
        answer = self.generate_answer(question, context)
        
        result = {
            'query': question,
            'answer': answer,
            'retrieval_results': retrieval_results,
            'context': context,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.query_history.append(result)
        print(f"  ✓ Complete in {result['processing_time']:.2f}s")
        
        return result


# Initialize RAG+ system
print("\n[2/3] Initializing Math RAG+ system...")
rag_system = MathRAGPlusSystem(
    embedder=embedding_model,
    knowledge_idx=knowledge_index,
    application_idx=application_index,
    llm=llm
)
print("✓ Math RAG+ system ready!")

print("\n[3/3] System ready for queries!")
print("=" * 80)
print("✅ MATH DOMAIN RAG+ SYSTEM READY!")
print("=" * 80)
