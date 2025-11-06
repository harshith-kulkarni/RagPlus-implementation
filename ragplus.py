# Install required packages
# !pip install -q pinecone-client sentence-transformers google-generativeai pandas tqdm

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Optional
import json
import time
from datetime import datetime

print("=" * 80)
print("RAG+ DUAL CORPUS RETRIEVAL SYSTEM")
print("=" * 80)

# Configuration
PINECONE_API_KEY = "pcsk_6WURux_2kZ1ZsvMZJaDqJ1m5nRdvte2Shrfu5frLguCkp9ZWncmdEqyeXUWpQ26jqUn2eK"
GEMINI_API_KEY = "AIzaSyAT8ei1x-a_RJcieCOuq2BZAyEg68CDcR4"
KNOWLEDGE_INDEX = "legal-knowledge-corpus"
APPLICATION_INDEX = "legal-application-corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize components
print("\n[1/3] Initializing components...")
try:
    # Embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")
    
    # Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    knowledge_index = pc.Index(KNOWLEDGE_INDEX)
    application_index = pc.Index(APPLICATION_INDEX)
    print(f"✓ Connected to Pinecone indexes")
    
    # Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel("gemini-2.0-flash")
    print(f"✓ Gemini LLM initialized")
    
except Exception as e:
    print(f"✗ Initialization error: {e}")
    exit()


class RAGPlusSystem:
    """
    Advanced RAG+ system with dual corpus retrieval
    Combines knowledge base and application examples
    """
    
    def __init__(self, embedder, knowledge_idx, application_idx, llm):
        self.embedder = embedder
        self.knowledge_index = knowledge_idx
        self.application_index = application_idx
        self.llm = llm
        self.query_history = []
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant knowledge from statutory corpus
        """
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
        """
        Retrieve relevant case applications
        """
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
        """
        Perform hybrid retrieval from both corpora
        """
        print(f"\n🔍 Retrieving for: '{query}'")
        
        # Parallel retrieval
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
        """
        Format retrieved documents into context for LLM
        """
        context_parts = []
        
        # Knowledge context
        if retrieval_results['knowledge']:
            context_parts.append("=== LEGAL KNOWLEDGE ===")
            for i, doc in enumerate(retrieval_results['knowledge'], 1):
                context_parts.append(f"\n[K{i}] Section: {doc['section_reference']}")
                context_parts.append(f"Law: {doc['statutory_text']}")
                context_parts.append(f"Relevance: {doc['score']:.3f}")
        
        # Application context  
        if retrieval_results['applications']:
            context_parts.append("\n\n=== CASE APPLICATIONS ===")
            for i, doc in enumerate(retrieval_results['applications'], 1):
                context_parts.append(f"\n[A{i}] Case: {doc['case_name']}")
                context_parts.append(f"Section Applied: {doc['section_applied']}")
                context_parts.append(f"Court: {doc['court']} ({doc['year']})")
                context_parts.append(f"Summary: {doc['summary']}")
                context_parts.append(f"Relevance: {doc['score']:.3f}")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate comprehensive answer using LLM
        """
        prompt = f"""You are a legal expert AI assistant. Answer the user's question using the provided legal knowledge and case applications.

INSTRUCTIONS:
- Provide accurate, comprehensive legal information
- Reference specific sections and cases when relevant
- Explain both the law and its practical application
- Use clear, professional language
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
        """
        Complete RAG+ query pipeline
        """
        start_time = time.time()
        
        # Step 1: Hybrid retrieval
        retrieval_results = self.hybrid_retrieve(question, knowledge_k, application_k)
        
        # Step 2: Format context
        context = self.format_context(retrieval_results)
        
        # Step 3: Generate answer
        print("  🤖 Generating answer...")
        answer = self.generate_answer(question, context)
        
        # Step 4: Compile results
        result = {
            'query': question,
            'answer': answer,
            'retrieval_results': retrieval_results,
            'context': context,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.query_history.append(result)
        
        print(f"  ✓ Complete in {result['processing_time']:.2f}s")
        
        return result
    
    def display_result(self, result: Dict):
        """
        Display query result in formatted way
        """
        print("\n" + "=" * 80)
        print("RAG+ QUERY RESULT")
        print("=" * 80)
        
        print(f"\n📝 QUESTION:")
        print(f"   {result['query']}")
        
        print(f"\n🤖 ANSWER:")
        print(f"   {result['answer']}")
        
        print(f"\n📚 SOURCES:")
        knowledge_docs = result['retrieval_results']['knowledge']
        application_docs = result['retrieval_results']['applications']
        
        if knowledge_docs:
            print("   Legal Knowledge:")
            for i, doc in enumerate(knowledge_docs, 1):
                print(f"   [{i}] {doc['section_reference']} (score: {doc['score']:.3f})")
        
        if application_docs:
            print("   Case Applications:")
            for i, doc in enumerate(application_docs, 1):
                print(f"   [{i}] {doc['case_name']} - {doc['section_applied']} (score: {doc['score']:.3f})")
        
        print(f"\n⏱️  Processing Time: {result['processing_time']:.2f} seconds")
        print("=" * 80)


# Initialize RAG+ system
print("\n[2/3] Initializing RAG+ system...")
rag_system = RAGPlusSystem(
    embedder=embedding_model,
    knowledge_idx=knowledge_index,
    application_idx=application_index,
    llm=llm
)
print("✓ RAG+ system ready!")

# Test queries
test_queries = [
    "What is the punishment for murder under Indian law?",
    "What are the penalties for failure to redress investor grievances?",
    "How does Section 26A work for firm registration?",
    "What powers does SEBI have to make regulations?"
]

print("\n[3/3] Running test queries...")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n🔍 TEST QUERY {i}/{len(test_queries)}")
    result = rag_system.query(query)
    rag_system.display_result(result)
    
    if i < len(test_queries):
        print("\n" + "-" * 40)
        time.sleep(1)  # Brief pause between queries

print("\n" + "=" * 80)
print("✅ RAG+ SYSTEM TESTING COMPLETE!")
print("=" * 80)

print(f"\n📊 SUMMARY:")
print(f"   Total queries processed: {len(rag_system.query_history)}")
avg_time = sum(r['processing_time'] for r in rag_system.query_history) / len(rag_system.query_history)
print(f"   Average processing time: {avg_time:.2f}s")
print(f"   Knowledge index: {KNOWLEDGE_INDEX}")
print(f"   Application index: {APPLICATION_INDEX}")

print(f"\n🚀 READY FOR INTERACTIVE USE!")
print("   Use: result = rag_system.query('your question here')")
print("   Display: rag_system.display_result(result)")