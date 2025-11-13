"""
Comprehensive RAG Metrics Evaluation for Both Domains
Evaluates: Legal Domain & Math Domain
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from dotenv import load_dotenv
import json

load_dotenv()

print("="*80)
print("MULTI-DOMAIN RAG METRICS EVALUATION")
print("="*80)

# Initialize components
print("\n[1/6] Initializing components...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("✓ Components initialized")

# Domain configurations
DOMAINS = {
    "Legal": {
        "knowledge_index": "legal-knowledge-corpus",
        "application_index": "legal-application-corpus",
        "test_queries": [
            "What are the penalties for insider trading under SEBI regulations?",
            "How does SEBI regulate stock exchanges?",
            "What are the listing requirements for securities?",
            "What happens if a broker fails to segregate client funds?",
            "Explain the role of Special Courts in securities law"
        ]
    },
    "Math": {
        "knowledge_index": "math-knowledge-corpus",
        "application_index": "math-application-corpus",
        "test_queries": [
            "What is the banker's gain formula?",
            "How do you calculate simple interest?",
            "Explain the concept of true discount",
            "What is the formula for present worth?",
            "How to solve age-related problems in mathematics?"
        ]
    }
}

def calculate_retrieval_metrics(query, knowledge_index, application_index, top_k=3):
    """Calculate retrieval metrics for a query"""
    query_embedding = embedding_model.encode(query).tolist()
    
    # Retrieve from knowledge corpus
    k_results = knowledge_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Retrieve from application corpus
    a_results = application_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Calculate metrics
    k_scores = [match['score'] for match in k_results['matches']]
    a_scores = [match['score'] for match in a_results['matches']]
    
    metrics = {
        'knowledge_avg_score': np.mean(k_scores) if k_scores else 0,
        'knowledge_max_score': max(k_scores) if k_scores else 0,
        'application_avg_score': np.mean(a_scores) if a_scores else 0,
        'application_max_score': max(a_scores) if a_scores else 0,
        'knowledge_retrieved': len(k_results['matches']),
        'application_retrieved': len(a_results['matches']),
        'combined_avg_score': np.mean(k_scores + a_scores) if (k_scores + a_scores) else 0
    }
    
    return metrics

def evaluate_domain(domain_name, config):
    """Evaluate a single domain"""
    print(f"\n[{domain_name}] Evaluating domain...")
    
    # Get indices
    knowledge_index = pc.Index(config['knowledge_index'])
    application_index = pc.Index(config['application_index'])
    
    # Get index stats
    k_stats = knowledge_index.describe_index_stats()
    a_stats = application_index.describe_index_stats()
    
    print(f"  Knowledge vectors: {k_stats['total_vector_count']}")
    print(f"  Application vectors: {a_stats['total_vector_count']}")
    
    # Evaluate queries
    all_metrics = []
    for i, query in enumerate(config['test_queries'], 1):
        print(f"  Query {i}/{len(config['test_queries'])}: {query[:50]}...")
        metrics = calculate_retrieval_metrics(query, knowledge_index, application_index)
        metrics['query'] = query
        all_metrics.append(metrics)
        time.sleep(0.5)  # Rate limiting
    
    # Calculate aggregate metrics
    df = pd.DataFrame(all_metrics)
    
    aggregate = {
        'domain': domain_name,
        'total_knowledge_vectors': k_stats['total_vector_count'],
        'total_application_vectors': a_stats['total_vector_count'],
        'avg_knowledge_relevance': df['knowledge_avg_score'].mean(),
        'avg_application_relevance': df['application_avg_score'].mean(),
        'avg_combined_relevance': df['combined_avg_score'].mean(),
        'max_knowledge_relevance': df['knowledge_max_score'].max(),
        'max_application_relevance': df['application_max_score'].max(),
        'retrieval_consistency': df['knowledge_avg_score'].std(),  # Lower is better
        'queries_evaluated': len(config['test_queries'])
    }
    
    return aggregate, df

# Evaluate both domains
print("\n[2/6] Evaluating Legal Domain...")
legal_aggregate, legal_df = evaluate_domain("Legal", DOMAINS["Legal"])

print("\n[3/6] Evaluating Math Domain...")
math_aggregate, math_df = evaluate_domain("Math", DOMAINS["Math"])

# Create summary
print("\n[4/6] Creating summary...")
summary_df = pd.DataFrame([legal_aggregate, math_aggregate])

# Save results
print("\n[5/6] Saving results...")
summary_df.to_csv('metrics_summary.csv', index=False)
legal_df.to_csv('legal_metrics_detailed.csv', index=False)
math_df.to_csv('math_metrics_detailed.csv', index=False)

print("✓ Saved metrics_summary.csv")
print("✓ Saved legal_metrics_detailed.csv")
print("✓ Saved math_metrics_detailed.csv")

# Display results
print("\n[6/6] Results Summary")
print("="*80)
print("\n📊 LEGAL DOMAIN METRICS")
print("-"*80)
print(f"Knowledge Vectors: {legal_aggregate['total_knowledge_vectors']}")
print(f"Application Vectors: {legal_aggregate['total_application_vectors']}")
print(f"Avg Knowledge Relevance: {legal_aggregate['avg_knowledge_relevance']:.4f}")
print(f"Avg Application Relevance: {legal_aggregate['avg_application_relevance']:.4f}")
print(f"Avg Combined Relevance: {legal_aggregate['avg_combined_relevance']:.4f}")
print(f"Max Knowledge Relevance: {legal_aggregate['max_knowledge_relevance']:.4f}")
print(f"Max Application Relevance: {legal_aggregate['max_application_relevance']:.4f}")
print(f"Retrieval Consistency (std): {legal_aggregate['retrieval_consistency']:.4f}")

print("\n📊 MATH DOMAIN METRICS")
print("-"*80)
print(f"Knowledge Vectors: {math_aggregate['total_knowledge_vectors']}")
print(f"Application Vectors: {math_aggregate['total_application_vectors']}")
print(f"Avg Knowledge Relevance: {math_aggregate['avg_knowledge_relevance']:.4f}")
print(f"Avg Application Relevance: {math_aggregate['avg_application_relevance']:.4f}")
print(f"Avg Combined Relevance: {math_aggregate['avg_combined_relevance']:.4f}")
print(f"Max Knowledge Relevance: {math_aggregate['max_knowledge_relevance']:.4f}")
print(f"Max Application Relevance: {math_aggregate['max_application_relevance']:.4f}")
print(f"Retrieval Consistency (std): {math_aggregate['retrieval_consistency']:.4f}")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print("\nMetrics Explanation:")
print("- Avg Relevance: Higher is better (0-1 scale, cosine similarity)")
print("- Max Relevance: Best match score for any query")
print("- Retrieval Consistency: Lower is better (less variance in scores)")
print("\nFiles saved:")
print("  1. metrics_summary.csv - Aggregate metrics for both domains")
print("  2. legal_metrics_detailed.csv - Per-query metrics for legal")
print("  3. math_metrics_detailed.csv - Per-query metrics for math")
