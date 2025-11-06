# Install required packages
# !pip install -q matplotlib seaborn pandas numpy scikit-learn ragas langchain

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
import json
from datetime import datetime

print("=" * 80)
print("RAG+ METRICS & EVALUATION SYSTEM")
print("=" * 80)

# Configuration
GEMINI_API_KEY = "AIzaSyAT8ei1x-a_RJcieCOuq2BZAyEg68CDcR4"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
genai.configure(api_key=GEMINI_API_KEY)
eval_model = genai.GenerativeModel("gemini-2.0-flash")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class RAGMetrics:
    """
    Comprehensive RAG evaluation metrics
    """
    
    def __init__(self, embedder, llm):
        self.embedder = embedder
        self.llm = llm
    
    def retrieval_precision_at_k(self, retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """
        Precision@K: Proportion of relevant documents in top-k results
        """
        top_k = retrieved[:k]
        relevant_count = sum(1 for item in top_k if item.get('knowledge_id') in relevant_ids or item.get('application_id') in relevant_ids)
        return relevant_count / k if k > 0 else 0.0
    
    def retrieval_recall_at_k(self, retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """
        Recall@K: Proportion of relevant documents retrieved in top-k
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved[:k]
        retrieved_relevant = sum(1 for item in top_k if item.get('knowledge_id') in relevant_ids or item.get('application_id') in relevant_ids)
        return retrieved_relevant / len(relevant_ids)
    
    def mean_reciprocal_rank(self, retrieved: List[Dict], relevant_ids: List[str]) -> float:
        """
        MRR: Average of reciprocal ranks of first relevant document
        """
        for idx, item in enumerate(retrieved, 1):
            item_id = item.get('knowledge_id') or item.get('application_id')
            if item_id in relevant_ids:
                return 1.0 / idx
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[Dict], relevant_scores: Dict[str, float], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        """
        dcg = 0.0
        for idx, item in enumerate(retrieved[:k], 1):
            item_id = item.get('knowledge_id') or item.get('application_id')
            relevance = relevant_scores.get(item_id, 0.0)
            dcg += relevance / np.log2(idx + 1)
        
        # Ideal DCG
        ideal_scores = sorted(relevant_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def semantic_similarity(self, query: str, retrieved_text: str) -> float:
        """
        Cosine similarity between query and retrieved text
        """
        query_emb = self.embedder.encode(query).reshape(1, -1)
        text_emb = self.embedder.encode(retrieved_text).reshape(1, -1)
        return cosine_similarity(query_emb, text_emb)[0][0]
    
    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Semantic similarity between query and generated answer
        """
        return self.semantic_similarity(query, answer)
    
    def faithfulness_score(self, answer: str, context: str) -> float:
        """
        LLM-based faithfulness: Does answer align with context?
        """
        prompt = f"""Evaluate if the ANSWER is faithful to the CONTEXT (i.e., no hallucinations or unsupported claims).

CONTEXT:
{context}

ANSWER:
{answer}

Rate faithfulness from 0.0 (completely unfaithful) to 1.0 (completely faithful).
Consider:
- Are all claims in the answer supported by the context?
- Are there any contradictions?
- Is there any fabricated information?

Respond with ONLY a number between 0.0 and 1.0."""

        try:
            response = self.llm.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default if evaluation fails
    
    def answer_correctness(self, query: str, answer: str, ground_truth: str) -> float:
        """
        LLM-based correctness evaluation
        """
        prompt = f"""Evaluate how correct the ANSWER is compared to the GROUND TRUTH for the given QUERY.

QUERY:
{query}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{answer}

Rate correctness from 0.0 (completely wrong) to 1.0 (completely correct).
Consider factual accuracy, completeness, and legal precision.

Respond with ONLY a number between 0.0 and 1.0."""

        try:
            response = self.llm.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def context_relevance(self, query: str, contexts: List[str]) -> float:
        """
        Average relevance of retrieved contexts to query
        """
        if not contexts:
            return 0.0
        
        scores = [self.semantic_similarity(query, ctx) for ctx in contexts]
        return np.mean(scores)
    
    def coverage_score(self, retrieval_results: Dict) -> float:
        """
        Dual corpus coverage: Are both knowledge and applications retrieved?
        """
        has_knowledge = len(retrieval_results.get('knowledge', [])) > 0
        has_applications = len(retrieval_results.get('applications', [])) > 0
        
        if has_knowledge and has_applications:
            return 1.0
        elif has_knowledge or has_applications:
            return 0.5
        else:
            return 0.0


class RAGEvaluator:
    """
    Complete RAG system evaluation
    """
    
    def __init__(self, metrics: RAGMetrics):
        self.metrics = metrics
        self.results = []
    
    def evaluate_single_query(
        self,
        query: str,
        retrieval_results: Dict,
        generated_answer: str,
        ground_truth: str = None,
        relevant_ids: List[str] = None
    ) -> Dict:
        """
        Evaluate a single RAG query
        """
        print(f"\n📊 Evaluating: '{query}'")
        
        # Extract contexts
        knowledge_contexts = [k['statutory_text'] for k in retrieval_results.get('knowledge', [])]
        application_contexts = [a['summary'] for a in retrieval_results.get('applications', [])]
        all_contexts = knowledge_contexts + application_contexts
        combined_context = "\n\n".join(all_contexts)
        
        # Retrieval metrics
        all_retrieved = retrieval_results.get('knowledge', []) + retrieval_results.get('applications', [])
        
        metrics_result = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            
            # Retrieval metrics
            'num_knowledge_retrieved': len(retrieval_results.get('knowledge', [])),
            'num_applications_retrieved': len(retrieval_results.get('applications', [])),
            'coverage_score': self.metrics.coverage_score(retrieval_results),
            'context_relevance': self.metrics.context_relevance(query, all_contexts),
            
            # Generation metrics
            'answer_relevance': self.metrics.answer_relevance(query, generated_answer),
            'faithfulness': self.metrics.faithfulness_score(generated_answer, combined_context),
            'answer_length': len(generated_answer),
        }
        
        # Optional metrics if ground truth provided
        if ground_truth:
            metrics_result['answer_correctness'] = self.metrics.answer_correctness(
                query, generated_answer, ground_truth
            )
        
        # Optional retrieval precision/recall if relevant IDs provided
        if relevant_ids:
            metrics_result['precision@3'] = self.metrics.retrieval_precision_at_k(all_retrieved, relevant_ids, k=3)
            metrics_result['recall@5'] = self.metrics.retrieval_recall_at_k(all_retrieved, relevant_ids, k=5)
            metrics_result['mrr'] = self.metrics.mean_reciprocal_rank(all_retrieved, relevant_ids)
        
        self.results.append(metrics_result)
        
        print(f"  ✓ Coverage: {metrics_result['coverage_score']:.2f}")
        print(f"  ✓ Context Relevance: {metrics_result['context_relevance']:.2f}")
        print(f"  ✓ Answer Relevance: {metrics_result['answer_relevance']:.2f}")
        print(f"  ✓ Faithfulness: {metrics_result['faithfulness']:.2f}")
        
        return metrics_result
    
    def evaluate_batch(self, test_cases: List[Dict]):
        """
        Evaluate multiple test cases
        """
        print("\n" + "=" * 80)
        print("BATCH EVALUATION")
        print("=" * 80)
        
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n[{idx}/{len(test_cases)}]")
            self.evaluate_single_query(**test_case)
    
    def generate_report(self):
        """
        Generate comprehensive evaluation report
        """
        if not self.results:
            print("No evaluation results available")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)
        
        print("\n📈 AGGREGATE METRICS:")
        print("-" * 80)
        
        metric_cols = ['coverage_score', 'context_relevance', 'answer_relevance', 'faithfulness']
        if 'answer_correctness' in df.columns:
            metric_cols.append('answer_correctness')
        if 'precision@3' in df.columns:
            metric_cols.extend(['precision@3', 'recall@5', 'mrr'])
        
        for col in metric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"{col:25s}: {mean_val:.3f} (±{std_val:.3f})")
        
        print(f"\n{'avg_knowledge_retrieved':25s}: {df['num_knowledge_retrieved'].mean():.1f}")
        print(f"{'avg_applications_retrieved':25s}: {df['num_applications_retrieved'].mean():.1f}")
        print(f"{'avg_answer_length':25s}: {df['answer_length'].mean():.0f} chars")
        
        return df
    
    def visualize_metrics(self, save_path: str = "rag_metrics.png"):
        """
        Create comprehensive visualization
        """
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RAG+ System Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # 1. Metric Distribution
        metric_cols = ['coverage_score', 'context_relevance', 'answer_relevance', 'faithfulness']
        metric_data = df[metric_cols].melt(var_name='Metric', value_name='Score')
        
        sns.boxplot(data=metric_data, x='Metric', y='Score', ax=axes[0, 0])
        axes[0, 0].set_title('Score Distributions')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Retrieval Counts
        retrieval_data = df[['num_knowledge_retrieved', 'num_applications_retrieved']].mean()
        axes[0, 1].bar(
            ['Knowledge', 'Applications'],
            retrieval_data.values,
            color=['#3498db', '#e74c3c']
        )
        axes[0, 1].set_title('Avg Retrieval Counts')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Coverage Score
        coverage_counts = df['coverage_score'].value_counts().sort_index()
        axes[0, 2].bar(
            coverage_counts.index,
            coverage_counts.values,
            color='#2ecc71'
        )
        axes[0, 2].set_title('Coverage Score Distribution')
        axes[0, 2].set_xlabel('Coverage Score')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Correlation Heatmap
        corr_cols = ['context_relevance', 'answer_relevance', 'faithfulness']
        if 'answer_correctness' in df.columns:
            corr_cols.append('answer_correctness')
        
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Metric Correlations')
        
        # 5. Time Series (if multiple queries)
        axes[1, 1].plot(df.index, df['answer_relevance'], marker='o', label='Answer Relevance')
        axes[1, 1].plot(df.index, df['faithfulness'], marker='s', label='Faithfulness')
        axes[1, 1].set_title('Metrics Over Queries')
        axes[1, 1].set_xlabel('Query Index')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        # 6. Summary Statistics
        axes[1, 2].axis('off')
        summary_text = f"""
        SUMMARY STATISTICS
        
        Total Queries: {len(df)}
        
        Avg Coverage: {df['coverage_score'].mean():.2f}
        Avg Context Rel: {df['context_relevance'].mean():.2f}
        Avg Answer Rel: {df['answer_relevance'].mean():.2f}
        Avg Faithfulness: {df['faithfulness'].mean():.2f}
        
        Avg Knowledge: {df['num_knowledge_retrieved'].mean():.1f}
        Avg Applications: {df['num_applications_retrieved'].mean():.1f}
        
        Avg Answer Length: {df['answer_length'].mean():.0f} chars
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {save_path}")
        plt.show()


# Initialize evaluation system
print("\n✓ Initializing evaluation system...")
metrics = RAGMetrics(embedding_model, eval_model)
evaluator = RAGEvaluator(metrics)


# Example test cases
example_test_cases = [
    {
        'query': 'What is the punishment for murder under IPC?',
        'retrieval_results': {
            'knowledge': [
                {
                    'knowledge_id': 'k1',
                    'section': 'IPC 302',
                    'statutory_text': 'Murder is punishable with death or life imprisonment under Section 302 IPC.',
                    'score': 0.92
                }
            ],
            'applications': [
                {
                    'application_id': 'a1',
                    'case_name': 'Ram vs State',
                    'section': 'IPC 302',
                    'summary': 'Accused convicted for murder under IPC 302, sentenced to life imprisonment.',
                    'score': 0.87
                }
            ]
        },
        'generated_answer': 'Under Section 302 of the Indian Penal Code, murder is punishable with death penalty or life imprisonment, and the offender is also liable to pay a fine. This has been consistently applied in cases like Ram vs State.',
        'ground_truth': 'Section 302 IPC prescribes punishment for murder as death or life imprisonment with fine.',
        'relevant_ids': ['k1', 'a1']
    }
]

# Run evaluation
print("\n" + "=" * 80)
print("RUNNING EXAMPLE EVALUATION")
print("=" * 80)

evaluator.evaluate_batch(example_test_cases)

# Generate report
report_df = evaluator.generate_report()

# Visualize
evaluator.visualize_metrics()

print("\n✓ Evaluation complete!")
print("\nTo evaluate your RAG system:")
print("1. Collect query results from your RAG+ system")
print("2. Format them as test_cases list")
print("3. Run: evaluator.evaluate_batch(test_cases)")
print("4. Generate report: evaluator.generate_report()")
print("5. Visualize: evaluator.visualize_metrics()")