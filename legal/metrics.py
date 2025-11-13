# RAG+ Comprehensive Metrics & Evaluation System
# Implements industry-standard RAG evaluation metrics (RAGAS Framework + Traditional IR Metrics)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
import json, time, os, re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("RAG+ COMPREHENSIVE METRICS & EVALUATION SYSTEM")
print("Implementing: RAGAS Framework + Traditional IR + RAG+ Specific Metrics")
print("=" * 80)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAT8ei1x-a_RJcieCOuq2BZAyEg68CDcR4")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("\n[1/3] Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✓ Model loaded: {EMBEDDING_MODEL}")

print("\n[2/3] Initializing Gemini LLM...")
genai.configure(api_key=GEMINI_API_KEY)
eval_model = genai.GenerativeModel("gemini-2.0-flash-exp")
print("✓ Gemini initialized")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 10

print("\n[3/3] System ready for evaluation")
print("=" * 80)


class RAGMetrics:
    """
    Comprehensive RAG evaluation metrics:
    1. Retrieval Metrics (IR): Precision@K, Recall@K, MRR, NDCG, Hit Rate, MAP
    2. Generation Metrics (RAGAS): Faithfulness, Answer Relevance, Context Relevance
    3. RAG+ Specific: Coverage, Balance, Diversity
    """
    
    def __init__(self, embedder, llm):
        self.embedder = embedder
        self.llm = llm
        self.cache = {}
    
    # ========== RETRIEVAL METRICS (Traditional IR) ==========
    
    def precision_at_k(self, retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """Precision@K = |Relevant ∩ Retrieved@K| / K"""
        if k == 0: return 0.0
        top_k = retrieved[:k]
        relevant_count = sum(1 for item in top_k if item.get('knowledge_id') in relevant_ids or item.get('application_id') in relevant_ids)
        return relevant_count / k
    
    def recall_at_k(self, retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|"""
        if not relevant_ids: return 0.0
        top_k = retrieved[:k]
        retrieved_relevant = sum(1 for item in top_k if item.get('knowledge_id') in relevant_ids or item.get('application_id') in relevant_ids)
        return retrieved_relevant / len(relevant_ids)
    
    def mean_reciprocal_rank(self, retrieved: List[Dict], relevant_ids: List[str]) -> float:
        """MRR = 1 / rank(first_relevant)"""
        for idx, item in enumerate(retrieved, 1):
            item_id = item.get('knowledge_id') or item.get('application_id')
            if item_id in relevant_ids:
                return 1.0 / idx
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[Dict], relevant_scores: Dict[str, float], k: int = 5) -> float:
        """NDCG@K = DCG@K / IDCG@K"""
        if not relevant_scores: return 0.0
        dcg = sum(relevant_scores.get(item.get('knowledge_id') or item.get('application_id'), 0.0) / np.log2(idx + 1) 
                  for idx, item in enumerate(retrieved[:k], 1))
        ideal_scores = sorted(relevant_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(ideal_scores))
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(self, retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """Hit Rate@K: Binary - did we retrieve at least one relevant doc?"""
        top_k = retrieved[:k]
        for item in top_k:
            if (item.get('knowledge_id') or item.get('application_id')) in relevant_ids:
                return 1.0
        return 0.0
    
    def average_precision(self, retrieved: List[Dict], relevant_ids: List[str]) -> float:
        """MAP: Mean of precision values at each relevant document"""
        if not relevant_ids: return 0.0
        precisions, relevant_count = [], 0
        for idx, item in enumerate(retrieved, 1):
            if (item.get('knowledge_id') or item.get('application_id')) in relevant_ids:
                relevant_count += 1
                precisions.append(relevant_count / idx)
        return np.mean(precisions) if precisions else 0.0
    
    # ========== SEMANTIC SIMILARITY ==========
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between embeddings"""
        cache_key = (text1[:50], text2[:50])
        if cache_key in self.cache:
            return self.cache[cache_key]
        emb1 = self.embedder.encode(text1).reshape(1, -1)
        emb2 = self.embedder.encode(text2).reshape(1, -1)
        score = float(cosine_similarity(emb1, emb2)[0][0])
        self.cache[cache_key] = score
        return score
    
    # ========== RAGAS FRAMEWORK METRICS ==========
    
    def context_relevance(self, query: str, contexts: List[str]) -> float:
        """Context Relevance: Average semantic similarity of contexts to query"""
        if not contexts: return 0.0
        scores = [self.semantic_similarity(query, ctx) for ctx in contexts]
        return float(np.mean(scores))
    
    def answer_relevance(self, query: str, answer: str) -> float:
        """Answer Relevance: How well does answer address the query?"""
        return self.semantic_similarity(query, answer)
    
    def faithfulness_score(self, answer: str, context: str) -> float:
        """Faithfulness: Is answer grounded in context? (No hallucinations)"""
        prompt = f"""Evaluate if ANSWER is faithful to CONTEXT (no hallucinations).

CONTEXT: {context[:2000]}
ANSWER: {answer[:1000]}

Rate 0.0 (unfaithful) to 1.0 (faithful). Respond with ONLY a number (e.g., 0.85)"""
        try:
            response = self.llm.generate_content(prompt)
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response.text.strip())
            return max(0.0, min(1.0, float(numbers[0]))) if numbers else 0.5
        except:
            return 0.5
    
    def answer_correctness(self, query: str, answer: str, ground_truth: str) -> float:
        """Answer Correctness: Factual accuracy vs ground truth"""
        prompt = f"""Rate correctness of ANSWER vs GROUND TRUTH.

QUERY: {query}
GROUND TRUTH: {ground_truth[:1000]}
ANSWER: {answer[:1000]}

Rate 0.0 (wrong) to 1.0 (correct). Respond with ONLY a number (e.g., 0.92)"""
        try:
            response = self.llm.generate_content(prompt)
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response.text.strip())
            return max(0.0, min(1.0, float(numbers[0]))) if numbers else 0.5
        except:
            return 0.5
    
    def context_precision(self, query: str, contexts: List[str], answer: str) -> float:
        """Context Precision: How many contexts actually contributed to answer?"""
        if not contexts: return 0.0
        useful_count = 0
        for ctx in contexts:
            prompt = f"Did CONTEXT contribute to ANSWER for QUERY?\nQUERY: {query}\nCONTEXT: {ctx[:500]}\nANSWER: {answer[:500]}\nRespond 'yes' or 'no'."
            try:
                response = self.llm.generate_content(prompt)
                if 'yes' in response.text.lower():
                    useful_count += 1
                time.sleep(0.5)
            except:
                pass
        return useful_count / len(contexts)
    
    # ========== RAG+ SPECIFIC METRICS ==========
    
    def coverage_score(self, retrieval_results: Dict) -> float:
        """Coverage: Are both knowledge and application corpora utilized?"""
        has_k = len(retrieval_results.get('knowledge', [])) > 0
        has_a = len(retrieval_results.get('applications', [])) > 0
        return 1.0 if (has_k and has_a) else (0.5 if (has_k or has_a) else 0.0)
    
    def corpus_balance_score(self, retrieval_results: Dict) -> float:
        """Corpus Balance: How balanced is retrieval between corpora?"""
        k_count = len(retrieval_results.get('knowledge', []))
        a_count = len(retrieval_results.get('applications', []))
        total = k_count + a_count
        if total == 0: return 0.0
        k_ratio = k_count / total
        return 1.0 - abs(k_ratio - 0.5) * 2  # 1.0 = perfect balance
    
    def retrieval_diversity(self, retrieved: List[Dict]) -> float:
        """Retrieval Diversity: Semantic diversity of retrieved docs"""
        if len(retrieved) < 2: return 1.0
        texts = [doc.get('statutory_text', doc.get('summary', ''))[:500] for doc in retrieved]
        texts = [t for t in texts if t]
        if len(texts) < 2: return 1.0
        embeddings = [self.embedder.encode(text) for text in texts]
        similarities = [cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
                       for i in range(len(embeddings)) for j in range(i + 1, len(embeddings))]
        return 1.0 - np.mean(similarities)  # Diversity = 1 - avg_similarity



class RAGEvaluator:
    """Complete RAG system evaluation with comprehensive metrics"""
    
    def __init__(self, metrics: RAGMetrics):
        self.metrics = metrics
        self.results = []
    
    def evaluate_single_query(self, query: str, retrieval_results: Dict, generated_answer: str,
                             ground_truth: str = None, relevant_ids: List[str] = None) -> Dict:
        """Evaluate a single RAG query with all metrics"""
        print(f"\n📊 Evaluating: '{query[:60]}...'")
        
        # Extract contexts
        knowledge_contexts = [k.get('statutory_text', '') for k in retrieval_results.get('knowledge', [])]
        application_contexts = [a.get('summary', '') for a in retrieval_results.get('applications', [])]
        all_contexts = knowledge_contexts + application_contexts
        combined_context = "\n\n".join(all_contexts)
        all_retrieved = retrieval_results.get('knowledge', []) + retrieval_results.get('applications', [])
        
        metrics_result = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            
            # Retrieval counts
            'num_knowledge': len(retrieval_results.get('knowledge', [])),
            'num_applications': len(retrieval_results.get('applications', [])),
            
            # RAGAS metrics
            'context_relevance': self.metrics.context_relevance(query, all_contexts),
            'answer_relevance': self.metrics.answer_relevance(query, generated_answer),
            'faithfulness': self.metrics.faithfulness_score(generated_answer, combined_context),
            
            # RAG+ specific
            'coverage_score': self.metrics.coverage_score(retrieval_results),
            'corpus_balance': self.metrics.corpus_balance_score(retrieval_results),
            'retrieval_diversity': self.metrics.retrieval_diversity(all_retrieved),
            
            # Metadata
            'answer_length': len(generated_answer),
        }
        
        # Optional: Ground truth metrics
        if ground_truth:
            metrics_result['answer_correctness'] = self.metrics.answer_correctness(query, generated_answer, ground_truth)
            metrics_result['answer_similarity'] = self.metrics.semantic_similarity(generated_answer, ground_truth)
        
        # Optional: Retrieval metrics with relevance labels
        if relevant_ids:
            metrics_result['precision@3'] = self.metrics.precision_at_k(all_retrieved, relevant_ids, k=3)
            metrics_result['recall@5'] = self.metrics.recall_at_k(all_retrieved, relevant_ids, k=5)
            metrics_result['mrr'] = self.metrics.mean_reciprocal_rank(all_retrieved, relevant_ids)
            metrics_result['hit_rate@5'] = self.metrics.hit_rate_at_k(all_retrieved, relevant_ids, k=5)
            metrics_result['map'] = self.metrics.average_precision(all_retrieved, relevant_ids)
        
        self.results.append(metrics_result)
        
        print(f"  ✓ Context Relevance: {metrics_result['context_relevance']:.3f}")
        print(f"  ✓ Answer Relevance: {metrics_result['answer_relevance']:.3f}")
        print(f"  ✓ Faithfulness: {metrics_result['faithfulness']:.3f}")
        print(f"  ✓ Coverage: {metrics_result['coverage_score']:.3f}")
        
        return metrics_result
    
    def evaluate_batch(self, test_cases: List[Dict]):
        """Evaluate multiple test cases"""
        print("\n" + "=" * 80)
        print(f"BATCH EVALUATION: {len(test_cases)} queries")
        print("=" * 80)
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n[{idx}/{len(test_cases)}]")
            self.evaluate_single_query(**test_case)
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        if not self.results:
            print("No evaluation results available")
            return None
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)
        
        print("\n📊 CORE METRICS (RAGAS Framework):")
        print("-" * 80)
        core_metrics = ['context_relevance', 'answer_relevance', 'faithfulness']
        for col in core_metrics:
            if col in df.columns:
                print(f"{col:25s}: {df[col].mean():.3f} (±{df[col].std():.3f})")
        
        print("\n📊 RAG+ SPECIFIC METRICS:")
        print("-" * 80)
        rag_metrics = ['coverage_score', 'corpus_balance', 'retrieval_diversity']
        for col in rag_metrics:
            if col in df.columns:
                print(f"{col:25s}: {df[col].mean():.3f} (±{df[col].std():.3f})")
        
        if 'precision@3' in df.columns:
            print("\n📊 RETRIEVAL METRICS (IR):")
            print("-" * 80)
            ir_metrics = ['precision@3', 'recall@5', 'mrr', 'hit_rate@5', 'map']
            for col in ir_metrics:
                if col in df.columns:
                    print(f"{col:25s}: {df[col].mean():.3f} (±{df[col].std():.3f})")
        
        if 'answer_correctness' in df.columns:
            print("\n📊 GROUND TRUTH METRICS:")
            print("-" * 80)
            gt_metrics = ['answer_correctness', 'answer_similarity']
            for col in gt_metrics:
                if col in df.columns:
                    print(f"{col:25s}: {df[col].mean():.3f} (±{df[col].std():.3f})")
        
        print("\n📊 RETRIEVAL STATISTICS:")
        print("-" * 80)
        print(f"{'avg_knowledge_docs':25s}: {df['num_knowledge'].mean():.1f}")
        print(f"{'avg_application_docs':25s}: {df['num_applications'].mean():.1f}")
        print(f"{'avg_answer_length':25s}: {df['answer_length'].mean():.0f} chars")
        
        return df
    
    def visualize_metrics(self, save_path: str = "rag_metrics_comprehensive.png"):
        """Create comprehensive visualization dashboard"""
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('RAG+ Comprehensive Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. RAGAS Metrics Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ragas_cols = ['context_relevance', 'answer_relevance', 'faithfulness']
        ragas_data = df[ragas_cols].melt(var_name='Metric', value_name='Score')
        sns.boxplot(data=ragas_data, x='Metric', y='Score', ax=ax1, palette='Set2')
        ax1.set_title('RAGAS Metrics Distribution', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.3, label='Good Threshold')
        ax1.legend()
        
        # 2. RAG+ Specific Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        rag_cols = ['coverage_score', 'corpus_balance', 'retrieval_diversity']
        rag_data = df[rag_cols].melt(var_name='Metric', value_name='Score')
        sns.boxplot(data=rag_data, x='Metric', y='Score', ax=ax2, palette='Set3')
        ax2.set_title('RAG+ Specific Metrics', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Retrieval Counts
        ax3 = fig.add_subplot(gs[0, 2])
        retrieval_data = df[['num_knowledge', 'num_applications']].mean()
        bars = ax3.bar(['Knowledge\nCorpus', 'Application\nCorpus'], retrieval_data.values, 
                       color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
        ax3.set_title('Average Retrieval Counts', fontweight='bold')
        ax3.set_ylabel('Documents Retrieved')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Coverage Score Distribution
        ax4 = fig.add_subplot(gs[0, 3])
        coverage_counts = df['coverage_score'].value_counts().sort_index()
        colors_map = {0.0: '#e74c3c', 0.5: '#f39c12', 1.0: '#2ecc71'}
        colors = [colors_map.get(x, '#95a5a6') for x in coverage_counts.index]
        ax4.bar(coverage_counts.index, coverage_counts.values, color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_title('Coverage Score Distribution', fontweight='bold')
        ax4.set_xlabel('Coverage Score')
        ax4.set_ylabel('Frequency')
        ax4.set_xticks([0.0, 0.5, 1.0])
        ax4.set_xticklabels(['None\n(0.0)', 'Single\n(0.5)', 'Dual\n(1.0)'])
        
        # 5. Metric Correlations
        ax5 = fig.add_subplot(gs[1, :2])
        corr_cols = ['context_relevance', 'answer_relevance', 'faithfulness', 'coverage_score', 'corpus_balance']
        corr_cols = [c for c in corr_cols if c in df.columns]
        if len(corr_cols) > 1:
            corr_matrix = df[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5, 
                       fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            ax5.set_title('Metric Correlations', fontweight='bold')
        
        # 6. Metrics Over Queries (Time Series)
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.plot(df.index, df['context_relevance'], marker='o', label='Context Relevance', linewidth=2)
        ax6.plot(df.index, df['answer_relevance'], marker='s', label='Answer Relevance', linewidth=2)
        ax6.plot(df.index, df['faithfulness'], marker='^', label='Faithfulness', linewidth=2)
        ax6.set_title('Metrics Across Queries', fontweight='bold')
        ax6.set_xlabel('Query Index')
        ax6.set_ylabel('Score')
        ax6.legend(loc='best')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0.7, color='r', linestyle='--', alpha=0.3)
        
        # 7. Retrieval Metrics (if available)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'precision@3' in df.columns:
            ir_cols = ['precision@3', 'recall@5', 'mrr', 'hit_rate@5']
            ir_cols = [c for c in ir_cols if c in df.columns]
            ir_data = df[ir_cols].mean()
            bars = ax7.barh(range(len(ir_data)), ir_data.values, color='skyblue', edgecolor='black', linewidth=1.5)
            ax7.set_yticks(range(len(ir_data)))
            ax7.set_yticklabels(ir_data.index)
            ax7.set_xlabel('Score')
            ax7.set_title('Retrieval Metrics (IR)', fontweight='bold')
            ax7.set_xlim(0, 1)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax7.text(width, bar.get_y() + bar.get_height()/2., f'{width:.3f}',
                        ha='left', va='center', fontweight='bold')
        else:
            ax7.text(0.5, 0.5, 'No Retrieval Metrics\n(Requires relevant_ids)', 
                    ha='center', va='center', fontsize=12, transform=ax7.transAxes)
            ax7.axis('off')
        
        # 8. Answer Length Distribution
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(df['answer_length'], bins=15, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax8.axvline(df['answer_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["answer_length"].mean():.0f}')
        ax8.set_title('Answer Length Distribution', fontweight='bold')
        ax8.set_xlabel('Characters')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        
        # 9. Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis('off')
        summary_text = f"""
╔══════════════════════════════════════════════════════════╗
║           EVALUATION SUMMARY STATISTICS                  ║
╠══════════════════════════════════════════════════════════╣
║  Total Queries Evaluated: {len(df):>3d}                           ║
║                                                          ║
║  RAGAS METRICS (0-1 scale):                              ║
║    • Context Relevance:    {df['context_relevance'].mean():>5.3f} (±{df['context_relevance'].std():.3f})      ║
║    • Answer Relevance:     {df['answer_relevance'].mean():>5.3f} (±{df['answer_relevance'].std():.3f})      ║
║    • Faithfulness:         {df['faithfulness'].mean():>5.3f} (±{df['faithfulness'].std():.3f})      ║
║                                                          ║
║  RAG+ METRICS:                                           ║
║    • Coverage Score:       {df['coverage_score'].mean():>5.3f} (±{df['coverage_score'].std():.3f})      ║
║    • Corpus Balance:       {df['corpus_balance'].mean():>5.3f} (±{df['corpus_balance'].std():.3f})      ║
║    • Retrieval Diversity:  {df['retrieval_diversity'].mean():>5.3f} (±{df['retrieval_diversity'].std():.3f})      ║
║                                                          ║
║  RETRIEVAL STATS:                                        ║
║    • Avg Knowledge Docs:   {df['num_knowledge'].mean():>5.1f}                    ║
║    • Avg Application Docs: {df['num_applications'].mean():>5.1f}                    ║
║    • Avg Answer Length:    {df['answer_length'].mean():>5.0f} chars               ║
╚══════════════════════════════════════════════════════════╝
        """
        ax9.text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
                verticalalignment='top', transform=ax9.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Visualization saved: {save_path}")
        plt.show()
        return fig


# Initialize evaluation system
print("\n✓ Initializing evaluation system...")
metrics = RAGMetrics(embedding_model, eval_model)
evaluator = RAGEvaluator(metrics)
print("✓ Evaluation system ready!")



# ========== DEMO: Example Test Cases ==========

print("\n" + "=" * 80)
print("RUNNING DEMO EVALUATION")
print("=" * 80)

# Example test cases demonstrating different scenarios
demo_test_cases = [
    {
        'query': 'What are the penalties for insider trading under SEBI regulations?',
        'retrieval_results': {
            'knowledge': [
                {
                    'knowledge_id': 'k1',
                    'statutory_text': 'Under SEBI Act Section 15G, insider trading is prohibited. Penalties include imprisonment up to 10 years or fine up to Rs. 25 crores, or both.',
                    'score': 0.89
                },
                {
                    'knowledge_id': 'k2',
                    'statutory_text': 'SEBI (Prohibition of Insider Trading) Regulations, 2015 define insider trading and prescribe penalties for violations.',
                    'score': 0.82
                }
            ],
            'applications': [
                {
                    'application_id': 'a1',
                    'case_name': 'SEBI vs Rakesh Agrawal',
                    'summary': 'The appellant was found guilty of insider trading and was penalized Rs. 15 crores and banned from securities market for 5 years.',
                    'score': 0.85
                },
                {
                    'application_id': 'a2',
                    'case_name': 'SEBI vs XYZ Securities Ltd',
                    'summary': 'Company directors engaged in insider trading before merger announcement. SEBI imposed penalty of Rs. 10 crores.',
                    'score': 0.78
                }
            ]
        },
        'generated_answer': 'Under SEBI Act Section 15G and SEBI (Prohibition of Insider Trading) Regulations 2015, insider trading is strictly prohibited. Penalties include imprisonment up to 10 years or fine up to Rs. 25 crores, or both. In cases like SEBI vs Rakesh Agrawal, violators were penalized Rs. 15 crores and banned from securities market for 5 years. The severity of punishment depends on the extent of violation and market impact.',
        'ground_truth': 'Insider trading under SEBI regulations is punishable with imprisonment up to 10 years or fine up to Rs. 25 crores under Section 15G of SEBI Act.',
        'relevant_ids': ['k1', 'k2', 'a1', 'a2']
    },
    {
        'query': 'What is the process for delisting securities from stock exchange?',
        'retrieval_results': {
            'knowledge': [
                {
                    'knowledge_id': 'k3',
                    'statutory_text': 'SEBI (Delisting of Equity Shares) Regulations, 2021 govern the delisting process. Companies must obtain shareholder approval and make exit offer.',
                    'score': 0.91
                },
                {
                    'knowledge_id': 'k4',
                    'statutory_text': 'Delisting can be voluntary or compulsory. Voluntary delisting requires special resolution and reverse book building process.',
                    'score': 0.86
                }
            ],
            'applications': [
                {
                    'application_id': 'a3',
                    'case_name': 'Vedanta Ltd Delisting Case',
                    'summary': 'Vedanta attempted voluntary delisting but failed to meet the threshold. The process involved reverse book building and shareholder voting.',
                    'score': 0.88
                }
            ]
        },
        'generated_answer': 'The delisting process is governed by SEBI (Delisting of Equity Shares) Regulations, 2021. For voluntary delisting, companies must obtain shareholder approval through special resolution and conduct reverse book building process to determine exit price. As seen in Vedanta Ltd case, the delisting succeeds only if the threshold is met. The process ensures fair treatment of minority shareholders.',
        'ground_truth': 'Delisting requires shareholder approval, reverse book building, and compliance with SEBI Delisting Regulations 2021.',
        'relevant_ids': ['k3', 'k4', 'a3']
    },
    {
        'query': 'How does SEBI regulate mutual funds?',
        'retrieval_results': {
            'knowledge': [
                {
                    'knowledge_id': 'k5',
                    'statutory_text': 'SEBI (Mutual Funds) Regulations, 1996 govern mutual fund operations. Asset Management Companies must be registered with SEBI.',
                    'score': 0.87
                }
            ],
            'applications': [
                {
                    'application_id': 'a4',
                    'case_name': 'SEBI vs Franklin Templeton',
                    'summary': 'SEBI investigated Franklin Templeton for winding up debt schemes. The case highlighted regulatory oversight of mutual fund operations.',
                    'score': 0.83
                },
                {
                    'application_id': 'a5',
                    'case_name': 'SEBI vs DHFL Pramerica MF',
                    'summary': 'SEBI imposed penalty for violation of investment norms and risk management failures in mutual fund schemes.',
                    'score': 0.79
                }
            ]
        },
        'generated_answer': 'SEBI regulates mutual funds through SEBI (Mutual Funds) Regulations, 1996. All Asset Management Companies must be registered with SEBI. SEBI monitors compliance with investment norms, risk management, and investor protection. Cases like Franklin Templeton and DHFL Pramerica demonstrate SEBI\'s active enforcement of regulations to protect investor interests.',
        'ground_truth': 'SEBI regulates mutual funds through SEBI (Mutual Funds) Regulations 1996, requiring AMC registration and compliance monitoring.',
        'relevant_ids': ['k5', 'a4', 'a5']
    }
]

# Run evaluation
evaluator.evaluate_batch(demo_test_cases)

# Generate report
print("\n" + "=" * 80)
print("GENERATING COMPREHENSIVE REPORT")
print("=" * 80)
report_df = evaluator.generate_report()

# Visualize
print("\n" + "=" * 80)
print("CREATING VISUALIZATION DASHBOARD")
print("=" * 80)
evaluator.visualize_metrics()

# Save results to CSV
if report_df is not None:
    report_df.to_csv('rag_evaluation_results.csv', index=False)
    print(f"\n✅ Results saved to: rag_evaluation_results.csv")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE!")
print("=" * 80)
print("\n📚 METRICS IMPLEMENTED:")
print("  • RAGAS Framework: Context Relevance, Answer Relevance, Faithfulness")
print("  • Traditional IR: Precision@K, Recall@K, MRR, NDCG, Hit Rate, MAP")
print("  • RAG+ Specific: Coverage Score, Corpus Balance, Retrieval Diversity")
print("\n💡 To evaluate your own queries:")
print("  1. Format your results as test_cases list (see demo_test_cases above)")
print("  2. Run: evaluator.evaluate_batch(your_test_cases)")
print("  3. Generate report: evaluator.generate_report()")
print("  4. Visualize: evaluator.visualize_metrics()")
print("=" * 80)
