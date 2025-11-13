# Multi-Domain RAG+ System - Unified Dashboard
# Supports: Legal Domain & Math Domain

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from typing import List, Dict
import time
from datetime import datetime
import uuid
import copy
import os
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Domain RAG+ AI System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.query_counter = 0
    st.session_state.selected_domain = None
    st.session_state.rag_system = None
    st.session_state.query_history = []
    st.session_state.current_result = None
    st.session_state.last_query_time = 0
    st.session_state.current_summary = None
    st.session_state.show_summary = False

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: slideInDown 1s ease-out;
    }
    
    .domain-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .domain-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.8);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    @keyframes slideInDown {
        from { transform: translateY(-100px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Domain configurations
DOMAINS = {
    "Legal": {
        "knowledge_index": "legal-knowledge-corpus",
        "application_index": "legal-application-corpus",
        "icon": "⚖️",
        "description": "Securities & SEBI Law",
        "color": "#3498db",
        "scope": """
        - Securities & Exchange Board of India (SEBI) regulations
        - Securities Contracts (Regulation) Act
        - Stock exchange rules and listing requirements
        - Investor protection laws
        - Corporate securities compliance
        """,
        "sample_queries": [
            "What are the penalties for insider trading?",
            "How does SEBI regulate stock exchanges?",
            "What are the listing requirements for securities?",
            "What happens if a broker fails to segregate client funds?"
        ]
    },
    "Math": {
        "knowledge_index": "math-knowledge-corpus",
        "application_index": "math-application-corpus",
        "icon": "🔢",
        "description": "Mathematical Concepts & Problem Solving",
        "color": "#e74c3c",
        "scope": """
        - Banking and finance mathematics
        - Simple and compound interest
        - True discount and present worth
        - Banker's gain calculations
        - Age and general math problems
        """,
        "sample_queries": [
            "What is the banker's gain formula?",
            "How do you calculate simple interest?",
            "Explain the concept of true discount",
            "What is the formula for present worth?"
        ]
    }
}


@st.cache_resource
def initialize_rag_system(domain: str):
    """Initialize RAG system for selected domain"""
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        domain_config = DOMAINS[domain]
        knowledge_index = pc.Index(domain_config["knowledge_index"])
        application_index = pc.Index(domain_config["application_index"])
        
        llm = Groq(api_key=GROQ_API_KEY)
        
        return RAGPlusSystem(embedding_model, knowledge_index, application_index, llm, domain)
    except Exception as e:
        st.error(f"❌ Failed to initialize {domain} system: {str(e)}")
        return None



class RAGPlusSystem:
    """Universal RAG+ system for any domain"""
    
    def __init__(self, embedder, knowledge_idx, application_idx, llm, domain):
        self.embedder = embedder
        self.knowledge_index = knowledge_idx
        self.application_index = application_idx
        self.llm = llm
        self.domain = domain
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve from knowledge corpus"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.knowledge_index.query(
                vector=query_embedding, 
                top_k=top_k, 
                include_metadata=True
            )
            
            knowledge_docs = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'score': float(match['score']),
                    'section_reference': str(match['metadata'].get('section_reference', 'N/A')),
                    'statutory_text': str(match['metadata'].get('statutory_text', 'N/A')),
                    'original_question': str(match['metadata'].get('original_question', 'N/A')),
                    'context': str(match['metadata'].get('context', '')),
                    'type': 'KNOWLEDGE'
                }
                knowledge_docs.append(doc)
            
            return knowledge_docs
        except Exception as e:
            st.error(f"Error retrieving knowledge: {e}")
            return []
    
    def retrieve_applications(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve from application corpus"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.application_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            application_docs = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'score': float(match['score']),
                    'case_name': str(match['metadata'].get('case_name', 'N/A')),
                    'section_applied': str(match['metadata'].get('section_applied', 'N/A')),
                    'year': int(match['metadata'].get('year', 0)),
                    'court': str(match['metadata'].get('court', 'N/A')),
                    'summary': str(match['metadata'].get('case_summary', 'N/A')),
                    'judgment_url': str(match['metadata'].get('judgment_url', '')),
                    'type': 'APPLICATION'
                }
                application_docs.append(doc)
            
            return application_docs
        except Exception as e:
            st.error(f"Error retrieving applications: {e}")
            return []


    def format_context(self, retrieval_results: Dict, mode: str, max_chars: int = 3000) -> str:
        """Format context for LLM with intelligent truncation"""
        context_parts = []
        total_chars = 0
        
        def truncate_text(text: str, max_len: int) -> str:
            if len(text) <= max_len:
                return text
            truncated = text[:max_len]
            last_period = truncated.rfind('.')
            if last_period > max_len * 0.7:
                return truncated[:last_period + 1]
            return truncated + "..."
        
        if retrieval_results.get('knowledge'):
            context_parts.append("KNOWLEDGE:")
            for i, doc in enumerate(retrieval_results['knowledge'], 1):
                content = truncate_text(doc['statutory_text'], 800)
                section = f"[{i}] {doc['section_reference']}: {content}"
                
                if total_chars + len(section) > max_chars:
                    break
                
                context_parts.append(section)
                total_chars += len(section)
        
        if mode == "RAG+" and retrieval_results.get('applications'):
            remaining_chars = max_chars - total_chars
            if remaining_chars > 500:
                context_parts.append("\nAPPLICATIONS:")
                for i, doc in enumerate(retrieval_results['applications'], 1):
                    summary = truncate_text(doc['summary'], 600)
                    app_text = f"[{i}] {doc['case_name']}: {summary}"
                    
                    if total_chars + len(app_text) > max_chars:
                        break
                    
                    context_parts.append(app_text)
                    total_chars += len(app_text)
        
        return "\n\n".join(context_parts)


    def generate_answer(self, query: str, context: str, word_count: int, mode: str) -> str:
        """Generate answer with advanced domain-specific prompts"""
        
        if mode == "Vanilla LLM":
            if self.domain == "Legal":
                prompt = f"""You are an expert legal scholar specializing in securities law, SEBI regulations, and corporate compliance. Using your comprehensive knowledge of Indian securities law, provide a detailed, authoritative answer.

QUESTION: {query}

INSTRUCTIONS:
- Draw upon your deep understanding of securities regulations, case law, and legal principles
- Provide technical legal analysis with proper terminology
- Explain the legal framework, implications, and practical considerations
- Reference relevant legal concepts and regulatory frameworks from your knowledge
- Structure your response logically with clear reasoning
- Target length: approximately {word_count} words

EXPERT LEGAL ANALYSIS:"""
            
            else:
                prompt = f"""You are an expert mathematician and educator with deep knowledge of mathematical concepts, problem-solving techniques, and real-world applications. Using your comprehensive mathematical expertise, provide a thorough explanation.

QUESTION: {query}

INSTRUCTIONS:
- Apply your extensive knowledge of mathematical principles and theories
- Explain concepts clearly with step-by-step reasoning
- Include relevant formulas, theorems, and mathematical relationships
- Provide intuitive explanations alongside technical details
- Show how concepts connect to broader mathematical ideas
- Target length: approximately {word_count} words

EXPERT MATHEMATICAL EXPLANATION:"""
        
        elif mode == "RAG":
            if self.domain == "Legal":
                prompt = f"""You are a legal research assistant. Answer STRICTLY based on the provided legal sources below. Do NOT use any external knowledge or assumptions.

LEGAL SOURCES PROVIDED:
{context}

QUESTION: {query}

STRICT INSTRUCTIONS:
- Use ONLY the information from the sources above
- Quote or paraphrase directly from the provided text
- If information is insufficient, state "Based on the provided sources..."
- Do NOT add legal interpretations beyond what's explicitly stated
- Cite which source ([1], [2], etc.) supports each point
- Be precise and technical, using exact legal terminology from sources
- Target length: approximately {word_count} words

ANSWER BASED SOLELY ON PROVIDED SOURCES:"""
            
            else:
                prompt = f"""You are a mathematical reference assistant. Answer STRICTLY based on the provided mathematical sources below. Do NOT use any external knowledge.

MATHEMATICAL SOURCES PROVIDED:
{context}

QUESTION: {query}

STRICT INSTRUCTIONS:
- Use ONLY the information from the sources above
- Reference formulas, definitions, and examples exactly as provided
- If information is insufficient, state "Based on the provided sources..."
- Do NOT add mathematical concepts beyond what's explicitly given
- Cite which source ([1], [2], etc.) supports each point
- Use precise mathematical notation from the sources
- Target length: approximately {word_count} words

ANSWER BASED SOLELY ON PROVIDED SOURCES:"""
        
        else:  # RAG+
            if self.domain == "Legal":
                prompt = f"""You are a senior legal expert with access to authoritative legal sources. Synthesize the provided sources with your deep legal expertise to deliver a comprehensive, authoritative answer.

AUTHORITATIVE LEGAL SOURCES:
{context}

QUESTION: {query}

ADVANCED INSTRUCTIONS:
- PRIMARY: Ground your answer in the provided legal sources
- SECONDARY: Enhance with your expert legal knowledge and reasoning
- Analyze how the sources apply to the specific question
- Provide deeper legal interpretation and implications
- Connect concepts across knowledge and application examples
- Explain practical significance and real-world impact
- Use technical legal terminology appropriately
- Show logical reasoning from principles to conclusions
- Target length: approximately {word_count} words

STRUCTURE YOUR RESPONSE:
1. Core Legal Framework (from sources)
2. Expert Analysis & Interpretation (your reasoning)
3. Practical Applications (from examples + your insights)
4. Key Takeaways

COMPREHENSIVE LEGAL ANALYSIS:"""
            
            else:
                prompt = f"""You are a master mathematician and educator with access to mathematical references. Synthesize the provided sources with your deep mathematical expertise to deliver a comprehensive, insightful explanation.

MATHEMATICAL SOURCES:
{context}

QUESTION: {query}

ADVANCED INSTRUCTIONS:
- PRIMARY: Ground your answer in the provided mathematical sources
- SECONDARY: Enhance with your expert mathematical knowledge
- Explain the underlying mathematical principles and reasoning
- Connect concepts across knowledge and application examples
- Show step-by-step problem-solving approaches when relevant
- Provide intuitive understanding alongside technical precision
- Include relevant formulas, theorems, and relationships
- Demonstrate how concepts apply in practice
- Target length: approximately {word_count} words

STRUCTURE YOUR RESPONSE:
1. Core Concepts (from sources)
2. Mathematical Reasoning (your expert analysis)
3. Practical Applications (from examples + your insights)
4. Key Insights

COMPREHENSIVE MATHEMATICAL EXPLANATION:"""
        
        max_retries = 3
        base_delay = 10
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    st.warning(f"⏳ Rate limit hit. Waiting {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                
                max_tokens = min(word_count * 2, 2000)
                
                completion = self.llm.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    stream=False
                )
                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e).lower()
                
                if any(keyword in error_str for keyword in ['429', 'quota', 'rate limit', 'resource exhausted', 'resource_exhausted']):
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return f"""⚠️ **API Rate Limit Exceeded - Please Wait**

**IMMEDIATE SOLUTIONS:**
1. **WAIT 1 MINUTE** then try again
2. **USE THESE OPTIMAL SETTINGS:**
   - Word Count: 100-200
   - Top-K: 2
   - Wait 5 seconds between queries

**Groq Free Tier Limits:**
- 30 requests per minute
- 14,400 tokens per minute

**Best Practice:**
- Use word count 100-200
- Wait 5 seconds between queries"""
                
                else:
                    if attempt < max_retries - 1:
                        st.warning(f"⚠️ Error on attempt {attempt + 1}: {str(e)[:100]}... Retrying...")
                        time.sleep(base_delay)
                        continue
                    else:
                        return f"""❌ **Error Generating Answer**

An unexpected error occurred after {max_retries} attempts.

**Error:** {str(e)[:300]}

**Suggestions:**
1. Check your internet connection
2. Verify your API key is valid
3. Try again in a few minutes
4. Reduce word count
5. Try a different AI mode"""
        
        return "❌ Error: Maximum retries exceeded. Please wait 60 seconds and try again."


    def generate_summary(self, answer: str, query: str) -> str:
        """Generate a concise summary of the answer"""
        try:
            if self.domain == "Legal":
                summary_prompt = f"""You are a legal expert. Provide a concise 3-4 sentence summary of the following legal answer.

ORIGINAL QUESTION: {query}

FULL ANSWER:
{answer}

INSTRUCTIONS:
- Extract the most critical legal points
- Use clear, precise legal terminology
- Focus on key regulations, implications, and conclusions
- Keep it under 100 words

CONCISE SUMMARY:"""
            
            else:
                summary_prompt = f"""You are a mathematics expert. Provide a concise 3-4 sentence summary of the following mathematical explanation.

ORIGINAL QUESTION: {query}

FULL ANSWER:
{answer}

INSTRUCTIONS:
- Extract the core mathematical concepts
- Include key formulas or principles
- Focus on the main solution approach
- Keep it under 100 words

CONCISE SUMMARY:"""
            
            completion = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5,
                max_tokens=200,
                top_p=0.9,
                stream=False
            )
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            return f"⚠️ Could not generate summary: {str(e)[:100]}"

    
    def process_query(self, query: str, mode: str, word_count: int, top_k: int) -> Dict:
        """Process query and return results"""
        start_time = time.time()
        
        result = {
            'query': query,
            'mode': mode,
            'domain': self.domain,
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.session_id,
            'query_id': str(uuid.uuid4()),
            'retrieval_results': {},
            'answer': '',
            'processing_time': 0
        }
        
        try:
            if mode == "Vanilla LLM":
                result['retrieval_results'] = {'knowledge': [], 'applications': []}
                result['answer'] = self.generate_answer(query, "", word_count, mode)
            
            elif mode == "RAG":
                knowledge_docs = self.retrieve_knowledge(query, top_k)
                result['retrieval_results'] = {
                    'knowledge': copy.deepcopy(knowledge_docs),
                    'applications': []
                }
                max_context_chars = min(2000, word_count * 8)
                context = self.format_context(result['retrieval_results'], mode, max_context_chars)
                result['answer'] = self.generate_answer(query, context, word_count, mode)
            
            else:  # RAG+
                knowledge_docs = self.retrieve_knowledge(query, top_k)
                application_docs = self.retrieve_applications(query, top_k)
                result['retrieval_results'] = {
                    'knowledge': copy.deepcopy(knowledge_docs),
                    'applications': copy.deepcopy(application_docs)
                }
                max_context_chars = min(3000, word_count * 10)
                context = self.format_context(result['retrieval_results'], mode, max_context_chars)
                result['answer'] = self.generate_answer(query, context, word_count, mode)
            
            result['processing_time'] = time.time() - start_time
            
        except Exception as e:
            result['answer'] = f"Error processing query: {str(e)}"
            result['processing_time'] = time.time() - start_time
        
        return result



# Main UI
if st.session_state.selected_domain is None:
    # Dashboard / Domain Selection
    st.markdown('<div class="main-header"><h1>🚀 Multi-Domain RAG+ AI System</h1><p>Advanced Retrieval-Augmented Generation with Dual-Corpus Architecture</p></div>', unsafe_allow_html=True)
    
    st.markdown("## 📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>2</h3><p>Domains</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>3</h3><p>AI Modes</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Dual</h3><p>Corpus</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>RAG+</h3><p>Architecture</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")

    
    # Performance Metrics Section
    st.markdown("## 📈 Performance Metrics")
    
    try:
        metrics_df = pd.read_csv('metrics_summary.csv')
        
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Relevance Scores", "📋 Detailed Metrics"])
        
        with tab1:
            st.markdown("### System Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚖️ Legal Domain")
                legal_data = metrics_df[metrics_df['domain'] == 'Legal'].iloc[0]
                
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("📚 Knowledge Vectors", f"{int(legal_data['total_knowledge_vectors'])}")
                    st.metric("🎯 Avg Knowledge Relevance", f"{legal_data['avg_knowledge_relevance']:.3f}")
                with subcol2:
                    st.metric("💡 Application Vectors", f"{int(legal_data['total_application_vectors'])}")
                    st.metric("🎯 Avg Application Relevance", f"{legal_data['avg_application_relevance']:.3f}")
                
                st.metric("⭐ Combined Relevance", f"{legal_data['avg_combined_relevance']:.3f}")
            
            with col2:
                st.markdown("#### 🔢 Math Domain")
                math_data = metrics_df[metrics_df['domain'] == 'Math'].iloc[0]
                
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("📚 Knowledge Vectors", f"{int(math_data['total_knowledge_vectors'])}")
                    st.metric("🎯 Avg Knowledge Relevance", f"{math_data['avg_knowledge_relevance']:.3f}")
                with subcol2:
                    st.metric("💡 Application Vectors", f"{int(math_data['total_application_vectors'])}")
                    st.metric("🎯 Avg Application Relevance", f"{math_data['avg_application_relevance']:.3f}")
                
                st.metric("⭐ Combined Relevance", f"{math_data['avg_combined_relevance']:.3f}")
        
        with tab2:
            st.markdown("### Relevance Score Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Legal - Knowledge',
                x=['Knowledge Relevance'],
                y=[legal_data['avg_knowledge_relevance']],
                marker_color='#3498db',
                text=[f"{legal_data['avg_knowledge_relevance']:.3f}"],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Legal - Application',
                x=['Application Relevance'],
                y=[legal_data['avg_application_relevance']],
                marker_color='#2980b9',
                text=[f"{legal_data['avg_application_relevance']:.3f}"],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Math - Knowledge',
                x=['Knowledge Relevance'],
                y=[math_data['avg_knowledge_relevance']],
                marker_color='#e74c3c',
                text=[f"{math_data['avg_knowledge_relevance']:.3f}"],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Math - Application',
                x=['Application Relevance'],
                y=[math_data['avg_application_relevance']],
                marker_color='#c0392b',
                text=[f"{math_data['avg_application_relevance']:.3f}"],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Average Relevance Scores by Domain and Corpus Type",
                yaxis_title="Relevance Score (0-1)",
                barmode='group',
                height=400,
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure()
            
            categories = ['Knowledge Relevance', 'Application Relevance', 'Combined Relevance', 
                         'Max Knowledge', 'Max Application']
            
            fig2.add_trace(go.Scatterpolar(
                r=[legal_data['avg_knowledge_relevance'], 
                   legal_data['avg_application_relevance'],
                   legal_data['avg_combined_relevance'],
                   legal_data['max_knowledge_relevance'],
                   legal_data['max_application_relevance']],
                theta=categories,
                fill='toself',
                name='Legal Domain',
                line_color='#3498db'
            ))
            
            fig2.add_trace(go.Scatterpolar(
                r=[math_data['avg_knowledge_relevance'], 
                   math_data['avg_application_relevance'],
                   math_data['avg_combined_relevance'],
                   math_data['max_knowledge_relevance'],
                   math_data['max_application_relevance']],
                theta=categories,
                fill='toself',
                name='Math Domain',
                line_color='#e74c3c'
            ))
            
            fig2.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 0.7]
                    )),
                showlegend=True,
                title="Multi-Dimensional Performance Comparison",
                height=500,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.markdown("### Detailed Metrics Explanation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Metric Definitions:**
                
                - **Knowledge Vectors**: Total concept/definition entries
                - **Application Vectors**: Total practical examples/cases
                - **Avg Knowledge Relevance**: Cosine similarity for knowledge (0-1)
                - **Avg Application Relevance**: Cosine similarity for applications (0-1)
                - **Combined Relevance**: Overall average across both corpus types
                - **Max Relevance**: Best match score achieved
                - **Retrieval Consistency**: Standard deviation (lower = more consistent)
                """)
            
            with col2:
                st.markdown("""
                **🎯 Significance:**
                
                - **Higher Relevance (>0.5)**: Strong semantic matching
                - **Moderate Relevance (0.3-0.5)**: Good matching
                - **Lower Relevance (<0.3)**: Weak matching
                - **Low Consistency (<0.1)**: Stable retrieval
                - **High Consistency (>0.15)**: Variable performance
                """)
            
            st.markdown("---")
            st.markdown("### 📋 Raw Metrics Data")
            st.dataframe(metrics_df, use_container_width=True)
            
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name="rag_metrics_summary.csv",
                mime="text/csv"
            )
    
    except FileNotFoundError:
        st.warning("⚠️ Metrics data not found. Run `python evaluate_both_domains.py` to generate metrics.")
    except Exception as e:
        st.error(f"❌ Error loading metrics: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("## 🎯 Select Your Domain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="domain-card">
            <h2>{DOMAINS['Legal']['icon']} Legal Domain</h2>
            <h4>{DOMAINS['Legal']['description']}</h4>
            <p><strong>Coverage:</strong></p>
            <pre>{DOMAINS['Legal']['scope']}</pre>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch Legal Domain", key="legal_btn", use_container_width=True):
            st.session_state.selected_domain = "Legal"
            st.rerun()
    
    with col2:
        st.markdown(f"""
        <div class="domain-card">
            <h2>{DOMAINS['Math']['icon']} Math Domain</h2>
            <h4>{DOMAINS['Math']['description']}</h4>
            <p><strong>Coverage:</strong></p>
            <pre>{DOMAINS['Math']['scope']}</pre>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch Math Domain", key="math_btn", use_container_width=True):
            st.session_state.selected_domain = "Math"
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 🏗️ System Architecture")
    
    st.markdown("""
    <div class="info-box">
        <h3>RAG+ Dual-Corpus Architecture</h3>
        <p><strong>Knowledge Corpus:</strong> Core concepts, definitions, and theoretical foundations</p>
        <p><strong>Application Corpus:</strong> Real-world examples, case studies, and practical applications</p>
        <p><strong>Three AI Modes:</strong></p>
        <ul>
            <li><strong>Vanilla LLM:</strong> Direct AI response without retrieval (baseline)</li>
            <li><strong>RAG:</strong> Knowledge corpus only (concise, fact-based)</li>
            <li><strong>RAG+:</strong> Dual corpus (comprehensive, with examples)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔧 Technical Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Embeddings**\nall-MiniLM-L6-v2\n384 dimensions")
    with col2:
        st.info("**Vector DB**\nPinecone Serverless\nCosine similarity")
    with col3:
        st.info("**LLM**\nGroq llama-3.1-8b\nFast inference")


else:
    # Domain-specific interface
    domain = st.session_state.selected_domain
    domain_config = DOMAINS[domain]
    
    st.markdown(f'<div class="main-header"><h1>{domain_config["icon"]} {domain} Domain - RAG+ System</h1><p>{domain_config["description"]}</p></div>', unsafe_allow_html=True)
    
    # Initialize system
    if st.session_state.rag_system is None or st.session_state.rag_system.domain != domain:
        with st.spinner(f"🔄 Initializing {domain} RAG+ System..."):
            st.session_state.rag_system = initialize_rag_system(domain)
            st.session_state.query_history = []
            st.session_state.current_result = None
    
    if st.session_state.rag_system is None:
        st.error("❌ Failed to initialize system. Please check your configuration.")
        if st.button("← Back to Dashboard"):
            st.session_state.selected_domain = None
            st.rerun()
        st.stop()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown(f"### {domain_config['icon']} {domain} Configuration")
        
        if st.button("← Back to Dashboard", use_container_width=True):
            st.session_state.selected_domain = None
            st.session_state.rag_system = None
            st.rerun()
        
        st.markdown("---")
        
        mode = st.selectbox(
            "🤖 Select AI Model",
            ["Vanilla LLM", "RAG", "RAG+"],
            help="Choose between baseline, single-corpus, or dual-corpus retrieval"
        )
        
        word_count = st.slider("📝 Response Length (words)", 50, 500, 150, 25,
                               help="Lower values use fewer tokens and avoid rate limits")
        top_k = st.slider("🔍 Documents to Retrieve", 1, 3, 2,
                         help="Fewer documents = fewer tokens = less rate limiting")
        
        st.markdown("---")
        st.markdown(f"**Session:** `{st.session_state.session_id[:8]}...`")
        st.markdown(f"**Queries:** {st.session_state.query_counter}")
        
        if st.session_state.last_query_time > 0:
            time_since_last = time.time() - st.session_state.last_query_time
            if time_since_last < 4:
                st.warning(f"⏳ Wait {4 - time_since_last:.1f}s before next query")
        
        st.markdown("---")
        st.markdown("### ✅ Groq API")
        st.success("""**Groq Free Tier:**
- 30 requests/minute
- 14,400 tokens/minute
- Very fast responses!
- Wait 2-3 seconds between queries""")
        
        st.markdown("---")
        st.markdown("### 📚 Domain Scope")
        st.info(domain_config["scope"])
    
    # Query Input
    st.markdown("### 💬 Ask Your Question")
    
    with st.expander("💡 Sample queries for this domain"):
        for sq in domain_config["sample_queries"]:
            st.markdown(f"- {sq}")
    
    query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder=f"e.g., {domain_config['sample_queries'][0]}"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.query_history = []
        st.session_state.current_result = None
        st.session_state.query_counter = 0
        st.session_state.current_summary = None
        st.session_state.show_summary = False
        st.rerun()
    
    # Process Query with throttling
    if search_button and query.strip():
        current_time = time.time()
        time_since_last_query = current_time - st.session_state.last_query_time
        
        if mode == "RAG+":
            min_delay = 3
        elif mode == "RAG":
            min_delay = 2
        else:
            min_delay = 2
        
        if time_since_last_query < min_delay and st.session_state.last_query_time > 0:
            wait_time = min_delay - time_since_last_query
            st.warning(f"⏳ Throttling: Please wait {wait_time:.1f} more seconds...")
            st.info(f"💡 **{mode} mode: waiting {min_delay}s (Groq is fast!)**")
            time.sleep(wait_time)
        
        with st.spinner(f"🔄 Processing with {mode}..."):
            result = st.session_state.rag_system.process_query(query, mode, word_count, top_k)
            st.session_state.current_result = result
            st.session_state.query_history.append(result)
            st.session_state.query_counter += 1
            st.session_state.last_query_time = time.time()
            st.session_state.current_summary = None
            st.session_state.show_summary = False
        
        if "⚠️" in result['answer'] or "❌" in result['answer']:
            st.error("Query completed with warnings or errors. See response below.")
        else:
            st.success(f"✅ Query processed in {result['processing_time']:.2f}s")
    
    # Display Results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 Model", result['mode'])
        with col2:
            st.metric("⏱️ Time", f"{result['processing_time']:.2f}s")
        with col3:
            knowledge_count = len(result['retrieval_results'].get('knowledge', []))
            st.metric("📚 Knowledge", knowledge_count)
        with col4:
            app_count = len(result['retrieval_results'].get('applications', []))
            st.metric("💡 Applications", app_count)
        
        # Answer
        st.markdown("### 💡 AI Response")
        st.markdown(f"<div style='background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;'>{result['answer']}</div>", unsafe_allow_html=True)
        
        # Summarize Button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("📝 Summarize", key="summarize_btn", use_container_width=True):
                with st.spinner("🔄 Generating summary..."):
                    st.session_state.current_summary = st.session_state.rag_system.generate_summary(
                        result['answer'], 
                        result['query']
                    )
                    st.session_state.show_summary = True
                    st.rerun()
        
        with col2:
            if st.session_state.show_summary and st.button("❌ Hide Summary", key="hide_summary_btn", use_container_width=True):
                st.session_state.show_summary = False
                st.session_state.current_summary = None
                st.rerun()
        
        # Display Summary if available
        if st.session_state.show_summary and st.session_state.current_summary:
            st.markdown("### 📋 Summary")
            st.markdown(f"<div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #f59e0b;'>{st.session_state.current_summary}</div>", unsafe_allow_html=True)
        
        # Retrieved Data
        if result['retrieval_results'].get('knowledge') or result['retrieval_results'].get('applications'):
            st.markdown("---")
            st.markdown("## 📚 Retrieved Sources")
            
            if result['retrieval_results'].get('knowledge'):
                st.markdown("### 📖 Knowledge Sources")
                for i, doc in enumerate(result['retrieval_results']['knowledge'], 1):
                    score = doc['score']
                    if score >= 0.5:
                        relevance_badge = "🟢 HIGH"
                        score_color = "#10b981"
                    elif score >= 0.3:
                        relevance_badge = "🟡 MEDIUM"
                        score_color = "#f59e0b"
                    else:
                        relevance_badge = "🔴 LOW"
                        score_color = "#ef4444"
                    
                    with st.expander(f"📜 [KNOWLEDGE {i}] {doc['section_reference']} | {relevance_badge} (Score: {doc['score']:.4f})", expanded=(i==1)):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"<h2 style='color: {score_color};'>{doc['score']:.4f}</h2>", unsafe_allow_html=True)
                            st.markdown(f"**Level:** {relevance_badge}")
                        with col2:
                            st.markdown("**📋 Topic:**")
                            st.info(doc['section_reference'])
                            st.markdown("**📜 Content:**")
                            st.text_area("", doc['statutory_text'], height=150, key=f"k_{result['query_id']}_{i}", disabled=True)
            
            if result['retrieval_results'].get('applications'):
                st.markdown("### 💡 Application Examples")
                for i, doc in enumerate(result['retrieval_results']['applications'], 1):
                    score = doc['score']
                    if score >= 0.4:
                        relevance_badge = "🟢 HIGH"
                        score_color = "#10b981"
                    elif score >= 0.25:
                        relevance_badge = "🟡 MEDIUM"
                        score_color = "#f59e0b"
                    else:
                        relevance_badge = "🔴 LOW"
                        score_color = "#ef4444"
                    
                    with st.expander(f"📋 [APPLICATION {i}] {doc['case_name']} | {relevance_badge} (Score: {doc['score']:.4f})", expanded=(i==1)):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"<h2 style='color: {score_color};'>{doc['score']:.4f}</h2>", unsafe_allow_html=True)
                            st.markdown(f"**Level:** {relevance_badge}")
                        with col2:
                            st.markdown("**📋 Case:**")
                            st.info(doc['case_name'])
                            st.markdown("**📜 Summary:**")
                            st.text_area("", doc['summary'], height=150, key=f"a_{result['query_id']}_{i}", disabled=True)
                            if doc['judgment_url']:
                                st.markdown(f"**🔗 URL:** [{doc['judgment_url']}]({doc['judgment_url']})")
