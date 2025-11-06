import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Optional
import json
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title=" Advanced RAG+ Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        animation: slideInDown 1s ease-out;
    }
    
    @keyframes slideInDown {
        from { transform: translateY(-100px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
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
    
    .search-status {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 0.9rem;
        margin: 0.3rem;
        display: inline-block;
        animation: bounce 1s infinite;
        box-shadow: 0 5px 15px rgba(66, 153, 225, 0.4);
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .summary-box {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #f6ad55;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Handle both local development and deployment
import os

try:
    # Try Streamlit secrets first (for deployment)
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    st.info("🔑 Using Streamlit secrets for API keys")
except (KeyError, FileNotFoundError):
    # Fallback to environment variables (for local development)
    try:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if not PINECONE_API_KEY or not GEMINI_API_KEY:
            # Try reading from .env file
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('PINECONE_API_KEY'):
                            PINECONE_API_KEY = line.split('=')[1].strip().strip('"')
                        elif line.startswith('GEMINI_API_KEY'):
                            GEMINI_API_KEY = line.split('=')[1].strip().strip('"')
                st.info("🔑 Using .env file for API keys")
            except FileNotFoundError:
                st.error("❌ No API keys found. Please check your configuration.")
                st.info("📖 See DEPLOYMENT_GUIDE.md for setup instructions.")
                st.stop()
        else:
            st.info("🔑 Using environment variables for API keys")
    except Exception as e:
        st.error(f"❌ Error loading API keys: {e}")
        st.stop()

# Validate API keys
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    st.error("❌ API keys are empty. Please check your configuration.")
    st.stop()

if PINECONE_API_KEY == "your_pinecone_api_key_here" or GEMINI_API_KEY == "your_gemini_api_key_here":
    st.error("❌ Please replace placeholder API keys with actual keys.")
    st.stop()

KNOWLEDGE_INDEX = "legal-knowledge-corpus"
APPLICATION_INDEX = "legal-application-corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None

@st.cache_resource
def initialize_rag_system():
    try:
        with st.spinner("🔄 Loading AI models..."):
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        with st.spinner("🔄 Connecting to Pinecone..."):
            pc = Pinecone(api_key=PINECONE_API_KEY)
            knowledge_index = pc.Index(KNOWLEDGE_INDEX)
            application_index = pc.Index(APPLICATION_INDEX)
        
        with st.spinner("🔄 Initializing Gemini AI..."):
            genai.configure(api_key=GEMINI_API_KEY)
            llm = genai.GenerativeModel("gemini-2.0-flash")
        
        return AdvancedRAGSystem(embedding_model, knowledge_index, application_index, llm)
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.error("Please check your API keys and internet connection.")
        return None

class AdvancedRAGSystem:
    def __init__(self, embedder, knowledge_idx, application_idx, llm):
        self.embedder = embedder
        self.knowledge_index = knowledge_idx
        self.application_index = application_idx
        self.llm = llm
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.knowledge_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            
            knowledge_docs = []
            for match in results['matches']:
                knowledge_docs.append({
                    'id': match['id'],
                    'score': match['score'],
                    'section_reference': match['metadata'].get('section_reference', 'N/A'),
                    'statutory_text': match['metadata'].get('statutory_text', 'N/A'),
                    'original_question': match['metadata'].get('original_question', 'N/A'),
                    'context': match['metadata'].get('context', ''),
                    'type': 'KNOWLEDGE'
                })
            return knowledge_docs
        except Exception as e:
            st.error(f"Error retrieving knowledge: {e}")
            return []
    
    def retrieve_applications(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.application_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            
            application_docs = []
            for match in results['matches']:
                application_docs.append({
                    'id': match['id'],
                    'score': match['score'],
                    'case_name': match['metadata'].get('case_name', 'N/A'),
                    'section_applied': match['metadata'].get('section_applied', 'N/A'),
                    'year': match['metadata'].get('year', 0),
                    'court': match['metadata'].get('court', 'N/A'),
                    'summary': match['metadata'].get('case_summary', 'N/A'),
                    'judgment_url': match['metadata'].get('judgment_url', ''),
                    'type': 'APPLICATION'
                })
            return application_docs
        except Exception as e:
            st.error(f"Error retrieving applications: {e}")
            return []
    
    def format_context(self, retrieval_results: Dict, mode: str = "RAG+") -> str:
        context_parts = []
        
        if retrieval_results['knowledge']:
            context_parts.append("=== LEGAL KNOWLEDGE BASE ===")
            for i, doc in enumerate(retrieval_results['knowledge'], 1):
                context_parts.append(f"\n[STATUTE {i}] - Relevance: {doc['score']:.4f}")
                context_parts.append(f"Sections: {doc['section_reference']}")
                context_parts.append(f"Legal Text: {doc['statutory_text']}")
                context_parts.append(f"Related Question: {doc['original_question']}")
                context_parts.append("=" * 80)
        
        if mode == "RAG+" and retrieval_results.get('applications'):
            context_parts.append("\n=== CASE LAW DATABASE ===")
            for i, doc in enumerate(retrieval_results['applications'], 1):
                context_parts.append(f"\n[CASE {i}] - Relevance: {doc['score']:.4f}")
                context_parts.append(f"Case: {doc['case_name']}")
                context_parts.append(f"Section Applied: {doc['section_applied']}")
                context_parts.append(f"Court: {doc['court']} ({doc['year']})")
                context_parts.append(f"Summary: {doc['summary']}")
                context_parts.append("=" * 80)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str, word_count: int = 1000, mode: str = "RAG+") -> str:
        mode_description = "RAG+ (Knowledge + Case Law)" if mode == "RAG+" else "RAG (Knowledge Only)"
        
        prompt = f"""You are a distinguished Senior Advocate with 30+ years of experience. Provide comprehensive legal analysis using {mode_description} approach.

MANDATORY REQUIREMENTS:
1. Write EXACTLY {word_count} words (±50 words tolerance)
2. Use ALL provided legal sources extensively
3. Provide detailed legal reasoning and analysis
4. Structure as a professional legal brief
5. Include citations and references to provided sources

ANALYSIS STRUCTURE:
I. EXECUTIVE SUMMARY
II. LEGAL FRAMEWORK ANALYSIS  
III. DETAILED LEGAL REASONING
IV. {"PRECEDENTIAL ANALYSIS" if mode == "RAG+" else "STATUTORY INTERPRETATION"}
V. PRACTICAL IMPLICATIONS
VI. CONCLUSION

LEGAL RESEARCH CONTEXT:
{context}

USER'S LEGAL QUESTION: {query}

TARGET WORD COUNT: {word_count} words

PROVIDE YOUR COMPREHENSIVE LEGAL ANALYSIS:"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_summary(self, original_answer: str) -> str:
        prompt = f"""You are a legal expert. Provide a CONCISE SUMMARY of the following legal analysis in exactly 200-300 words.

REQUIREMENTS:
1. Capture the key legal points and conclusions
2. Maintain legal accuracy and terminology
3. Structure as: Key Issue → Legal Principles → Conclusion
4. Write in clear, professional language

ORIGINAL LEGAL ANALYSIS:
{original_answer}

PROVIDE A CONCISE SUMMARY (200-300 words):"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def query(self, question: str, mode: str = "RAG+", word_count: int = 1000, knowledge_k: int = 3, application_k: int = 3) -> Dict:
        start_time = time.time()
        
        knowledge_docs = self.retrieve_knowledge(question, knowledge_k)
        
        application_docs = []
        if mode == "RAG+":
            application_docs = self.retrieve_applications(question, application_k)
        
        retrieval_results = {
            'knowledge': knowledge_docs,
            'applications': application_docs,
            'query': question,
            'mode': mode,
            'timestamp': datetime.now().isoformat()
        }
        
        context = self.format_context(retrieval_results, mode)
        answer = self.generate_answer(question, context, word_count, mode)
        
        result = {
            'query': question,
            'answer': answer,
            'retrieval_results': retrieval_results,
            'context': context,
            'mode': mode,
            'word_count': word_count,
            'actual_word_count': len(answer.split()),
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

def save_to_history(result: Dict):
    if len(st.session_state.query_history) >= 20:
        st.session_state.query_history.pop(0)
    
    result['history_id'] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(st.session_state.query_history)}"
    st.session_state.query_history.append(result)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">⚖️ Advanced RAG+ Legal AI</h1>
        <p style="font-size: 1.3rem; opacity: 0.9;">🚀 Intelligent Legal Research with Customizable Analysis</p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.2rem;">📚 RAG Mode</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.2rem;">⚖️ RAG+ Mode</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.2rem;">🎯 Custom Word Count</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_rag_system()
    
    if st.session_state.rag_system is None:
        st.error("❌ Failed to initialize system. Please check the deployment guide.")
        st.info("📖 See DEPLOYMENT_GUIDE.md for setup instructions.")
        return
    
    # Success message
    st.success("✅ RAG+ Legal AI System Initialized Successfully!")
    
    # Example queries
    st.markdown("### 📋 Example Legal Questions")
    example_queries = [
        "When can High Court jurisdiction be invoked after Supreme Court dismissal?",
        "What are the comprehensive penalties for failure to redress investor grievances?",
        "Analyze the constitutional validity of SEBI's regulatory powers",
        "Detailed analysis of Article 226 writ jurisdiction limitations"
    ]
    
    cols = st.columns(2)
    for i, query in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(f"📝 {query[:60]}...", key=f"example_{i}", use_container_width=True):
                st.session_state.current_query = query
                st.rerun()
    
    # Question input
    st.markdown("### 🔍 Your Legal Question")
    
    current_query = st.session_state.get('current_query', '')
    user_query = st.text_area(
        "Enter your legal question:",
        value=current_query,
        height=120,
        placeholder="e.g., Analyze the legal implications of High Court jurisdiction after Supreme Court dismissal...",
        label_visibility="collapsed"
    )
    
    # Configuration options
    st.markdown("### ⚙️ Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        word_count = st.slider(
            "📝 Target Word Count",
            min_value=100,
            max_value=2000,
            value=1000,
            step=50,
            help="Set the desired length of the legal analysis"
        )
        st.markdown(f"**Target:** {word_count} words")
    
    with col2:
        st.markdown("🔍 **Analysis Mode**")
        mode = st.radio(
            "Select analysis mode:",
            ["RAG+ (Knowledge + Cases)", "RAG (Knowledge Only)"],
            help="RAG+ uses both legal statutes and case law, RAG uses only legal statutes"
        )
        selected_mode = "RAG+" if "RAG+" in mode else "RAG"
    
    with col3:
        knowledge_k = st.selectbox("📚 Knowledge Sources", [1, 2, 3, 4, 5], index=2)
        if selected_mode == "RAG+":
            application_k = st.selectbox("⚖️ Case Sources", [1, 2, 3, 4, 5], index=2)
        else:
            application_k = 0
            st.info("Case sources disabled in RAG mode")
    
    # Query button
    query_button = st.button(
        f"🚀 Generate {word_count}-Word {selected_mode} Analysis", 
        type="primary", 
        use_container_width=True
    )
    
    # Process query
    if query_button and user_query.strip():
        if 'current_query' in st.session_state:
            del st.session_state.current_query
        
        st.session_state.summary_result = None
        
        # Search status
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #667eea;">🔍 {selected_mode} Analysis in Progress</h3>
            <span class="search-status">📚 Analyzing Legal Knowledge...</span>
            {f'<span class="search-status">⚖️ Examining Case Law...</span>' if selected_mode == "RAG+" else ''}
            <span class="search-status">🤖 Generating {word_count}-Word Analysis...</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner(f"🔬 Performing {selected_mode} Legal Analysis..."):
            result = st.session_state.rag_system.query(
                user_query, 
                mode=selected_mode,
                word_count=word_count,
                knowledge_k=knowledge_k, 
                application_k=application_k
            )
            st.session_state.current_result = result
            save_to_history(result)
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.markdown("### 📋 Legal Analysis Results")
        
        # Analysis info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Mode:** {result['mode']}")
            st.info(f"**Target Words:** {result['word_count']}")
        with col2:
            st.success(f"**Actual Words:** {result['actual_word_count']}")
            st.success(f"**Processing Time:** {result['processing_time']:.2f}s")
        
        # Main analysis
        with st.expander(f"📖 **Complete {result['mode']} Legal Analysis** ({result['actual_word_count']} words)", expanded=True):
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); 
                        padding: 2rem; border-radius: 15px; 
                        border-left: 6px solid #48bb78; 
                        line-height: 1.8; font-size: 1.1rem;'>
                {result['answer'].replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Summarize button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📄 Summarize Output", type="secondary"):
                with st.spinner("🔄 Generating summary..."):
                    summary = st.session_state.rag_system.generate_summary(result['answer'])
                    st.session_state.summary_result = summary
        
        # Display summary
        if st.session_state.summary_result:
            st.markdown("""
            <div class="summary-box">
                <h3 style="color: #f6ad55; margin-bottom: 1rem;">📄 Brief Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); 
                        padding: 1.5rem; border-radius: 15px; 
                        border-left: 6px solid #f6ad55; 
                        line-height: 1.6; font-size: 1rem;'>
                {st.session_state.summary_result.replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("### 📊 Analysis Statistics")
        
        knowledge_matches = len(result['retrieval_results']['knowledge'])
        application_matches = len(result['retrieval_results']['applications'])
        
        col1, col2, col3= st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{knowledge_matches}</h2>
                <p>📚 Legal Statutes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{application_matches}</h2>
                <p>⚖️ Case Precedents</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{result['actual_word_count']:,}</h2>
                <p>📝 Words Generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        
        
        # Enhanced Sources Display - This is what was missing!
        st.markdown("### 📚 Retrieved Corpus Data Analysis")
        
        if result['retrieval_results']['knowledge']:
            st.markdown("#### 📖 Legal Knowledge Sources Retrieved")
            for i, doc in enumerate(result['retrieval_results']['knowledge'], 1):
                with st.expander(f"📜 [STATUTE {i}] {doc['section_reference']} (Relevance Score: {doc['score']:.4f})", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Relevance Score", f"{doc['score']:.4f}")
                        st.metric("Document ID", doc['id'][:12] + "...")
                        st.metric("Type", doc['type'])
                    
                    with col2:
                        st.markdown(f"**📋 Section Reference:**")
                        st.info(doc['section_reference'])
                        
                        st.markdown(f"**⚖️ Complete Legal Text:**")
                        st.text_area("", doc['statutory_text'], height=200, key=f"knowledge_text_{i}", disabled=True)
                        
                        st.markdown(f"**❓ Original Question:**")
                        st.text_area("", doc['original_question'], height=100, key=f"knowledge_question_{i}", disabled=True)
                        
                        if doc['context']:
                            st.markdown(f"**📝 Additional Context:**")
                            st.text_area("", doc['context'], height=100, key=f"knowledge_context_{i}", disabled=True)
        
        if result['mode'] == "RAG+" and result['retrieval_results']['applications']:
            st.markdown("#### ⚖️ Case Law & Application Sources Retrieved")
            for i, doc in enumerate(result['retrieval_results']['applications'], 1):
                with st.expander(f"🏛️ [CASE {i}] {doc['case_name']} (Relevance Score: {doc['score']:.4f})", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Relevance Score", f"{doc['score']:.4f}")
                        st.metric("Document ID", doc['id'][:12] + "...")
                        st.metric("Year", doc['year'] if doc['year'] else "N/A")
                        st.metric("Type", doc['type'])
                    
                    with col2:
                        st.markdown(f"**🏛️ Case Name:**")
                        st.info(doc['case_name'])
                        
                        st.markdown(f"**📜 Section Applied:**")
                        st.info(doc['section_applied'])
                        
                        st.markdown(f"**⚖️ Court & Year:**")
                        st.info(f"{doc['court']} ({doc['year']})")
                        
                        st.markdown(f"**📋 Complete Case Summary:**")
                        st.text_area("", doc['summary'], height=200, key=f"case_summary_{i}", disabled=True)
                        
                        if doc['judgment_url']:
                            st.markdown(f"**🔗 Official Judgment:**")
                            st.link_button("View Full Case", doc['judgment_url'])
        
        # Context sent to LLM
        with st.expander("🔍 Complete Context Sent to AI Model", expanded=False):
            st.markdown("**This is the complete context that was sent to the AI model for analysis:**")
            st.text_area("Complete Context", result['context'], height=400, disabled=True)
    
    # Sidebar with history
    with st.sidebar:
        st.markdown("### 📈 System Dashboard")
        
        if st.session_state.query_history:
            total_queries = len(st.session_state.query_history)
            avg_time = sum(r['processing_time'] for r in st.session_state.query_history) / total_queries
            total_words = sum(r['actual_word_count'] for r in st.session_state.query_history)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", total_queries)
                st.metric("Avg Time", f"{avg_time:.1f}s")
            with col2:
                st.metric("Total Words", f"{total_words:,}")
                rag_count = sum(1 for r in st.session_state.query_history if r['mode'] == 'RAG')
                st.metric("RAG/RAG+", f"{rag_count}/{total_queries-rag_count}")
            
            st.markdown("### 📋 Recent Queries")
            for i, hist_result in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                with st.expander(f"Query {total_queries - i + 1} - {hist_result['mode']}"):
                    st.markdown(f"**Q:** {hist_result['query'][:100]}...")
                    st.markdown(f"**Mode:** {hist_result['mode']}")
                    st.markdown(f"**Words:** {hist_result['actual_word_count']}")
                    st.markdown(f"**Time:** {hist_result['processing_time']:.2f}s")
                    
                    if st.button(f"🔄 Load", key=f"load_{hist_result['history_id']}"):
                        st.session_state.current_result = hist_result
                        st.rerun()
        else:
            st.info("🚀 No queries yet!")
        
        st.markdown("### ⚙️ System Status")
        st.success("🟢 Advanced RAG+ AI Online")
        st.info(f"📚 Knowledge: {KNOWLEDGE_INDEX}")
        st.info(f"⚖️ Applications: {APPLICATION_INDEX}")
        st.info("🤖 Model: gemini-2.0-flash")
        
        if st.session_state.query_history:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.query_history = []
                st.session_state.current_result = None
                st.session_state.summary_result = None
                st.rerun()

if __name__ == "__main__":
    main()