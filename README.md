# 🚀 Multi-Domain RAG+ AI System

**Advanced Retrieval-Augmented Generation with Dual-Corpus Architecture**

A production-ready, multi-domain AI system implementing the RAG+ architecture with dual-corpus retrieval, advanced domain-specific prompts, interactive metrics visualization, and AI-powered summarization.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![Groq](https://img.shields.io/badge/Groq-llama--3.1--8b-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📊 System Overview

This system implements the **RAG+ (Retrieval-Augmented Generation Plus)** architecture, which enhances traditional RAG by utilizing **dual corpus retrieval** with domain-specific reasoning:

- **Knowledge Corpus**: Core concepts, definitions, and theoretical foundations
- **Application Corpus**: Real-world examples, case studies, and practical applications

### 🎯 Supported Domains

| Domain | Icon | Description | Knowledge | Applications |
|--------|------|-------------|-----------|--------------|
| **Legal** | ⚖️ | Securities & SEBI Law | 50 concepts | 220 cases |
| **Math** | 🔢 | Mathematical Problem Solving | 62 concepts | 800 problems |

---

## ✨ Key Features

### 🤖 Three AI Modes with Advanced Prompts

1. **Vanilla LLM** - Deep Expert Reasoning
   - Uses LLM's internal knowledge only
   - Domain-specific expert personas (Legal Scholar / Master Mathematician)
   - Comprehensive analysis with structured reasoning
   - No external retrieval

2. **RAG** - Strict Source-Only Responses
   - Retrieves from knowledge corpus only
   - Uses ONLY provided sources (no internal knowledge)
   - Precise citations and source references
   - Fact-based, concise answers

3. **RAG+** - Sophisticated Synthesis
   - Dual-corpus retrieval (Knowledge + Applications)
   - Combines external sources with expert reasoning
   - Structured 4-part responses:
     - Core Framework (from sources)
     - Expert Analysis (reasoning)
     - Practical Applications (examples + insights)
     - Key Takeaways
   - Deep cross-referencing and synthesis

### 📝 AI-Powered Summarization

- **One-Click Summaries**: Generate concise 3-4 sentence summaries
- **Domain-Specific**: Tailored prompts for legal and math contexts
- **Toggle Visibility**: Show/hide summaries as needed
- **Automatic Reset**: Clears on new queries

### 📈 Interactive Performance Metrics

**Three Visualization Tabs:**

1. **Overview Tab**
   - Corpus statistics (vectors count)
   - Average relevance scores
   - Combined performance metrics
   - Side-by-side domain comparison

2. **Relevance Scores Tab**
   - Interactive bar charts (Plotly)
   - Multi-dimensional radar charts
   - Domain and corpus type comparison
   - Real-time performance visualization

3. **Detailed Metrics Tab**
   - Comprehensive metric definitions
   - Significance explanations
   - Raw data table
   - CSV download functionality

### 🎯 Advanced Features

- **Semantic Search**: 384-dim embeddings with cosine similarity
- **Relevance Scoring**: Color-coded indicators (🟢 HIGH, 🟡 MEDIUM, 🔴 LOW)
- **Session Management**: Multi-user support with isolated sessions
- **Rate Limiting**: Smart throttling with exponential backoff
- **Source Citations**: Transparent retrieval with expandable sources
- **Query History**: Track recent queries and results
- **Real-time Metrics**: Processing time, retrieval counts, scores

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              USER INTERFACE (Streamlit)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Dashboard   │  │   Metrics    │  │   Domain     │          │
│  │  & Metrics   │  │Visualization │  │  Interface   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         RAG+ QUERY PROCESSING ENGINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Vanilla LLM  │  │     RAG      │  │    RAG+      │          │
│  │ Deep Expert  │  │ Source-Only  │  │  Synthesis   │          │
│  │  Reasoning   │  │  Responses   │  │  + Examples  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │         Summarization Engine                      │           │
│  │  Domain-specific concise summary generation       │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           EMBEDDING & RETRIEVAL LAYER                            │
│  SentenceTransformer (all-MiniLM-L6-v2) - 384 dimensions        │
│  Parallel retrieval from dual corpora                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              VECTOR DATABASE (Pinecone Serverless)               │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ Knowledge Index     │  │ Application Index   │              │
│  │ Legal: 50 vectors   │  │ Legal: 220 vectors  │              │
│  │ Math: 62 vectors    │  │ Math: 800 vectors   │              │
│  │ Cosine Similarity   │  │ Cosine Similarity   │              │
│  └─────────────────────┘  └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LLM (Groq - llama-3.1-8b-instant)                │
│  Fast inference | Generous rate limits | High quality            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- Pinecone API Key ([Get it here](https://www.pinecone.io/))
- Groq API Key ([Get it here](https://console.groq.com/))

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rpfinal
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

4. **Verify data is indexed** (should already be done)

Check indices:
```bash
python check_sample_data.py
```

Expected output:
- legal-knowledge-corpus: 50 vectors
- legal-application-corpus: 220 vectors
- math-knowledge-corpus: 62 vectors
- math-application-corpus: 800 vectors

5. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🚀 Usage Guide

### 1. Dashboard & Metrics

When you launch the app, you'll see:

- **System Overview**: Domain count, AI modes, architecture
- **Performance Metrics**: Three interactive tabs
  - Overview: Key metrics for both domains
  - Relevance Scores: Bar and radar charts
  - Detailed Metrics: Explanations and data download
- **Domain Selection**: Choose Legal or Math

### 2. Domain Selection

Click on your desired domain:

- **⚖️ Legal Domain**: Securities & SEBI Law
  - 50 knowledge concepts
  - 220 legal cases
  - Focus: Regulations, compliance, penalties

- **🔢 Math Domain**: Mathematical Problem Solving
  - 62 mathematical concepts
  - 800 problem examples
  - Focus: Formulas, solutions, applications

### 3. Configure Settings (Sidebar)

- **AI Model**: Choose mode
  - Vanilla LLM: Expert reasoning (no retrieval)
  - RAG: Source-only responses
  - RAG+: Comprehensive synthesis
  
- **Response Length**: 50-500 words
  - Lower = faster, fewer tokens
  - Higher = more detailed
  
- **Documents to Retrieve**: 1-3 per corpus
  - More docs = better context
  - Fewer docs = faster, less tokens

### 4. Ask Questions

Enter your query in the text area. Sample queries provided:

**Legal Examples:**
- "What are the penalties for insider trading?"
- "How does SEBI regulate stock exchanges?"
- "What are listing requirements for securities?"

**Math Examples:**
- "What is the banker's gain formula?"
- "How do you calculate simple interest?"
- "Explain the concept of true discount"

### 5. View Results

The system displays:

- **AI Response**: Comprehensive answer
- **Summarize Button**: Click for concise summary
- **Metrics**: Processing time, retrieval counts
- **Retrieved Sources**: 
  - Knowledge documents with relevance scores
  - Application examples with relevance scores
  - Expandable details for each source

### 6. Generate Summary

- Click **"📝 Summarize"** button
- Wait for AI to generate 3-4 sentence summary
- View concise summary below answer
- Click **"❌ Hide Summary"** to toggle off

---

## 📊 Performance Metrics

### Evaluation Results

**Legal Domain:**
- Avg Knowledge Relevance: **0.535** (53.5%) - Excellent
- Avg Application Relevance: **0.399** (39.9%) - Good
- Combined Relevance: **0.467** (46.7%)
- Retrieval Consistency: **0.066** (Very stable)

**Math Domain:**
- Avg Knowledge Relevance: **0.371** (37.1%) - Good
- Avg Application Relevance: **0.539** (53.9%) - Excellent
- Combined Relevance: **0.455** (45.5%)
- Retrieval Consistency: **0.113** (Moderate)

### Metric Interpretation

**Relevance Scores (0-1 scale):**
- 0.6-1.0: Excellent match
- 0.4-0.6: Good match
- 0.2-0.4: Fair match
- 0.0-0.2: Poor match

**Consistency (Standard Deviation):**
- <0.08: Excellent consistency
- 0.08-0.12: Good consistency
- >0.12: Variable performance

### Running Evaluation

Generate fresh metrics:
```bash
python evaluate_both_domains.py
```

This creates:
- `metrics_summary.csv` - Aggregate metrics
- `legal_metrics_detailed.csv` - Per-query legal metrics
- `math_metrics_detailed.csv` - Per-query math metrics

---

## 📁 Project Structure

```
rpfinal/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── README.md                      # This file
├── METRICS_GUIDE.md               # Detailed metrics explanation
├── FINAL_STATUS.md                # Complete feature status
│
├── evaluate_both_domains.py       # Metrics evaluation script
├── check_sample_data.py           # Verify indexed data
├── convert_all_math_json_to_csv.py # JSON to CSV converter
├── run_legal_indexing.py          # Index legal data
├── run_math_indexing.py           # Index math data
│
├── metrics_summary.csv            # Aggregate metrics
├── legal_metrics_detailed.csv     # Per-query legal metrics
├── math_metrics_detailed.csv      # Per-query math metrics
│
├── legal/                         # Legal domain data
│   ├── knowledge_corpus.csv       # 50 legal concepts
│   └── application_corpus.csv     # 220 legal cases
│
└── math/                          # Math domain data
    ├── knowledge_corpus_math.json # Source: 62 concepts
    ├── application_corpus_maths.json # Source: 800 problems
    ├── knowledge_corpus.csv       # Processed: 62 concepts
    └── application_corpus.csv     # Processed: 800 problems
```

---

## 🔧 Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Frontend** | Streamlit | 1.31.0 | Interactive web interface |
| **Embeddings** | all-MiniLM-L6-v2 | - | 384-dim sentence embeddings |
| **Vector DB** | Pinecone Serverless | 3.0.2 | Scalable vector storage |
| **LLM** | Groq (llama-3.1-8b) | 0.4.1 | Fast answer generation |
| **Visualization** | Plotly | 5.18.0 | Interactive charts |
| **Data Processing** | Pandas | 2.1.4 | Data manipulation |
| **ML** | scikit-learn | 1.4.0 | Metrics calculation |

---

## 🎯 Advanced Prompt Engineering

### Vanilla LLM Mode

**Legal Domain:**
```
You are an expert legal scholar specializing in securities law, 
SEBI regulations, and corporate compliance. Using your comprehensive 
knowledge of Indian securities law, provide a detailed, authoritative 
answer with proper legal terminology and reasoning.
```

**Math Domain:**
```
You are an expert mathematician and educator with deep knowledge of 
mathematical concepts, problem-solving techniques, and real-world 
applications. Provide thorough explanations with step-by-step reasoning.
```

### RAG Mode

**Strict Instructions:**
- Use ONLY the information from provided sources
- Quote or paraphrase directly from sources
- Cite which source ([1], [2], etc.) supports each point
- Do NOT add interpretations beyond what's stated
- If insufficient, state "Based on the provided sources..."

### RAG+ Mode

**Structured Response:**
1. **Core Framework**: Ground answer in provided sources
2. **Expert Analysis**: Enhance with reasoning and interpretation
3. **Practical Applications**: Connect examples with insights
4. **Key Takeaways**: Summarize main points

---

## 📈 Performance & Scalability

### Latency Breakdown

| Component | Time | Percentage |
|-----------|------|------------|
| Query Embedding | 10-50ms | 1-2% |
| Vector Search (2x) | 100-400ms | 10-15% |
| Context Formatting | 5-10ms | <1% |
| LLM Generation | 1-5s | 80-85% |
| UI Rendering | 50-100ms | 2-3% |
| **Total** | **1.5-6s** | **100%** |

### Rate Limits

**Groq Free Tier:**
- 30 requests per minute
- 14,400 tokens per minute
- Much faster than alternatives
- Generous limits for development

**Recommended Settings:**
- Word Count: 100-200 (optimal)
- Top-K: 2 (balanced)
- Wait: 2-3 seconds between queries

### Scalability

- **Pinecone**: 100 QPS (queries per second)
- **Groq**: 30 RPM (requests per minute)
- **Streamlit**: ~10 concurrent users

**Bottleneck**: Groq API rate limits (easily upgradeable)

---

## 🎓 Research Implementation

This system implements and extends the **RAG+** architecture:

### Core Implementations ✅

- **Dual Corpus Architecture**: Separate knowledge and application corpora
- **Application-Aware Reasoning**: Cross-referencing concepts with examples
- **Hybrid Retrieval**: Parallel retrieval from both corpora
- **Mode Comparison**: Vanilla LLM vs RAG vs RAG+
- **Comprehensive Metrics**: Cosine similarity, relevance scoring

### Novel Extensions 🚀

- **Multi-Domain Support**: Extensible to any domain
- **Advanced Prompts**: Domain-specific, mode-specific prompting
- **AI Summarization**: One-click concise summaries
- **Interactive Metrics**: Real-time visualization with Plotly
- **Production UI**: Clean, professional Streamlit dashboard
- **Session Management**: Multi-user support
- **Rate Limiting**: Smart throttling and retry logic

---

## 🔍 Use Cases

### Legal Domain

**Ideal For:**
- Securities law research
- SEBI regulation queries
- Compliance questions
- Penalty and enforcement understanding
- Stock exchange regulations

**Example Queries:**
- "What are the penalties for insider trading under SEBI?"
- "How does SEBI regulate stock exchanges?"
- "What happens if a broker fails to segregate client funds?"

### Math Domain

**Ideal For:**
- Mathematical concept learning
- Problem-solving assistance
- Formula explanations
- Step-by-step solutions
- Real-world applications

**Example Queries:**
- "What is the banker's gain formula and how to apply it?"
- "How do you calculate simple interest?"
- "Solve age-related problems in mathematics"

---

## 🛠️ Development

### Adding a New Domain

1. **Prepare Data**
   - Create knowledge corpus (concepts, definitions)
   - Create application corpus (examples, cases)
   - Format as CSV with embeddings

2. **Index to Pinecone**
   - Create new indices
   - Upload vectors with metadata

3. **Update Configuration**
   - Add domain to `DOMAINS` dict in `app.py`
   - Define icon, description, scope
   - Add sample queries

4. **Customize Prompts**
   - Add domain-specific prompts in `generate_answer()`
   - Tailor for Vanilla LLM, RAG, and RAG+ modes

5. **Test & Evaluate**
   - Run evaluation script
   - Check metrics
   - Adjust as needed

### Running Tests

```bash
# Check indexed data
python check_sample_data.py

# Evaluate metrics
python evaluate_both_domains.py

# Convert new data
python convert_all_math_json_to_csv.py
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Ideas

- Add new domains (Science, History, Finance)
- Improve prompt engineering
- Add more visualization types
- Implement caching
- Add batch processing
- Enhance error handling

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **RAG+ Architecture**: Dual-corpus retrieval research
- **Pinecone**: Vector database infrastructure
- **Groq**: Fast LLM inference
- **Sentence Transformers**: Embedding models
- **Streamlit**: Web framework
- **Plotly**: Interactive visualizations

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check `METRICS_GUIDE.md` for detailed metrics explanation
- Review `FINAL_STATUS.md` for complete feature list

---

## 🔮 Roadmap

### Planned Features

- [ ] More domains (Science, History, Finance, Medical)
- [ ] Hybrid ranking (BM25 + Dense retrieval)
- [ ] Query classification for auto-domain detection
- [ ] Multi-hop reasoning for complex queries
- [ ] User feedback loop
- [ ] Export functionality (PDF, DOCX)
- [ ] API endpoints for programmatic access
- [ ] Batch query processing
- [ ] Advanced caching
- [ ] A/B testing framework

### Performance Improvements

- [ ] Implement caching for common queries
- [ ] Optimize embedding generation
- [ ] Add query preprocessing
- [ ] Implement streaming responses
- [ ] Add response caching

---

## 📊 Quick Stats

- **Total Vectors**: 1,132 (50 + 220 + 62 + 800)
- **Domains**: 2 (Legal, Math)
- **AI Modes**: 3 (Vanilla LLM, RAG, RAG+)
- **Features**: 15+ (Summarization, Metrics, Visualization, etc.)
- **Average Response Time**: 1.5-6 seconds
- **Supported Languages**: English
- **Deployment**: Local (Streamlit)

---

<p align="center">
  <strong>Built with ❤️ using RAG+ Architecture</strong>
</p>

<p align="center">
  🚀 Multi-Domain RAG+ AI System | Powered by Pinecone + Groq + Streamlit
</p>

<p align="center">
  <a href="#-installation">Installation</a> •
  <a href="#-usage-guide">Usage</a> •
  <a href="#-performance-metrics">Metrics</a> •
  <a href="#-technical-stack">Tech Stack</a> •
  <a href="#-contributing">Contributing</a>
</p>
