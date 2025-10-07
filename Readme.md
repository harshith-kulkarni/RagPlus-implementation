# RAG+ Implementation for Indian Legal Domain

## Overview
This project implements the RAG+ (Retrieval-Augmented Generation Plus) research paper for the Indian legal domain, focusing on Indian law and the judiciary system. The implementation enhances traditional RAG by creating both a knowledge corpus and an application corpus to improve legal reasoning and context-aware retrieval.

## Project Status

### ✅ Completed
- Knowledge corpus cleaning and preparation
- Application corpus generation pipeline implementation
- Multi-source legal document retrieval system
- LLM-based summarization and reasoning components

### 🚧 In Progress
- Vector database conversion for both corpora
- RAG application development
- Retrieval accuracy evaluation
- RAG+ model refinement

## Architecture

### 1. Knowledge Corpus
- **Source**: `nyayanumana_knowledge.csv`
- **Purpose**: Base legal principles and foundational knowledge
- **Status**: Cleaned and ready

### 2. Application Corpus
- **Generation Method**: Automated pipeline using LLM and web scraping
- **Components**:
  - Search query generation from base legal principles
  - Multi-source retrieval from trusted legal domains
  - Content summarization and legal reasoning generation
- **Output**: `application_corpus.csv` with columns:
  - Base Text
  - Sources
  - Retrieved Summary
  - Final Reasoning

## Prerequisites

### Required API Keys
- **Groq API**: For LLM operations (GPT-OSS-20B model)
- **SerpAPI**: For Google search functionality

### Python Dependencies
```bash
pip install SerpApi
pip install groq
pip install google-search-results
pip install pandas
pip install beautifulsoup4
pip install requests
```

## Usage

### Step 1: Prepare Knowledge Corpus
1. Upload `nyayanumana_knowledge.csv` to your working directory
2. Run the data selection cell to extract desired number of datapoints:
   ```python
   start_row = 0
   end_row = 199  # Adjust as needed
   ```

### Step 2: Configure API Keys
```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
os.environ["SERPAPI_API_KEY"] = "your_serpapi_key"
```

### Step 3: Extract Text Data
Run the text extraction cell to create `extracted_data.json` from the selected datapoints.

### Step 4: Generate Application Corpus
Execute the main application corpus generation code. The pipeline will:
1. Generate legal search queries for each base principle
2. Retrieve relevant content from trusted legal sources
3. Scrape and clean webpage content
4. Summarize retrieved content
5. Generate final legal reasoning
6. Save results to `application_corpus.csv`

**Note**: The process includes a 3-second delay between entries to respect API rate limits.

## Trusted Legal Sources

The system filters search results to include only these verified legal domains:
- indiankanoon.org
- barandbench.com
- livelaw.in
- scobserver.in
- legalserviceindia.com

## Pipeline Components

### 1. Search Query Generation
Uses LLM to create targeted legal search queries from base principles.

### 2. Multi-Source Retrieval
Fetches top-3 relevant documents from trusted legal sources using SerpAPI.

### 3. Content Extraction
Scrapes and cleans webpage content, extracting main legal text (limited to 4000 characters per source).

### 4. Summarization
LLM-based summarization connecting retrieved cases to base principles (max 6 sentences).

### 5. Reasoning Generation
Generates cause-effect legal reasoning statements linking principles to applications.

## Output Format

### application_corpus.csv
| Column | Description |
|--------|-------------|
| Base Text | Original legal principle from knowledge corpus |
| Sources | URLs of retrieved legal documents (semicolon-separated) |
| Retrieved Summary | LLM-generated summary of how cases apply to the principle |
| Final Reasoning | Legal reasoning connecting principle to application |

## Next Steps

1. **Vector Database Conversion**
   - Convert knowledge corpus to vector embeddings
   - Convert application corpus to vector embeddings
   - Choose appropriate embedding model for legal domain

2. **RAG Application Development**
   - Implement dual-corpus retrieval mechanism
   - Design query routing logic
   - Build response generation pipeline

3. **Evaluation**
   - Measure retrieval accuracy
   - Assess legal reasoning quality
   - Compare against baseline RAG

4. **Refinement**
   - Fine-tune retrieval parameters
   - Optimize corpus balance
   - Improve reasoning coherence

## File Structure
```
.
├── README.md
├── applicationcorpus.ipynb        # Main pipeline notebook
├── nyayanumana_knowledge.csv      # Source knowledge corpus
├── selected_datapoints.csv        # Extracted subset
├── extracted_data.json            # Parsed text data
├── application_corpus.csv         # Generated application corpus
└── progress.txt                   # Development tracking
```

## Error Handling

The pipeline includes robust error handling for:
- Missing API keys
- Failed HTTP requests
- Parsing errors in CSV/JSON files
- Empty search results
- Network timeouts

## Configuration Options

### Adjustable Parameters
- `start_row` / `end_row`: Control dataset size
- `top_k`: Number of search results per query (default: 3)
- `text_limit`: Maximum characters per scraped page (default: 4000)
- `delay`: Seconds between API calls (default: 3)

## License
[Add your license information]

## Contributors
[Add contributor information]

## Acknowledgments
- RAG+ Research Paper authors
- Indian legal source websites
- Groq and SerpAPI for their services

## Contact
[Add contact information]