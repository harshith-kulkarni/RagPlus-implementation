# Setup Guide for Cloned Repository

## 🚀 Quick Start

### Option 1: Use Your Own API Keys (Recommended)

**Step 1: Get API Keys**

1. **Pinecone** (Free Tier)
   - Go to https://www.pinecone.io/
   - Sign up for free account
   - Create a new project
   - Copy your API key

2. **Groq** (Free Tier)
   - Go to https://console.groq.com/
   - Sign up for free account
   - Generate API key
   - Copy your API key

**Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Configure Environment**

Create a `.env` file in the root directory:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Step 4: Index the Data**

This creates YOUR own Pinecone indices with the data:

```bash
# Index legal domain (takes ~2-3 minutes)
python run_legal_indexing.py

# Index math domain (takes ~5-7 minutes)
python run_math_indexing.py
```

**Expected Output:**
```
Legal Domain:
- legal-knowledge-corpus: 50 vectors
- legal-application-corpus: 220 vectors

Math Domain:
- math-knowledge-corpus: 62 vectors
- math-application-corpus: 800 vectors
```

**Step 5: Run the Application**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

### Option 2: Use Shared API Keys (If Provided)

If someone has shared their API keys with you:

**Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Create .env File**
```bash
PINECONE_API_KEY=shared_pinecone_key
GROQ_API_KEY=shared_groq_key
```

**Step 3: Run the Application**
```bash
streamlit run app.py
```

**Note**: You'll be using shared indices. No need to run indexing scripts.

---

## 📊 Verify Setup

### Check Pinecone Indices

```bash
python check_sample_data.py
```

**Expected Output:**
```
INDEX: legal-knowledge-corpus
Total Vectors: 50

INDEX: legal-application-corpus
Total Vectors: 220

INDEX: math-knowledge-corpus
Total Vectors: 62

INDEX: math-application-corpus
Total Vectors: 800
```

### Generate Metrics

```bash
python evaluate_both_domains.py
```

This creates:
- `metrics_summary.csv`
- `legal_metrics_detailed.csv`
- `math_metrics_detailed.csv`

---

## 🔧 Troubleshooting

### Issue: "Pinecone index not found"

**Solution**: Run the indexing scripts
```bash
python run_legal_indexing.py
python run_math_indexing.py
```

### Issue: "API key invalid"

**Solution**: Check your .env file
```bash
# Verify .env exists
cat .env  # Linux/Mac
type .env  # Windows

# Make sure keys are correct (no quotes, no spaces)
PINECONE_API_KEY=pc-xxxxx
GROQ_API_KEY=gsk_xxxxx
```

### Issue: "Rate limit exceeded"

**Solution**: Wait and adjust settings
- Wait 60 seconds
- Use lower word count (100-150)
- Use fewer documents (top_k=2)
- Wait 3-5 seconds between queries

### Issue: "Module not found"

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## 💰 Cost Information

### Pinecone Free Tier
- **1 index** (you need 4 - requires paid plan)
- **100K vectors** (you'll use 1,132)
- **Unlimited queries**

**Note**: Free tier allows only 1 index. To run this project, you need:
- **Starter Plan**: $70/month (5 indexes)
- **OR**: Modify code to use 1 index with namespaces

### Groq Free Tier
- **30 requests/minute**
- **14,400 tokens/minute**
- **Free forever**

**Sufficient for**: Development, testing, demos

---

## 🎯 Alternative: Single Index Setup (Free Tier)

If you want to use Pinecone free tier (1 index only):

**Modify the indexing scripts to use namespaces:**

```python
# Instead of 4 separate indices, use 1 index with 4 namespaces:
# - namespace: "legal-knowledge"
# - namespace: "legal-application"
# - namespace: "math-knowledge"
# - namespace: "math-application"
```

**Contact me if you need help with this setup.**

---

## 📚 Data Sources

### Legal Domain
- Source: Securities & SEBI regulations
- Knowledge: 50 legal concepts
- Applications: 220 legal cases

### Math Domain
- Source: Mathematical problems dataset
- Knowledge: 62 mathematical concepts
- Applications: 800 problem examples

---

## 🆘 Need Help?

1. **Check Documentation**:
   - README.md
   - DEPLOYMENT_GUIDE.md
   - METRICS_GUIDE.md

2. **Common Issues**:
   - API keys: Check .env file format
   - Indices: Run indexing scripts
   - Rate limits: Wait and reduce settings

3. **Open an Issue**:
   - GitHub Issues: https://github.com/harshith-kulkarni/RagPlus-implementation/issues

---

## ✅ Setup Checklist

- [ ] Cloned repository
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Got Pinecone API key
- [ ] Got Groq API key
- [ ] Created `.env` file
- [ ] Ran legal indexing script
- [ ] Ran math indexing script
- [ ] Verified indices (`python check_sample_data.py`)
- [ ] Generated metrics (`python evaluate_both_domains.py`)
- [ ] Ran application (`streamlit run app.py`)
- [ ] Tested both domains
- [ ] Tested all three AI modes
- [ ] Tested summarization feature

---

**Setup Time**: 15-20 minutes
**Status**: Ready to use!
