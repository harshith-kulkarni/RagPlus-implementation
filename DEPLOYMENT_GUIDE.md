# 🚀 RAG+ Legal AI - Deployment Guide

## 📋 Prerequisites

1. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
2. **GitHub Repository**: Push your code to GitHub
3. **API Keys**:
   - Pinecone API Key
   - Google Gemini API Key

## 🔧 Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ragplus-implementation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up local secrets**:
   - Edit `.streamlit/secrets.toml`
   - Add your API keys:
   ```toml
   PINECONE_API_KEY = "your_pinecone_api_key_here"
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

4. **Run locally**:
   ```bash
   streamlit run final_rag_app.py
   ```

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

1. **Push to GitHub** (make sure `.gitignore` excludes secrets):
   ```bash
   git add .
   git commit -m "Deploy RAG+ Legal AI"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Click "New app"**
3. **Connect your GitHub repository**
4. **Set the main file**: `final_rag_app.py`
5. **Click "Deploy"**

### Step 3: Add Secrets to Streamlit Cloud

1. **In your deployed app, click "Manage app"**
2. **Go to "Secrets" tab**
3. **Add your secrets**:
   ```toml
   PINECONE_API_KEY = "your_actual_pinecone_api_key"
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   ```
4. **Save secrets**

### Step 4: Restart the App

1. **Click "Reboot app"** to apply the secrets
2. **Your app should now work!**

## 🔍 Troubleshooting

### Common Issues:

1. **Pinecone Import Error**:
   - Make sure `requirements.txt` has `pinecone>=5.0.0` (not `pinecone-client`)

2. **API Key Errors**:
   - Check that secrets are properly set in Streamlit Cloud
   - Verify API keys are valid and have proper permissions

3. **Memory Issues**:
   - Streamlit Cloud has memory limits
   - Consider using smaller embedding models if needed

4. **Index Not Found**:
   - Make sure your Pinecone indexes exist
   - Run `indexing.py` locally first to create indexes

## 📊 Data Files

**Note**: The CSV files (`application_corpus.csv`, `knowledge_corpus.csv`) are excluded from git due to size. For deployment:

1. **Option 1**: Upload smaller sample datasets
2. **Option 2**: Use cloud storage (S3, Google Cloud) and load data dynamically
3. **Option 3**: Create indexes directly in Pinecone cloud

## 🎯 Production Considerations

1. **Rate Limiting**: Implement rate limiting for API calls
2. **Caching**: Use Streamlit caching effectively
3. **Error Handling**: Robust error handling for API failures
4. **Monitoring**: Set up logging and monitoring
5. **Security**: Never commit API keys to git

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all secrets are set correctly
3. Test locally first
4. Check API key permissions and quotas