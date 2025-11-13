# Math Domain - Dual Corpus to Pinecone VectorDB Builder

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("MATH DOMAIN - DUAL CORPUS TO PINECONE VECTORDB BUILDER")
print("=" * 80)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Create separate indexes for math domain
KNOWLEDGE_INDEX = "math-knowledge-corpus"
APPLICATION_INDEX = "math-application-corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384
BATCH_SIZE = 100

# Initialize
print("\n[1/4] Loading CSVs...")
try:
    knowledge_df = pd.read_csv('math/knowledge_corpus.csv')
    application_df = pd.read_csv('math/application_corpus.csv')
    print(f"✓ Knowledge corpus: {len(knowledge_df)} entries")
    print(f"✓ Application corpus: {len(application_df)} entries")
except Exception as e:
    print(f"✗ Error loading CSVs: {e}")
    exit()

print("\n[2/4] Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded: {EMBEDDING_MODEL} (dimension: {DIMENSION})")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit()

print("\n[3/4] Initializing Pinecone...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✓ Pinecone client initialized")
except Exception as e:
    print(f"✗ Error initializing Pinecone: {e}")
    exit()


def create_index_if_not_exists(index_name):
    """Create Pinecone index if it doesn't exist"""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"  Creating index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"  Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"  ✓ Index {index_name} created and ready")
    else:
        print(f"  ✓ Index {index_name} already exists")


create_index_if_not_exists(KNOWLEDGE_INDEX)
create_index_if_not_exists(APPLICATION_INDEX)

knowledge_index = pc.Index(KNOWLEDGE_INDEX)
application_index = pc.Index(APPLICATION_INDEX)
print("✓ Connected to Pinecone indexes")
print(f"  Knowledge: {KNOWLEDGE_INDEX}")
print(f"  Application: {APPLICATION_INDEX}")


def process_knowledge_corpus():
    """Process and upload knowledge corpus to Pinecone"""
    print("\n" + "=" * 80)
    print("[4/4] PROCESSING KNOWLEDGE CORPUS")
    print("=" * 80)
    
    vectors = []
    total_chunks = 0
    
    for idx, row in tqdm(knowledge_df.iterrows(), total=len(knowledge_df), desc="Encoding knowledge"):
        if 'embedding' in row and pd.notna(row['embedding']):
            try:
                embedding = json.loads(row['embedding'])
                
                metadata = {
                    'knowledge_id': row['knowledge_id'],
                    'point_type': 'KNOWLEDGE',
                    'section_reference': str(row['section_reference']),
                    'statutory_text': str(row['statutory_text']),
                    'original_question': str(row['original_question']),
                    'context': str(row.get('context', '')),
                    'text_preview': str(row['statutory_text'])[:500]
                }
                
                vectors.append({
                    'id': row['knowledge_id'],
                    'values': embedding,
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                if len(vectors) >= BATCH_SIZE:
                    knowledge_index.upsert(vectors=vectors)
                    vectors = []
                    
            except Exception as e:
                print(f"\n✗ Error processing row {idx}: {e}")
                continue
    
    if vectors:
        knowledge_index.upsert(vectors=vectors)
    
    print(f"✓ Knowledge corpus uploaded: {total_chunks} vectors")
    print(f"  Index stats: {knowledge_index.describe_index_stats()['total_vector_count']} total vectors")


def process_application_corpus():
    """Process and upload application corpus to Pinecone"""
    print("\n" + "=" * 80)
    print("PROCESSING APPLICATION CORPUS")
    print("=" * 80)
    
    vectors = []
    total_chunks = 0
    
    for idx, row in tqdm(application_df.iterrows(), total=len(application_df), desc="Encoding applications"):
        if 'embedding' in row and pd.notna(row['embedding']):
            try:
                embedding = json.loads(row['embedding'])
                
                metadata = {
                    'application_id': row['application_id'],
                    'knowledge_id': row['knowledge_id'],
                    'point_type': 'APPLICATION',
                    'case_name': str(row['case_name']),
                    'section_applied': str(row['section_applied']),
                    'year': int(row['year']),
                    'court': str(row['court']),
                    'case_summary': str(row['case_summary']),
                    'judgment_url': str(row['judgment_url']),
                    'text_preview': str(row['case_summary'])[:500]
                }
                
                vectors.append({
                    'id': row['application_id'],
                    'values': embedding,
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                if len(vectors) >= BATCH_SIZE:
                    application_index.upsert(vectors=vectors)
                    vectors = []
                    
            except Exception as e:
                print(f"\n✗ Error processing row {idx}: {e}")
                continue
    
    if vectors:
        application_index.upsert(vectors=vectors)
    
    print(f"✓ Application corpus uploaded: {total_chunks} vectors")
    print(f"  Index stats: {application_index.describe_index_stats()['total_vector_count']} total vectors")


# Process both corpora
process_knowledge_corpus()
process_application_corpus()

# Final statistics
print("\n" + "=" * 80)
print("VECTORDB BUILD COMPLETE!")
print("=" * 80)

knowledge_stats = knowledge_index.describe_index_stats()
application_stats = application_index.describe_index_stats()

print(f"\n📚 KNOWLEDGE INDEX: {KNOWLEDGE_INDEX}")
print(f"   Total Vectors: {knowledge_stats['total_vector_count']}")
print(f"   Dimension: {knowledge_stats['dimension']}")

print(f"\n⚖️  APPLICATION INDEX: {APPLICATION_INDEX}")
print(f"   Total Vectors: {application_stats['total_vector_count']}")
print(f"   Dimension: {application_stats['dimension']}")

print("\n" + "=" * 80)
print("✓ MATH DOMAIN VECTORDB READY FOR RAG+ QUERIES!")
print("=" * 80)
