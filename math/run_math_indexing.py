import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("MATH DOMAIN - INDEXING TO PINECONE")
print("="*80)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
KNOWLEDGE_INDEX = "math-knowledge-corpus"
APPLICATION_INDEX = "math-application-corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384
BATCH_SIZE = 100

print("\n[1/4] Loading CSVs...")
knowledge_df = pd.read_csv('math/knowledge_corpus.csv')
application_df = pd.read_csv('math/application_corpus.csv')
print(f"Knowledge: {len(knowledge_df)} entries")
print(f"Application: {len(application_df)} entries")

print("\n[2/4] Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"Model loaded: {EMBEDDING_MODEL}")

print("\n[3/4] Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index(index_name):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} created")
    else:
        print(f"Index {index_name} exists")

create_index(KNOWLEDGE_INDEX)
create_index(APPLICATION_INDEX)

knowledge_index = pc.Index(KNOWLEDGE_INDEX)
application_index = pc.Index(APPLICATION_INDEX)

print("\n[4/4] Processing knowledge corpus...")
vectors = []
for idx, row in tqdm(knowledge_df.iterrows(), total=len(knowledge_df)):
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
            if len(vectors) >= BATCH_SIZE:
                knowledge_index.upsert(vectors=vectors)
                vectors = []
        except:
            continue

if vectors:
    knowledge_index.upsert(vectors=vectors)

print(f"Knowledge corpus uploaded: {knowledge_index.describe_index_stats()['total_vector_count']} vectors")

print("\nProcessing application corpus...")
vectors = []
for idx, row in tqdm(application_df.iterrows(), total=len(application_df)):
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
            if len(vectors) >= BATCH_SIZE:
                application_index.upsert(vectors=vectors)
                vectors = []
        except:
            continue

if vectors:
    application_index.upsert(vectors=vectors)

print(f"Application corpus uploaded: {application_index.describe_index_stats()['total_vector_count']} vectors")

print("\n"+"="*80)
print("MATH DOMAIN INDEXING COMPLETE!")
print("="*80)
print(f"Knowledge index: {KNOWLEDGE_INDEX} - {knowledge_index.describe_index_stats()['total_vector_count']} vectors")
print(f"Application index: {APPLICATION_INDEX} - {application_index.describe_index_stats()['total_vector_count']} vectors")
