# Install required packages
# !pip install -q pinecone-client sentence-transformers pandas tqdm

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import json
import time

print("=" * 80)
print("DUAL CORPUS TO PINECONE VECTORDB BUILDER")
print("=" * 80)

# Configuration
PINECONE_API_KEY = "pcsk_6WURux_2kZ1ZsvMZJaDqJ1m5nRdvte2Shrfu5frLguCkp9ZWncmdEqyeXUWpQ26jqUn2eK"
KNOWLEDGE_INDEX = "legal-knowledge-corpus"
APPLICATION_INDEX = "legal-application-corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
BATCH_SIZE = 100

# Initialize
print("\n[1/4] Loading CSVs...")
try:
    knowledge_df = pd.read_csv('knowledge_corpus.csv')
    application_df = pd.read_csv('application_corpus.csv')
    print(f"✓ Knowledge corpus: {len(knowledge_df)} entries")
    print(f"✓ Application corpus: {len(application_df)} entries")
except Exception as e:
    print(f"✗ Error loading CSVs: {e}")
    print("Make sure knowledge_corpus.csv and application_corpus.csv exist!")
    exit()

# Load embedding model
print("\n[2/4] Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded: {EMBEDDING_MODEL} (dimension: {DIMENSION})")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit()

# Initialize Pinecone
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
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        print(f"  Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"  ✓ Index {index_name} created and ready")
    else:
        print(f"  ✓ Index {index_name} already exists")


# Create indexes
create_index_if_not_exists(KNOWLEDGE_INDEX)
create_index_if_not_exists(APPLICATION_INDEX)

# Get index connections
knowledge_index = pc.Index(KNOWLEDGE_INDEX)
application_index = pc.Index(APPLICATION_INDEX)

print("✓ Connected to Pinecone indexes")


def chunk_text(text, max_length=500):
    """
    Intelligent chunking for long texts
    Preserves sentence boundaries
    """
    if not text or pd.isna(text) or len(str(text)) <= max_length:
        return [str(text)]
    
    text = str(text)
    
    # Split by sentences
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_length]]


def process_knowledge_corpus():
    """Process and upload knowledge corpus to Pinecone"""
    print("\n" + "=" * 80)
    print("[4/4] PROCESSING KNOWLEDGE CORPUS")
    print("=" * 80)
    
    vectors = []
    total_chunks = 0
    
    for idx, row in tqdm(knowledge_df.iterrows(), total=len(knowledge_df), desc="Encoding knowledge"):
        # Create composite text for embedding (using pre-computed embedding if available)
        if 'embedding' in row and pd.notna(row['embedding']):
            # Use existing embedding from CSV
            try:
                embedding = json.loads(row['embedding'])
                
                # Create vector
                metadata = {
                    'knowledge_id': row['knowledge_id'],
                    'point_type': 'KNOWLEDGE',
                    'section_reference': str(row['section_reference']),
                    'statutory_text': str(row['statutory_text']),
                    'original_question': str(row['original_question']),
                    'context': str(row.get('context', '')),
                    'text_preview': str(row['statutory_text'])
                }
                
                vectors.append({
                    'id': row['knowledge_id'],
                    'values': embedding,
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                # Upload in batches
                if len(vectors) >= BATCH_SIZE:
                    knowledge_index.upsert(vectors=vectors)
                    vectors = []
                    
            except Exception as e:
                print(f"\n✗ Error processing row {idx}: {e}")
                continue
        else:
            # Generate new embedding if not present
            composite_text = f"""
            Section: {row['section_reference']}
            Question: {row['original_question']}
            Law: {row['statutory_text']}
            Context: {row.get('context', '')}
            """.strip()
            
            # Chunk if too long
            chunks = chunk_text(composite_text, max_length=500)
            
            for chunk_idx, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk).tolist()
                
                vector_id = f"{row['knowledge_id']}_chunk_{chunk_idx}" if len(chunks) > 1 else row['knowledge_id']
                
                metadata = {
                    'knowledge_id': row['knowledge_id'],
                    'point_type': 'KNOWLEDGE',
                    'section_reference': str(row['section_reference']),
                    'statutory_text': str(row['statutory_text']),
                    'original_question': str(row['original_question']),
                    'context': str(row.get('context', '')),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'text_preview': chunk
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                if len(vectors) >= BATCH_SIZE:
                    knowledge_index.upsert(vectors=vectors)
                    vectors = []
    
    # Upload remaining vectors
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
        # Use existing embedding if available
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
                    'text_preview': str(row['case_summary'])
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
        else:
            # Generate new embedding
            composite_text = f"""
            Case: {row['case_name']}
            Section Applied: {row['section_applied']}
            Court: {row['court']}
            Year: {row['year']}
            Summary: {row['case_summary']}
            """.strip()
            
            chunks = chunk_text(composite_text, max_length=500)
            
            for chunk_idx, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk).tolist()
                
                vector_id = f"{row['application_id']}_chunk_{chunk_idx}" if len(chunks) > 1 else row['application_id']
                
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
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'text_preview': chunk
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                if len(vectors) >= BATCH_SIZE:
                    application_index.upsert(vectors=vectors)
                    vectors = []
    
    # Upload remaining
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

# Test retrieval
print("\n" + "=" * 80)
print("TESTING RETRIEVAL")
print("=" * 80)

test_query = "What is the punishment for murder?"
print(f"\nTest Query: '{test_query}'")
test_embedding = embedding_model.encode(test_query).tolist()

knowledge_results = knowledge_index.query(
    vector=test_embedding,
    top_k=2,
    include_metadata=True
)

if knowledge_results['matches']:
    print(f"\n✓ Knowledge Retrieval Working!")
    top_match = knowledge_results['matches'][0]
    print(f"  Score: {top_match['score']:.4f}")
    print(f"  Section: {top_match['metadata']['section_reference']}")
    print(f"  Preview: {top_match['metadata']['text_preview'][:100]}...")

application_results = application_index.query(
    vector=test_embedding,
    top_k=2,
    include_metadata=True
)

if application_results['matches']:
    print(f"\n✓ Application Retrieval Working!")
    top_match = application_results['matches'][0]
    print(f"  Score: {top_match['score']:.4f}")
    print(f"  Case: {top_match['metadata']['case_name']}")
    print(f"  Court: {top_match['metadata']['court']}")

print("\n" + "=" * 80)
print("✓ VECTORDB READY FOR RAG+ QUERIES!")
print("=" * 80)