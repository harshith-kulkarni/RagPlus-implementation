from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

indexes = [
    "legal-knowledge-corpus",
    "legal-application-corpus", 
    "math-knowledge-corpus",
    "math-application-corpus"
]

print("="*80)
print("SAMPLE DATA FROM ALL 4 INDEXES")
print("="*80)

for idx_name in indexes:
    print(f"\n{'='*80}")
    print(f"INDEX: {idx_name}")
    print("="*80)
    
    try:
        index = pc.Index(idx_name)
        stats = index.describe_index_stats()
        print(f"Total Vectors: {stats['total_vector_count']}")
        print(f"Dimension: {stats['dimension']}")
        
        # Query for 1 sample vector
        dummy_vector = [0.0] * 384
        results = index.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True
        )
        
        if results['matches']:
            match = results['matches'][0]
            print(f"\nSAMPLE VECTOR:")
            print(f"  ID: {match['id']}")
            print(f"  Score: {match['score']:.4f}")
            print(f"\n  METADATA:")
            for key, value in match['metadata'].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
        else:
            print("  No vectors found!")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*80)
print("SAMPLE CHECK COMPLETE")
print("="*80)
