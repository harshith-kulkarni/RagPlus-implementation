import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import uuid
from tqdm import tqdm

print("="*80)
print("CONVERTING ALL MATH JSON DATA TO CSV WITH EMBEDDINGS")
print("="*80)

# Load embedding model
print("\n[1/5] Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded: all-MiniLM-L6-v2")

# Load JSON files
print("\n[2/5] Loading JSON files...")
with open('math/knowledge_corpus_math.json', 'r', encoding='utf-8') as f:
    knowledge_data = json.load(f)
    
with open('math/application_corpus_maths.json', 'r', encoding='utf-8') as f:
    application_data = json.load(f)

print(f"✓ Knowledge entries loaded: {len(knowledge_data)}")
print(f"✓ Application entries loaded: {len(application_data)}")

# Limit application data to top 800
MAX_APPLICATIONS = 800
if len(application_data) > MAX_APPLICATIONS:
    print(f"⚠️  Limiting application corpus to top {MAX_APPLICATIONS} entries (from {len(application_data)})")
    application_data = application_data[:MAX_APPLICATIONS]
    print(f"✓ Using {len(application_data)} application entries")

# Convert knowledge corpus
print("\n[3/5] Converting knowledge corpus...")
knowledge_rows = []
for item in tqdm(knowledge_data, desc="Processing knowledge"):
    # Create text for embedding
    text = f"{item['title']}. {item['content']}"
    embedding = model.encode(text).tolist()
    
    # Get category if exists, otherwise use 'general'
    category = item.get('category', 'general')
    
    row = {
        'knowledge_id': item['knowledge_id'],
        'point_type': 'KNOWLEDGE',
        'section_reference': item['title'],
        'statutory_text': item['content'],
        'original_question': f"What is {item['title']}?",
        'context': f"Category: {category}. Mathematical concept.",
        'embedding': json.dumps(embedding),
        'metadata': json.dumps({
            'category': category,
            'title': item['title']
        })
    }
    knowledge_rows.append(row)

knowledge_df = pd.DataFrame(knowledge_rows)
knowledge_df.to_csv('math/knowledge_corpus.csv', index=False)
print(f"✓ Saved knowledge_corpus.csv with {len(knowledge_df)} entries")

# Convert application corpus
print("\n[4/5] Converting application corpus...")
application_rows = []
for item in tqdm(application_data, desc="Processing applications"):
    # Create text for embedding
    text = f"{item['problem']} {item['solution']}"
    embedding = model.encode(text).tolist()
    
    # Get app_id or generate one
    app_id = item.get('app_id', str(uuid.uuid4())[:12])
    
    # Get knowledge links or use category
    knowledge_id = 'general'
    if 'knowledge_links' in item and len(item['knowledge_links']) > 0:
        knowledge_id = item['knowledge_links'][0]
    elif 'category' in item:
        knowledge_id = f"category_{item['category']}"
    
    # Get category
    category = item.get('category', 'general')
    
    row = {
        'application_id': app_id,
        'knowledge_id': knowledge_id,
        'point_type': 'APPLICATION',
        'case_name': item['problem'][:200],
        'section_applied': category.replace('_', ' ').title(),
        'year': 2024,
        'court': 'Academic',
        'case_summary': f"Problem: {item['problem']} Solution: {item['solution']} Options: {item.get('options', 'N/A')} Correct: {item.get('correct_option', 'N/A')}",
        'judgment_url': f"https://example.com/math-problem/{app_id}",
        'embedding': json.dumps(embedding),
        'metadata': json.dumps({
            'category': category,
            'correct_option': item.get('correct_option', ''),
            'options': item.get('options', '')
        })
    }
    application_rows.append(row)

application_df = pd.DataFrame(application_rows)
application_df.to_csv('math/application_corpus.csv', index=False)
print(f"✓ Saved application_corpus.csv with {len(application_df)} entries")

# Summary
print("\n[5/5] Conversion Summary")
print("="*80)
print(f"✓ Knowledge corpus: {len(knowledge_df)} entries → math/knowledge_corpus.csv")
print(f"✓ Application corpus: {len(application_df)} entries → math/application_corpus.csv")
print("\n📊 Breakdown by category:")

# Count categories in knowledge
if len(knowledge_df) > 0:
    print("\nKnowledge categories:")
    for cat, count in knowledge_df['section_reference'].value_counts().head(10).items():
        print(f"  - {cat}: {count}")

# Count categories in application
if len(application_df) > 0:
    print("\nApplication categories:")
    for cat, count in application_df['section_applied'].value_counts().head(10).items():
        print(f"  - {cat}: {count}")

print("\n" + "="*80)
print("✅ CONVERSION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Delete old math indices: python fix_math_indices.py")
print("2. Index to Pinecone: python run_math_indexing.py")
print("3. Run the app: streamlit run app.py")
