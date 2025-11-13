import pandas as pd
from datasets import load_dataset
import re
import requests
from tqdm import tqdm
import time
import hashlib
from sentence_transformers import SentenceTransformer # Re-import as it was previously removed
import json

print("Starting RAG+ Dual Corpus Builder...")
print("=" * 80)

# Configuration
SERPAPI_KEY = "5253d9530c8bb37f77b4eeb42d1a2643cd405f346a89f2aa574d50423de91279"
SERPAPI_BASE = "https://serpapi.com/search"
MAX_KNOWLEDGE_POINTS = 50
CHECKPOINT_INTERVAL = 5

# Re-loading embedding model as it's needed for embedding generation
print("\n[1/5] Loading embedding model...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error loading embedding model: {e}")
    exit()

# Load dataset
print("\n[2/5] Loading dataset...")
try:
    dataset = load_dataset('axondendriteplus/legal-rag-embedding-dataset', split='train')
    print(f"✓ Dataset loaded: {len(dataset)} rows")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit()

# Inspect first row
print("\n" + "=" * 80)
print("DATASET STRUCTURE:")
print("=" * 80)
first_row = dataset[0]
for key, value in first_row.items():
    print(f"{key}: {str(value)[:150]}...")
print("=" * 80 + "\n")


def generate_id(text):
    """Generate unique 12-char ID"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def extract_sections(text):
    """Extract all types of legal section references"""
    if not text or not isinstance(text, str):
        return []

    sections = set()

    # Comprehensive patterns for all legal references
    patterns = [
        (r'Section\s+(\d+[A-Z]*(?:\([a-z0-9]+\))?)', 'Section'),
        (r'Sec\.\s*(\d+[A-Z]*)', 'Section'),
        (r'IPC\s+(\d+[A-Z]*)', 'IPC'),
        (r'(\d{3}[A-Z]?)\s+IPC', 'IPC'),
        (r'CrPC\s+(\d+[A-Z]*)', 'CrPC'),
        (r'(\d{3}[A-Z]?)\s+CrPC', 'CrPC'),
        (r'Article\s+(\d+[A-Z]*)', 'Article'),
        (r'Rule\s+(\d+[A-Z]*)', 'Rule'),
        (r'Regulation\s+(\d+[A-Z]*)', 'Regulation'),
        (r'Order\s+(\d+[A-Z]*)', 'Order'),
        # New pattern to capture standalone numbers (e.g., '23C', '2', '3') without explicit prefixes
        r'\b(\d{1,3}[A-Z]?)(?=[^\w])', # Matches 1-3 digits optionally followed by a letter, if followed by non-word char
    ]

    for pattern_info in patterns:
        if isinstance(pattern_info, tuple):
            pattern, prefix = pattern_info
        else: # For new standalone number pattern
            pattern = pattern_info
            prefix = "Section" # Default prefix for these cases

        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            section_num = match.group(1)
            # Add logic to prevent adding prefixes for already prefixed sections or obvious non-sections
            if not any(known_prefix in match.group(0).lower() for known_prefix in ['section', 'sec.', 'ipc', 'crpc', 'article', 'rule', 'regulation', 'order']):
                 # Heuristic to avoid common numbers that aren't sections (e.g., dates, counts)
                 if not (len(section_num) == 4 and section_num.isdigit()) and not (len(section_num) == 1 and section_num.isdigit()): # Exclude 4-digit years, single digits
                    sections.add(f"{prefix} {section_num}")
            else:
                 sections.add(f"{prefix} {section_num}") # Keep explicit prefixes

    return list(sections)[:5] # Limit to top 5 most relevant


def extract_case_name(title):
    """Extract case name"""
    pattern = r'(.+?)\s+(?:vs\.?|versus|v\.)\s+(.+?)(?:\s*[-|:(\[]|$)'
    match = re.search(pattern, title, re.IGNORECASE)

    if match:
        plaintiff = match.group(1).strip()
        defendant = match.group(2).strip()
        # Remove common suffixes
        for noise in [' - Indian', ' | Legal', ' Case', ' Judgment', ' - Supreme', ' - High']:
            defendant = defendant.split(noise)[0].strip()
        return f"{plaintiff} vs {defendant}"

    # Fallback
    return re.sub(r'\s*[-|]\s*.*', '', title).strip()


def extract_year(text):
    """Extract year"""
    if not text:
        return None
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', str(text))
    return int(years[-1]) if years else None


def extract_court(text):
    """Extract court name"""
    if not text:
        return 'Unknown Court'

    text_lower = str(text).lower()

    if 'supreme court' in text_lower:
        return 'Supreme Court of India'
    elif 'high court' in text_lower:
        # Try to extract specific high court
        hc_match = re.search(r'(\w+)\s+high court', text_lower)
        if hc_match:
            return f'{hc_match.group(1).title()} High Court'
        return 'High Court'
    elif 'tribunal' in text_lower:
        return 'Tribunal'
    elif 'district' in text_lower:
        return 'District Court'

    return 'Court'


def search_cases(section, context=""):
    """Search for relevant cases"""
    try:
        # Build search query
        query = f'"{section}" India case law judgment site:indiankanoon.org'

        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "num": 3,
            "gl": "in",
            "hl": "en"
        }

        response = requests.get(SERPAPI_BASE, params=params, timeout=15)

        if response.status_code != 200:
            print(f"      SerpAPI returned status code: {response.status_code}")
            return []

        data = response.json()
        cases = []

        for result in data.get("organic_results", [])[:3]:
            title = result.get('title', '')
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            # Must have case indicators
            if not any(kw in (title + snippet).lower() for kw in ['vs', 'v.', 'versus', 'case']):
                continue

            case_name = extract_case_name(title)

            cases.append({
                'case_name': case_name,
                'case_id': generate_id(case_name + link),
                'url': link,
                'year': extract_year(snippet + " " + title),
                'court': extract_court(snippet + " " + title),
                'section': section,
                'snippet': snippet
            })

        return cases

    except Exception as e:
        print(f"      Search error: {str(e)[:50]}")
        return []


# Initialize storage
knowledge_corpus = []
application_corpus = []
processed_knowledge_ids = set()
processed_application_ids = set()

print("\n" + "=" * 80)
print("BUILDING DUAL CORPUS")
print("=" * 80 + "\n")

successful = 0

for idx in tqdm(range(len(dataset)), desc='Processing'):
    if successful >= MAX_KNOWLEDGE_POINTS:
        break

    row = dataset[idx]

    # Extract data - handle different field names
    question = str(row.get('question', row.get('input', row.get('text', ''))) or '')
    # FIXED SYNTAX ERROR HERE
    context = str(row.get('context', row.get('background', '')) or '')
    # Implemented answer fallback logic
    answer = str(row.get('answer', row.get('output', row.get('response', context)))) or ''

    # Skip if too short
    if len(question) < 10 or len(answer) < 10:
        continue

    # Extract sections
    all_text = question + " " + answer + " " + context
    sections = extract_sections(all_text)

    if not sections:
        print(f"Skipping row {idx}: No sections found.")
        continue

    # Generate knowledge ID
    knowledge_id = generate_id(question + answer)

    if knowledge_id in processed_knowledge_ids:
        print(f"Skipping row {idx}: Duplicate knowledge ID.")
        continue

    print(f"\n{'='*80}")
    print(f"[Row {idx}] Sections: {sections}")
    print(f"Question: {question[:80]}...")

    # Create KNOWLEDGE CORPUS entry
    try:
        # Get embedding for statutory text (answer contains the law explanation)
        knowledge_embedding = embedding_model.encode(answer).tolist()

        knowledge_entry = {
            'knowledge_id': knowledge_id,
            'point_type': 'KNOWLEDGE',
            'section_reference': ', '.join(sections),
            'statutory_text': answer,  # Complete text, no truncation
            'original_question': question,  # Complete text
            'context': context,  # Complete text
            'embedding': json.dumps(knowledge_embedding),
            'metadata': json.dumps({
                'section_count': len(sections),
                'has_context': bool(context),
                'question_length': len(question),
                'answer_length': len(answer)
            })
        }

        knowledge_corpus.append(knowledge_entry)
        processed_knowledge_ids.add(knowledge_id)
        print(f"✓ Knowledge corpus entry created")

    except Exception as e:
        print(f"✗ Knowledge error: {e}")
        continue

    # Search for APPLICATION CORPUS entries (cases)
    total_cases = 0
    cases_per_section = {}

    # Search ALL sections (not just first 2) for comprehensive coverage
    for section in sections:
        print(f"  → Searching cases for: {section}")

        cases = search_cases(section, context)

        if not cases:
            print(f"    ✗ No cases found for {section}")
            cases_per_section[section] = 0
            continue

        print(f"    ✓ Found {len(cases)} case(s) for {section}")
        section_case_count = 0

        for case in cases:
            app_id = case['case_id']

            # Check if this specific case is already linked to the current knowledge point
            # Avoids duplicate entries for the same case linked to the same knowledge point
            if any(app['application_id'] == app_id and app['knowledge_id'] == knowledge_id for app in application_corpus):
                print(f"      • Case {case['case_name'][:50]}... already linked to this knowledge point.")
                continue

            try:
                # Validate required fields before creating entry
                if not case['case_name'] or not case['url']:
                    print(f"      ⚠ Skipping invalid case (missing name/url)")
                    continue

                # Generate embedding for case snippet
                app_embedding = embedding_model.encode(case['snippet']).tolist()

                application_entry = {
                    'application_id': app_id,
                    'knowledge_id': knowledge_id,  # CRITICAL: Link to knowledge corpus
                    'point_type': 'APPLICATION',
                    'case_name': case['case_name'],
                    'section_applied': section,
                    'year': case['year'] if case['year'] else 0,
                    'court': case['court'],
                    'case_summary': case['snippet'],  # Complete snippet
                    'judgment_url': case['url'],
                    'embedding': json.dumps(app_embedding),
                    'metadata': json.dumps({
                        'has_year': case['year'] is not None,
                        'court_level': case['court'],
                        'source': 'Indian Kanoon' if 'indiankanoon' in case['url'] else 'Legal DB',
                        'linked_section': section  # Track which section triggered this link
                    })
                }

                application_corpus.append(application_entry)
                processed_application_ids.add(app_id)
                total_cases += 1
                section_case_count += 1

                print(f"      • Added: {case['case_name'][:60]}")

            except Exception as e:
                print(f"      ✗ Application error for case {case.get('case_name', 'N/A')}: {e}")
                continue

        cases_per_section[section] = section_case_count

        # Rate limiting to avoid hitting API limits too quickly
        time.sleep(2)

    successful += 1
    print(f"✓ Entry #{successful}: 1 knowledge + {total_cases} applications")

    # Checkpoint save
    if successful % CHECKPOINT_INTERVAL == 0:
        pd.DataFrame(knowledge_corpus).to_csv('knowledge_corpus.csv', index=False)
        pd.DataFrame(application_corpus).to_csv('application_corpus.csv', index=False)
        print(f"\n💾 CHECKPOINT: {successful} entries saved\n")

# Final save
print("\n" + "=" * 80)
print("SAVING FINAL CORPUS...")
print("=" * 80)

df_knowledge = pd.DataFrame(knowledge_corpus)
df_application = pd.DataFrame(application_corpus)

df_knowledge.to_csv('knowledge_corpus.csv', index=False)
df_application.to_csv('application_corpus.csv', index=False)

print("\n" + "=" * 80)
print("✓ RAG+ DUAL CORPUS COMPLETE!")
print("=" * 80)
print(f"Knowledge Corpus Entries:    {len(knowledge_corpus)}")
print(f"Application Corpus Entries:  {len(application_corpus)}")
if knowledge_corpus:
    print(f"Avg applications/knowledge:  {len(application_corpus)/len(knowledge_corpus):.1f}")
print("=" * 80)

# Display samples
if knowledge_corpus:
    print("\n📋 SAMPLE KNOWLEDGE ENTRY:")
    print(f"ID: {knowledge_corpus[0]['knowledge_id']}")
    print(f"Sections: {knowledge_corpus[0]['section_reference']}")
    print(f"Question: {knowledge_corpus[0]['original_question'][:100]}...")
    print(f"Statutory: {knowledge_corpus[0]['statutory_text'][:150]}...")

if application_corpus:
    print("\n⚖️  SAMPLE APPLICATION ENTRY:")
    print(f"ID: {application_corpus[0]['application_id']}")
    print(f"Linked to Knowledge: {application_corpus[0]['knowledge_id']}")
    print(f"Case: {application_corpus[0]['case_name']}")
    print(f"Section: {application_corpus[0]['section_applied']}")
    print(f"Court: {application_corpus[0]['court']}")
    print(f"Summary: {application_corpus[0]['case_summary'][:100]}...")

print("\n✓ Files saved: knowledge_corpus.csv, application_corpus.csv")