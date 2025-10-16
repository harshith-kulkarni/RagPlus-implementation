# RAG+ Dual Corpus — Indian Legal Reasoning

This repository implements a Retrieval-Augmented Generation (RAG+) pipeline for the Indian legal domain using two corpora:
- Knowledge corpus: extracted legal texts and judgments.
- Application corpus: real-world application examples and LLM-synthesized reasoning.

## Project structure
- applicationcorpus.ipynb
- nyayanumana_knowledge.csv
- extracted_data.json
- application_corpus.csv
- selected_datapoints.csv
- Rag+.pdf
- progress.txt
- assets/semantic_similarity.png  (add image here)

## Summary
- Generated 200 datapoints for both knowledge and application corpora.
- Each datapoint includes: knowledge point, application point, reasoning, and case summary produced by the LLM.
- Semantic similarity between corpora was computed to verify overlap and relevance.

## Progress (latest)
IN THIS PROJECT WE ARE TRYING TO IMPLEMENT RAGPLUS RESEARCH PAPER FOR LEGAL DOMAIN BASED ON INDIAN LAW AND JUDICIARY SYSTEM.

TILL NOW:
- GENERATED 200 DATAPOINTS FOR KNOWLEDGE CORPUS AND APPLICATION CORPUS
- USING TEXT SECTION, THE COMPLETE DATA IS SENT TO THE LLM AND THEN KNOWLEDGE POINT, APPLICATION POINT, REASONING, SUMMARY OF CASE ARE GENERATED
- THE DATA IS APPENDED TO THE RELATIVE CORPORA
- SEMANTIC SIMILARITY OF BOTH THE CORPORA ARE TESTED

RESULTS:
- Mean Semantic Similarity Score: 0.631759524345398

Below: distribution of semantic similarity scores across the generated datapoints.



## Run notes
- Dataset : nyayanumana_knowledge.csv
- See `corpus_building.ipynb` for corpus creation and similarity computation steps.
- Ensure required API keys and Python dependencies are configured before running the notebook.
