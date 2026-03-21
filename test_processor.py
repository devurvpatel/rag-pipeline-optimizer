import os
from backend.utils.document_processor import DocumentProcessor

# Point this at whatever PDF you have in data/sample_docs/
PDF_PATH = "data/sample_docs/n8n guide.pdf"

processor = DocumentProcessor()

# Load
docs = processor.load_pdf(PDF_PATH)

# Run all 3 chunking strategies
fixed_chunks    = processor.chunk_fixed(docs)
recursive_chunks = processor.chunk_recursive(docs)
semantic_chunks = processor.chunk_semantic(docs)

# Compare stats
print("\n===== CHUNKING COMPARISON =====")
for name, chunks in [
    ("Fixed 512",  fixed_chunks),
    ("Recursive 1024", recursive_chunks),
    ("Semantic",   semantic_chunks),
]:
    stats = processor.get_chunk_stats(chunks)
    print(f"\n{name}:")
    print(f"  Chunks:     {stats['count']}")
    print(f"  Avg length: {stats['avg_length']} chars")
    print(f"  Min length: {stats['min_length']} chars")
    print(f"  Max length: {stats['max_length']} chars")
