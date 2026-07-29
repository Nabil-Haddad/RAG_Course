# LESSON 2: Data Structures for RAG


# WHY THIS MATTERS:
#   A RAG pipeline is a data transformation pipeline.
#   Text → chunks → embeddings → ranked results → answer.
#   Every step hands data to the next. If your data structures are loose, bugs appear three steps after the actual mistake.


# Lesson Content:
#   1. Python dataclasses: the Document, the core RAG data unit
#   2. Pydantic v2 models: validating LLM JSON output safely
#   3. List comprehensions: cleaning and filtering batches of chunks
#   4. Dict patterns: safe access, merging, iterating metadata


from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator
from typing import Optional
import json

# PART 1 — THE DOCUMENT DATACLASS

# A dataclass is a regular Python class with:
#   - __init__ generated automatically from the field definitions.
#   - __repr__ generated automatically (useful for debugging).
#   - Field defaults handled cleanly via field().

# We use a dataclass (not a plain dict) because:
#   - Autocomplete works: your editor knows what fields exist.
#   - Type errors are caught at write-time, not runtime.
#   - You can add methods (word_count, is_valid) that belong to the data.


@dataclass
class Document:
    # This is a single chunk of text ready for embedding and retrival
    # a primary data unit in the pipline
    content: str
    source : str
    chunk_id : int

    page_num : Optional[int] = None
    metadata : dict = field(default_factory=dict)
    embedding : list[int] = field(default_factory=list)

    # IMPORTANT: mutable defaults (dict, list) must use field(default_factory=...)
    # Never write `metadata: dict = {}` — that shares ONE dict across all instances.

    @property
    def word_count(self)->int:
        return len(self.content.split())

    @property
    def token_count_approx(self)->int:
        return int(self.word_count / 0.75)

    def is_valid(self)->bool:
        return self.word_count > 10
    
    def __repr__(self):
                return (
            f"Document(chunk_id={self.chunk_id!r}, "
            f"words={self.word_count}, "
            f"embedded={bool(self.embedding)}, "
            f"source={self.source!r})"
        )


print("Part 1: Document dataclass -----------")
 
doc = Document(
    content  = "Retrieval-Augmented Generation grounds LLM outputs in retrieved documents, reducing hallucinations.",
    source = "rag_primer.pdf",
    chunk_id = "rag_primer_p1_c001",
    page_num = 1,
    metadata = {"section": "introduction", "language": "en"},
)
print(doc)
print(f"word_count : {doc.word_count}")
print(f"token_count_approx: {doc.token_count_approx}")
print(f"is_valid : {doc.is_valid()}")
print(f"has embedding : {bool(doc.embedding)}") 
 
# Accessing metadata safely with .get() — never metadata["key"] directly
section = doc.metadata.get("section", "unknown")
print(f"section : {section}")



# PART 2 — PYDANTIC V2: VALIDATING LLM JSON OUTPUT

# When you ask an LLM to return structured JSON, it sometimes:
#   - Returns a float outside the expected range (confidence = 1.5)
#   - Returns an empty string for a required field
#   - Returns a string where you expected a list



# Pydantic catches all of these before they propagate downstream.
# It is the standard validation library used by LangChain and FastAPI.

# Pydantic v2 syntax (used here) differs from v1:
#   v1 used @validator       — DEPRECATED in v2
#   v2 uses @field_validator — correct, shown below


class RagAnswer(BaseModel):
      # we suppose that the we need from the llm to return a json output that has these 3 variables
      answer : str
      sources : list[str]
      confidence : float

      @field_validator('confidence')
      @classmethod
      def confidence_in_range(cls, value:float):
            if value <= 0.0 or value >= 1.0:
                  raise ValueError("confidence should be between 0 and 1")
            return round(value, 4)

      @field_validator('answer')
      @classmethod
      def answer_not_empty(cls, answer : str)->str:
            stripped = answer.strip()
            if not stripped:
                  raise ValueError("answer cannot be an empty string")
            return stripped


print("\n Part 2: Pydantic v2 validation -----------")

# valid json

valid_json = json.dumps({
    "answer" : "RAG reduces hallucinations by grounding outputs in retrieved documents.",
    "sources" : ["rag_primer.pdf", "langchain_docs.pdf"],
    "confidence" : 0.91
})

result = RagAnswer(**json.loads(valid_json))
print(result.answer)
print(result.sources)
print(result.confidence)

# Non valid json

bad_json = json.dumps({
    "answer" : "Some answer.",
    "sources" : ["doc.pdf"],
    "confidence" : 1.7   # invalid
})


try:
      result = RagAnswer(**json.loads(bad_json))
except Exception as exc:
      print(f"\nValidation caught bad confidence: {exc.errors()[0]['msg']}")


empty_json = json.dumps({
    "answer" : "   ",   # whitespace only
    "sources" : ["doc.pdf"],
    "confidence" : 0.5
})
 
try:
    RagAnswer(**json.loads(empty_json))
except Exception as exc:
    print(f"Validation caught empty answer : {exc.errors()[0]['msg']}")




# PART 3 — LIST COMPREHENSIONS: CLEANING BATCHES OF CHUNKS

# List comprehensions are the Pythonic way to transform and filter lists.
# In RAG you use them constantly: cleaning text, filtering short chunks,
# deduplicating, extracting fields from a list of Documents.


raw_chunks = [
    "  hello world  ",
    "",
    "RAG is a powerful technique for grounding large language model outputs in facts",
    "  ",
    "x",
    "Embeddings encode semantic meaning into dense numerical vectors for retrieval",
    "RAG is a powerful technique for grounding large language model outputs in facts",  # duplicate
]


# Step 1: strip whitespace from each string, drop anything that is now empty
stripped = [chunk.strip() for chunk in raw_chunks if chunk.strip() ]
print(f"after strip   : {len(stripped)} chunks")

# Step 2: keep only chunks with at least 5 words (too-short chunks add noise)
MIN_WORDS = 5
valid = [chunk for chunk in raw_chunks if len(chunk.split()) >= MIN_WORDS]
print(f"after min_words filter: {len(valid)} chunks")

# Step 3: deduplicate while preserving insertion order
#   - 'seen' is a set for O(1) membership checks
#   - 'seen.add(s)' always returns None (falsy), so the 'or' short-circuits to False when the item is new, keeping it in the output

"""
elements = {}
new_array = []
for i, e in enumerate(valid):
      if e in elements:
            continue
      else:
            elements[e] = i
            new_array.append(e)
print("@" * 20)
print(new_array)
"""

seen : set = set()
unique = [s for s in valid if not (s in seen or seen.add(s))]
print(f"after dedup   : {len(unique)} chunks")
for chunk in unique:
    print(f"  - {chunk[:70]}...")


# Extracting a single field from a list of Documents
docs = [
    Document("First chunk content here with enough words to pass the validity check.", "a.pdf", "c1"),
    Document("Second chunk content here with enough words to pass the validity check.", "b.pdf", "c2"),
    Document("Third chunk content here with enough words to pass the validity check.", "c.pdf", "c3"),
]
sources = [s.source for s in docs if doc.is_valid()]
print(f"\nsources from docs: {sources}")


# Filtering document by validity
valid_docs = [doc for doc in docs if doc.is_valid()]
print(f"valid docs: {len(valid_docs)} / {len(docs)}")



# PART 4 — DICT PATTERNS

# Dicts appear everywhere in RAG: metadata, API payloads, config, results.
# Know these patterns cold.


print("\n--- Part 4: Dict patterns ---")
 
metadata = {
    "source" : "fca_handbook.pdf",
    "page" : 42,
    "section"  : "chapter_3",
    "language" : "en",
}

# Safe access: returns None (or your default) instead of raising KeyError

author = metadata.get("author")
page = metadata.get("page", 0)

print(f"author (missing key): {author}")
print(f"page (present key): {page}")

# Merging two dicts, the right-hand dict wins on duplicate keys
chunk_meta = {"chunk_index": 5, "word_count": 120}
combined = {**metadata, **chunk_meta}       # spread operator merge
print(f"merged keys: {list(combined.keys())}")


# Iterating key-value pairs
print("metadata fields:")
for key, value in metadata.items():
    print(f" {key:10s} = {value}")

word_lengths = {word: len(word) for word in ["RAG", "embedding", "retrieval"]}
print(f"word lengths: {word_lengths}")