# LESSON 3: File I/O & Document Loading

# WHY THIS MATTERS:
#   Before any embedding or retrieval, you need to get text off disk and into
#   Python objects. UK enterprise RAG ingests PDFs (FCA handbooks, NICE
#   guidelines, legal contracts) as well as plain text and JSON.


#   Bad loading = corrupted context = wrong answers = real business risk.


# Lesson Content:
#   1. pathlib.Path: the modern, cross-platform way to handle file paths
#   2. Loading plain text files safely with correct encoding
#   3. Chunking text by words with overlap
#   4. Persisting chunk metadata to JSON and reloading it
#   5. Directory scanning — loading all documents from a folder


from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import fitz
import tempfile
import json
import os

# PART 0 — THE DOCUMENT DATACLASS

@dataclass
class Document:
    content : str
    source : str
    chunk_id : int
    page_num : Optional[int] = None
    metadata : dict = field(default_factory=dict)
    embedding : list[int] = field(default=list)

    @property
    def word_count(self)-> int:
        return len(self.content.split())

    def is_valid(self)->bool:
        return len(self.content) >= 10


# PART 1 — pathlib.Path: MODERN FILE PATH HANDLING

# Never build paths with string concatenation ("folder" + "/" + "file.txt").
# pathlib.Path is:
#   - Cross-platform: works on Windows (backslash) and Linux/Mac (forward slash)
#   - Composable: use / operator to join path segments
#   - Rich: .name, .stem, .suffix, .parent, .exists(), .glob() built in

# Build paths with the / operator — no string concatenation needed
project_root = Path(".")
data_dir = project_root / "data" / "raw"
output_dir = project_root / "data" / "processed"

print(f"project_root : {project_root.resolve()}")
print(f"data_dir : {data_dir}")
print(f"output_dir : {output_dir}")


# Useful Path attributes
example_path = Path("data/raw/fca_handbook_2024.pdf")
print(f"\nexample path  : {example_path}")
print(f".name : {example_path.name}")       # fca_handbook_2024.pdf
print(f".stem  : {example_path.stem}")       # fca_handbook_2024
print(f".suffix  : {example_path.suffix}")     # .pdf
print(f".parent  : {example_path.parent}")     # data/raw
print(f".exists() : {example_path.exists()}")   # True/False


# Creating directories safely — parents=True means it creates all intermediate
# folders; exist_ok=True means it does not raise if the folder already exists
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\nCreated dir : {output_dir}  (exists: {output_dir.exists()})")


# PART 2 — LOADING PLAIN TEXT FILES

# Always specify encoding="utf-8" explicitly.
# Without it, Python uses the system default, which differs between Windows
# (cp1252) and Linux (utf-8), causing silent corruption on non-ASCII text.

def load_text_file(path: Path)-> Optional[Document]:

    # Verify that path exists
    if not path.exists():
        print(f"WARNING: {path} does not exist — skipping")
        return None

    # Open the file 
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    # Verify that the file is not empty
    if not content:
        print(f"WARNING: {path} is empty — skipping")
        return None

    return Document(
        content= content,
        source= path.name,
        chunk_id=f"{path.stem}_full",
        metadata= {
            "file_type" : path.suffix.lstrip("."),
            "char_count": len(content),
            "path" : str(path),
        }

    )

# Create a small sample text file to demonstrate loading
sample_text = Path("data/raw/sample_rag_intro.txt")
sample_text.parent.mkdir(parents=True, exist_ok=True)
sample_text.write_text(
    "Retrieval-Augmented Generation (RAG) is an AI architecture that combines "
    "a retrieval system with a large language model. Instead of relying solely "
    "on knowledge baked into the model's weights during training, RAG retrieves "
    "relevant passages from an external document store at query time and passes "
    "them to the model as additional context. This grounds the model's outputs "
    "in authoritative, up-to-date information and dramatically reduces hallucination.",
    encoding="utf-8"
)

# Loading txt
doc = load_text_file(sample_text)
if doc:
    print(f"Loaded: {doc.source}")
    print(f"Words : {doc.word_count}")
    print(f"Valid : {doc.is_valid()}")
    print(f"Preview: {doc.content[:80]}...")



# PART 3 — CHUNKING TEXT WITH OVERLAP



# Why overlap?
#   Chunking splits sentences. A sentence that starts near the end of chunk N
#   is cut in half, the second half is in chunk N+1. Neither chunk has the
#   full sentence. Overlap repeats the last `overlap` words of chunk N at the
#   start of chunk N+1, so context is preserved across the boundary.


# chunk_size=300 words ≈ 400 tokens — a safe default for most models.
# overlap=30 words ≈ 10% of chunk_size — standard practice.


def chunk_document(doc:Document, chunk_size: int = 300, overlap : int = 30) -> list[Document]:
    # verify that the size is bigger that the overlap
    if chunk_size <= overlap:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    words = doc.content.split()
    chunks = []
    i = 0
    ci = 0

    while i < len(words):
        window = words[i: i + chunk_size]
        content = " ".join(window)

        if len(window) >= 10:  
            chunks.append(Document(
                content  = content,
                source   = doc.source,
                chunk_id = f"{doc.chunk_id}_c{ci:03d}",
                page_num = doc.page_num,
                metadata = {
                    **doc.metadata,
                    "chunk_index" : ci,
                    "start_word" : i,
                    "end_word" : i + len(window),
                    "chunk_size" : chunk_size,
                    "overlap" : overlap,
                }
            ))


        # this function will throw away the rest of words if the len of the last 
        # chunk is less that the minimum (i should find a solution to it)
        # maybe the last words should be added to a part of the latest chunk in the chunks array
    

        ci = i
        i  += (chunk_size - overlap)
    return chunks


# Demonstrate on our sample document
if doc:
    # Use small chunk_size so we get multiple chunks from a short demo text
    chunks = chunk_document(doc, chunk_size=30, overlap=5)
    print(f"Source words : {doc.word_count}")
    print(f"Chunks created: {len(chunks)}")
 
    for chunk in chunks:
        start = chunk.metadata["start_word"]
        end = chunk.metadata["end_word"]
        print(f"{chunk.chunk_id}: words {start}–{end} ({chunk.word_count} words)")
 
    # Verify overlap: last 5 words of chunk[0] == first 5 words of chunk[1]
    if len(chunks) >= 2:
        last_words_of_c0 = chunks[0].content.split()[-5:]
        first_words_of_c1 = chunks[1].content.split()[:5]
        print(f"\nOverlap check:")
        print(f"End of chunk 0 : {last_words_of_c0}")
        print(f"Start of chunk 1 : {first_words_of_c1}")
        print(f"Overlap correct : {last_words_of_c0 == first_words_of_c1}")


# PART 4 — PERSISTING CHUNK METADATA TO JSON

# After chunking, you save metadata to JSON and embeddings to the vector DB.
# They are stored separately, JSON is human-readable and version-controllable;
# the vector DB handles high-dimensional float arrays efficiently.


def save_chunks(chunks : list[Document], path : Path)-> None:
    # asdict() converts a dataclass to a plain dict recursively

    # convert the list of dataclass into a list of dict 
    # "calculating the embedding"
    data = []
    for chunk in chunks:
        d = asdict(chunk)
        d["embedding"] = []
        data.append(d)

    # create the directory 
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store the data in a json file
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {path}")


def load_chunks(path: Path) -> list[Document]:
    # verify the path
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    # open the file
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # return a list of chunks in a Document objects 
    return [Document(**d) for d in data]


if doc and chunks:
    chunks_path = Path("data/processed/sample_chunks.json")
    save_chunks(chunks, chunks_path)
 
    reloaded = load_chunks(chunks_path)
    print(f"Reloaded {len(reloaded)} chunks from {chunks_path}")
    print(f"First chunk_id: {reloaded[0].chunk_id}")
    print(f"Round-trip OK : {reloaded[0].content == chunks[0].content}")


    
# PART 5 — DIRECTORY SCANNING


# Create a few dummy text files to demonstrate scanning
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
 
for i in range(1, 4):
    (raw_dir / f"document_{i:02d}.txt").write_text(
        f"This is the content of document number {i}. "
        "It contains enough words to be a valid chunk for our pipeline. "
        "RAG systems process many documents in parallel.",
        encoding="utf-8"
    )

# we need to load all text files from a library
def load_directory(directory: Path, extensions: list[str] = None)->list[Document]:
    if extensions is None:
        extensions = [".txt"]

    all_docs : list[Document] = []

    if not directory.exists():
        raise ValueError("WARNING path doesn't exits")

    # rglob("*") finds all files in all subdirectories
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in extensions:
            continue
        if not file_path.is_file():
            continue

        loaded = load_text_file(file_path)

        if loaded:
            all_docs.append(loaded)

    return all_docs




all_docs = load_directory(raw_dir, extensions=[".txt"])
print(f"Found {len(all_docs)} text files in {raw_dir}")

for d in all_docs:
    print(f"{d.source:35s} | {d.word_count} words | valid: {d.is_valid()}")


# PART 6 — THE PyMuPDF PATTERN (reference — requires pip install pymupdf)


# This is the standard pattern for PDF loading used in production.
 

def load_pdf(path: Path) -> list[Document]:
    docs: list[Document] = []
    with fitz.open(path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue  # skip blank / image-only pages
            docs.append(Document(
                content  = text,
                source   = path.name,
                chunk_id = f"{path.stem}_page_{page_num:03d}",
                page_num = page_num,
                metadata = {"char_count": len(text)},
            ))
    return docs