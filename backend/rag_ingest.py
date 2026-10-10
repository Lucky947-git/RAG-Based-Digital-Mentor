"""Run with: python -m backend.rag_ingest"""
from backend.rag import ingest_pdfs


if __name__ == "__main__":
    result = ingest_pdfs()
    print(f"Indexed {result['indexed_chunks']} chunks from {len(result['files'])} approved PDF(s).")
    if not result["files"]:
        print("No matching PDFs found. Add the approved book PDFs to the configured PDF_PATH.")
