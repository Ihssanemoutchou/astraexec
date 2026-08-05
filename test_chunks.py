from app.retrieval.document_manager import DocumentManager

dm = DocumentManager("app/api/data")

chunks = dm.load_documents()

for chunk in chunks:
    print("=" * 80)
    print("Chunk ID :", chunk["chunk_id"])
    print("Source   :", chunk.get("source"))
    print("Length   :", chunk.get("length"))
    print()
    print(chunk["content"])
    print()