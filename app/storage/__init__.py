"""
Package Storage — Base documentaire (Livrable 4)
==================================================

Composants développés par l'étudiante autour des bibliothèques
sentence-transformers et ChromaDB :

    embedding_generator.py   → vectorisation (SentenceTransformer all-MiniLM-L6-v2)
    chroma_manager.py        → unique point d'accès à ChromaDB (moteur de stockage)
    base_export.py           → export zip de la base pour le binôme

Règles d'encapsulation :
    - `import chromadb`            autorisé UNIQUEMENT dans chroma_manager.py
    - `import sentence_transformers` autorisé UNIQUEMENT dans embedding_generator.py

Ce sous-module est purement additif : aucun composant existant ne le référence.
"""
