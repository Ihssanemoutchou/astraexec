# Livrable 4 — Base Documentaire (documentation de l'implémentation)

> Document de livraison — AstraExec · Projet de fin d'études · EMSI
> Auteur : Ihssane MOUTCHOU
> Statut : **implémenté et validé** — 186 tests verts
> `docs/architecture_base_documentaire.md` reste la référence de conception (décisions
> d'architecture) ; ce document décrit l'**état implémenté** (chemins effectifs, API finales).

---

## 1. Résumé

Le Livrable 4 ajoute la **gestion de la base documentaire** : génération d'une base
vectorielle persistante (ChromaDB) à partir du corpus, et **livraison au binôme** sous
forme d'archive ZIP brute.

Conformément au CDC et à la remarque de l'encadrante (« les outils du CDC sont des
inspirations : développez vos propres outils »), **ChromaDB est uniquement le moteur de
stockage**. Tous les composants autour sont développés par l'étudiante :

| Composant | Fichier | Rôle |
|---|---|---|
| `EmbeddingGenerator` | `app/storage/embedding_generator.py` | vectorisation (sentence-transformers) |
| `ChromaManager` | `app/storage/chroma_manager.py` | **seul accès à ChromaDB** du projet |
| `BaseExporter` | `app/storage/base_export.py` | zip brut de la base |
| `demo_database.py` | racine | orchestration complète + livraison |

---

## 2. Architecture de `app/storage/`

```
app/api/data/  (.txt + .pdf)
      │
      ▼
DocumentManager ──────────────── (L1/L3, inchangé)
      │
      ▼
SmartSeg.process_text() ──────── (inchangé)
      │
      ▼
   chunks: List[Dict]  {chunk_id, source, length, word_count, content}
      │
      ▼
EmbeddingGenerator (sentence-transformers all-MiniLM-L6-v2)   [MON outil]
      │  embed_texts(contents)  →  np.ndarray (N × 384, float32)
      ▼
ChromaManager (persistent ChromaDB)                            [seul import chromadb]
      │  build(chunks, embeddings)
      ▼
storage/chroma/  (PersistentClient : sqlite + index HNSW)
      │
      ▼
BaseExporter.export()  →  exports/base_documentaire_v1.zip  →  envoyé au binôme
```

**Règles d'encapsulation (vérifiées à la revue)**
- `import chromadb` → **uniquement** dans `app/storage/chroma_manager.py` **dans le code
  applicatif `app/`** (les tests l'importent aussi, uniquement pour construire le client
  éphémère de `tests/test_chroma_manager.py`).
- `import sentence_transformers` → **uniquement** dans `app/storage/embedding_generator.py`
  (`tests/test_embedding_generator.py` charge le vrai modèle via le composant, pas d'import direct).
- `app/storage/` n'importe rien des autres sous-modules ; aucun composant existant ne
  dépend de `app/storage/` → **sous-module 100 % additif, zéro régression**.

---

## 3. Composants

### 3.1 `EmbeddingGenerator` — mon outil de vectorisation

Encapsule `SentenceTransformer("all-MiniLM-L6-v2")` (384 dimensions).

- `embed_text(text) -> np.ndarray (384, float32)` — encode un texte unique.
- `embed_texts(texts) -> np.ndarray (N, 384, float32)` — encode un lot (indexation et requêtes).
- `info() -> Dict` — état du composant.
- Attributs : `model_name = "all-MiniLM-L6-v2"`, `dimension = 384`.
- Import **différé** de `sentence_transformers` et chargement **paresseux** du modèle
  (~90 Mo, premier appel) : l'API et les démos L2/L3 démarrent sans coût.
- `normalize_embeddings=True` → similarité cosine prête à l'emploi.
- **Déterministe** : deux encodages du même texte sont identiques (contrat binôme).
- Erreurs explicites : `ValueError` sur entrées invalides, `ImportError` actionnable.

### 3.2 `ChromaManager` — le seul point d'accès à ChromaDB

Configure `chromadb.PersistentClient` sur `storage/chroma/` avec télémétrie désactivée.

- Collection : `astra_docs`, espace **cosine** (`{"hnsw:space": "cosine"}`).
- `build(chunks, embeddings)` — construction initiale **idempotente** (supprime puis recrée).
- `add_chunks(chunks, embeddings)` — ingestion **incrémentale** via **upsert**
  (id stable `source::chunk_id` → pas de doublon à la ré-indexation).
- `search(embedding, top_k)` — similarité **uniquement par vecteur**
  (`query_embeddings`, jamais `query_texts`) → retourne les chunks du projet
  (source, chunk_id, length, word_count, content) + `distance`, du plus proche au plus loin.
- `count()` · `delete_collection()` · `info()` · `close()`.
- `client_factory` **injectable** (défaut `PersistentClient`) → tests avec client éphémère.
- Métadonnées stockées : `{source, chunk_id, length, word_count}`.

### 3.3 `BaseExporter` — archive brute

- `export(source_dir="storage/chroma", output_zip="exports/base_documentaire_v1.zip")`
  → ZIP de la base, préfixe d'archive **`storage/chroma/`**.
- `unzip(zip_path, target_dir=".")` → restaure et **valide la présence de
  `storage/chroma/chroma.sqlite3`** après extraction.
- **Aucun format propriétaire** : ni `manifest.json`, ni `embeddings.npy`, ni `chunks.jsonl`.
  L'archive est une copie brute du dossier Chroma (sqlite + segments HNSW).
- Garde anti-traversal : seuls les membres sous `storage/chroma/` sans segment `..`
  sont extraits.

### 3.4 `demo_database.py` — orchestration (racine)

```
DocumentManager → EmbeddingGenerator → ChromaManager.build() → close() → BaseExporter.export()
```

- `try/finally` : `close()` garanti même si `build()` échoue, et exécuté **avant** la
  compression (verrous de fichiers Windows).
- Garde corpus vide (`sys.exit(1)`) avec message clair.
- Style identique aux démos existantes (couleurs ANSI, UTF-8 console Windows).

---

## 4. Usage

### 4.1 Générer la base (côté étudiante)

```bash
python demo_database.py
```

Sortie (corpus actuel) :

```
415 chunks générés
415 vecteurs × 384 dims (56.4s)
Chunks indexés : 415          (collection astra_docs, cosine)
Archive : exports\base_documentaire_v1.zip
```

La base est ensuite livrée au binôme : **`exports/base_documentaire_v1.zip`**.

### 4.2 Charger la base (côté binôme)

1. Dézipper `base_documentaire_v1.zip` à la racine du projet du binôme
   → obtient directement le dossier `storage/chroma/`.
2. Ouvrir la base :

```python
import chromadb

client = chromadb.PersistentClient(path="storage/chroma")
collection = client.get_collection("astra_docs")
```

3. Interroger avec le **même modèle d'embedding** (recommandé) :

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
results = collection.query(
    query_embeddings=[model.encode("machine learning")],
    n_results=5,
)
for doc, meta, dist in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(round(dist, 4), meta["source"], meta["chunk_id"], doc[:80])
```

Les vecteurs produits par le binôme étant **identiques** à ceux de la base (même
bibliothèque, même modèle), les distances et le classement sont exactement ceux attendus.

---

## 5. Contrat binôme (compatibilité)

| Élément | Valeur | Raison |
|---|---|---|
| Version ChromaDB | **`chromadb==1.5.9`** | le dossier persistant (sqlite + HNSW) est lié à la version majeure |
| Modèle d'embedding | **`all-MiniLM-L6-v2`** (384 dims) | mêmes vecteurs → mêmes résultats |
| Bibliothèque d'embedding | `sentence-transformers` (5.6.0 vérifiée) | voie recommandée : écart ONNX/PyTorch quasi nul mais pas bit-à-bit |
| Collection | `astra_docs` (espace **cosine**) | le binôme n'a rien à configurer |
| Archive | zip brut, préfixe `storage/chroma/` | dézipper → `PersistentClient(path="storage/chroma")` |

> **Important** : la version `chromadb==1.5.9` doit être **communiquée au binôme** (par
> mail / rapport). L'archive brute ne contient aucun fichier explicatif — c'est voulu :
> aucun format propriétaire.

---

## 6. Fichiers du Livrable 4

**Créés (additifs)**

| Fichier | Rôle |
|---|---|
| `app/storage/__init__.py` | package Storage |
| `app/storage/embedding_generator.py` | vectorisation (sentence-transformers) |
| `app/storage/chroma_manager.py` | seul import chromadb |
| `app/storage/base_export.py` | zip/unzip de la base |
| `demo_database.py` | orchestration + livraison |
| `docs/livrable4.md` | ce document |
| `tests/test_embedding_generator.py` | 10 tests |
| `tests/test_chroma_manager.py` | 19 tests (client éphémère) |
| `tests/test_base_export.py` | 12 tests (round-trip zip) |

**Modifiés (configuration/documentation uniquement)**

| Fichier | Changement |
|---|---|
| `requirements.txt` | + `chromadb==1.5.9` + `sentence-transformers` |
| `.gitignore` | + `/storage/` + `/exports/` (base et archive jamais commitées) |
| `README.md` | sous-module Storage + technologies + lien vers ce document |

**Jamais modifiés** : `app/retrieval/*`, `app/api/*`, `app/executor/*`, `app/registry/*`,
`app/guardrails/*`, `app/evaluation/*`, `app/telemetry/*`, `app/utils/*`, `app/schemas/*`,
`demo.py`, `demo_evaluation.py`, tous les `tests/test_*.py` existants.

---

## 7. Validation

- **186 tests verts** : 145 (L1/L2/L3) + 41 nouveaux (10 + 19 + 12) — zéro régression.
- Pipeline réel exécuté : 415 chunks → 415 vecteurs → 415 indexés → archive générée.
- Encapsulation vérifiée : aucun `import chromadb` dans le code applicatif `app/` hors de
  `app/storage/chroma_manager.py` (les tests l'importent uniquement pour le client éphémère).
- Revues de code systématiques à chaque étape — plusieurs bugs réels attrapés et corrigés
  (notamment une collision de constantes provoquant une double imbrication au dézip).

---

## 8. Limites et points d'attention

1. **Version ChromaDB** : la base persistante est liée à `chromadb==1.5.9` — communiquer
   cette contrainte au binôme (voir §5).
2. **Modèle d'embedding** : le binôme doit disposer de `all-MiniLM-L6-v2` (90 Mo,
   téléchargement unique ou partage du cache).
3. **Verrouillage Windows** : `close()` avant tout zip (géré par `demo_database.py`).
4. **Imports différés** : chromadb et sentence-transformers sont chargés à la demande
   pour préserver le démarrage de l'API et des démos existantes.
