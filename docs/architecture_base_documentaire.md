# Livrable 4 — Architecture de la Base Documentaire (version finale)

> Document d'architecture — AstraExec · Projet de fin d'études · EMSI
> Statut : **à valider avant toute implémentation**
> Auteur : Ihssane MOUTCHOU
> Note : ce document intègre les décisions finales de l'étudiante (v3).
> Il remplace les versions précédentes (hashing maison / format portable propriétaire abandonnés).
> ⚠️ **Référence de conception** : l'état implémenté (API finales, chemins effectifs,
> `exports/` à la racine) est décrit dans [`livrable4.md`](livrable4.md).

---

## 1. Contexte et périmètre

### 1.1 État de l'existant (validé, non modifié)

| Livrable | Contenu | Statut |
|---|---|---|
| L1 | SmartSeg, LexiRank, FusionSearch, QueryProfiler, EvidenceRank, DocumentManager, Executor, Registry | Validé |
| L2 | Campagne d'évaluation : Recall@5 = 0.8917, MRR = 1.0000, 17 chunks, 8 requêtes | Validé |
| L3 | EthicalFilter, support PDF (ReaderFactory, PDFReader, TXTReader), persistance pickle FusionSearch | Validé — **145 tests verts** |

### 1.2 Décisions finales (validées par l'étudiante)

1. **EmbeddingGenerator = MON outil, adossé à `sentence-transformers`** :
   `SentenceTransformer("all-MiniLM-L6-v2")` (384 dimensions). Pas de hashing maison.
   → embeddings **compatibles et reproductibles** par le binôme.
2. **Export = zip simple de `storage/chroma/`** : la base Chroma est déjà persistante.
   Pas de `manifest.json`, pas de `embeddings.npy`, pas de `chunks.jsonl`, aucun format propriétaire.
3. **Trois fichiers métier dans `app/storage/`** : `embedding_generator.py`,
   `chroma_manager.py`, `base_export.py`. **Seul `chroma_manager.py` importe chromadb.**
4. Simplicité avant tout : **aucune couche académique superflue** (pas de Protocol files séparés).

### 1.3 Contraintes (CDC + encadrante)

- « Les outils du CDC sont des inspirations : développez vos propres outils. »
  → ChromaDB n'est **que le moteur de stockage** ; toute la logique autour est développée par l'étudiante.
- « Générez une base documentaire et envoyez-la au binôme. » → le binôme doit
  pouvoir la charger **directement**, sans difficulté de compatibilité.
- Aucune régression L1/L2/L3.

---

## 2. Pipeline cible

```
app/api/data/  (.txt + .pdf)
      │
      ▼
DocumentManager ──────────────── (inchangé, résilience par lot)
      │
      ▼
SmartSeg.process_text() ──────── (inchangé)
      │
      ▼
   chunks: List[Dict]  {chunk_id, source, length, word_count, content}
      │
      ▼
EmbeddingGenerator (sentence-transformers all-MiniLM-L6-v2)   [MON outil]
      │  embed_texts(chunks)  →  np.ndarray (N × 384, float32)
      ▼
ChromaManager (persistent ChromaDB)                            [seul import chromadb]
      │  build(chunks, embeddings)
      ▼
storage/chroma/  (PersistentClient : sqlite + index HNSW)
      │
      ▼
BaseExporter.zip()  →  storage/exports/base_documentaire_vX.zip  →  envoyé au binôme
```

---

## 3. Graphe des dépendances

```
demo_database.py ──► DocumentManager ──► ReaderFactory, SmartSeg        [L3, intacts]
      │           ──► EmbeddingGenerator ──► sentence-transformers (numpy)
      │           ──► ChromaManager      ──► chromadb          (numpy)
      │           ──► BaseExporter       ──► zipfile, shutil   (numpy)
ChromaManager ──► chromadb        (SEUL import chromadb du projet)
EmbeddingGenerator ──► sentence_transformers, numpy
```

**Règles non négociables**
- **`import chromadb` autorisé uniquement dans `app/storage/chroma_manager.py`** (vérifié par grep à la revue).
- **`import sentence_transformers` autorisé uniquement dans `app/storage/embedding_generator.py`**.
- `app/storage/` n'importe rien de `app/retrieval`, `app/api`, `app/executor`, `app/evaluation`, `app/guardrails`.
- **Aucun** composant existant ne dépend de `app/storage/` : sous-module purement additif.
- Pas de dépendance circulaire possible (graphe orienté acyclique).

---

## 4. Responsabilités des composants

### 4.1 `EmbeddingGenerator` — MON outil de vectorisation

- Encapsule `SentenceTransformer("all-MiniLM-L6-v2")` (384 dims).
- API : `embed_text(text) -> np.ndarray` · `embed_texts(texts) -> np.ndarray (N×384 float32)` · `dimension` · `model_name`.
- Import différé de `sentence_transformers` (démarrage rapide de l'API et des démos L2/L3).
- **Ne fait pas** : lecture, segmentation, persistance. C'est un transformateur pur.
- **Déjà opérationnel hors-ligne** : le modèle est en cache local (91,6 Mo vérifiés).

### 4.2 `ChromaManager` — le seul point d'accès à ChromaDB

- Configure le `PersistentClient` sur `storage/chroma/`, télémétrie désactivée.
- Collection `get_or_create("astra_docs", metadata={"hnsw:space": "cosine"})`.
- `build(chunks, embeddings)` / `add_chunks(...)` : ids `"{source}::{chunk_id}"`,
  `document=content`, métadonnées = {source, chunk_id, length, word_count, page_range (None)}.
- `search(embedding, top_k)` → liste `{chunk, distance}` où `chunk` est le **dict chunk complet du projet** (content, source, chunk_id, length, word_count).
- `count()`, `get_chunk(id)`, `delete_collection()`, `info()`, `close()`.
- `client_factory` injectable (défaut `PersistentClient`) → tests avec client éphémère.
- **Aucune logique métier** : il ne classe pas, ne filtre pas, ne transforme pas le texte.

### 4.3 `BaseExporter` — emballage du dossier persistant

- `zip(storage/chroma) → storage/exports/base_documentaire_vX.zip`.
- `unzip(archive, destination)` avec validation (présence de `chroma.sqlite3`).
- Ne définit **aucun format propriétaire** : c'est le dossier Chroma, tel quel.
- Exige un client **fermé** (`close()` avant zip — verrouillage de fichiers sous Windows).

### 4.4 `demo_database.py` (racine) — composition

- Orchestre : DocumentManager → EmbeddingGenerator → ChromaManager → `close()` → BaseExporter.
- C'est le script de la démonstration « générer la base et la livrer au binôme ».

---

## 5. Où sont utilisés ChromaDB et SentenceTransformer (exactement)

| Bibliothèque | Fichier unique | Usage |
|---|---|---|
| `chromadb` | `app/storage/chroma_manager.py` | Client persistant, collection, insertion, recherche |
| `sentence_transformers` | `app/storage/embedding_generator.py` | `SentenceTransformer("all-MiniLM-L6-v2")` |

Rien d'autre dans le projet ne référence ces bibliothèques.

---

## 6. Persistance et stockage

```
storage/
├── chroma/                      ← PersistentClient (généré, gitignoré)
│   ├── chroma.sqlite3           ← catalogue, documents, métadonnées, ids
│   └── <uuid>/data_level0/      ← index HNSW (embeddings binaires)
└── exports/
    └── base_documentaire_v1.zip ← archive livrée au binôme
```

- Espace de distance **cosine** (stocké dans la configuration de la collection → le binôme n'a rien à configurer).
- Constantes partagées (documentées dans le code et le rapport) : `collection_name = "astra_docs"`,
  `model_name = "all-MiniLM-L6-v2"` (384 dims).

---

## 7. Comment le binôme charge la base (sans difficulté)

1. Dézipper `base_documentaire_v1.zip` (un dossier Chroma complet — voir §12.1 : version `chromadb==1.5.9` requise).
2. `import chromadb ; client = chromadb.PersistentClient(path="<dossier dézippé>")`.
3. `collection = client.get_collection("astra_docs")`.
4. Requête (2 options) :
   - **recommandé** : `collection.query(query_embeddings=[SentenceTransformer("all-MiniLM-L6-v2").encode(q)])`
     → vecteurs **identiques** à ceux de la base (même modèle, même bibliothèque) ;
   - option zéro-dépendance : `collection.query(query_texts=[q])` via l'embedding
     function par défaut de ChromaDB (même modèle sous-jacent en ONNX — écart
     numérique négligeable, à signaler dans le rapport).
5. Résultats : `documents` (texte des chunks) + `metadatas` (source, chunk_id, length, word_count) + `distances`.

Aucun autre prérequis : l'index HNSW et le sqlite sont **autonomes** dans le dossier.

---

## 8. Vérification CDC + remarques de l'encadrante

| Exigence | Statut | Justification |
|---|---|---|
| « Développez vos propres outils » | ✔ | EmbeddingGenerator (wrapper + logique maison), ChromaManager, BaseExporter, demo_database sont développés par l'étudiante. Aucune copie de ChromaDB. |
| « ChromaDB = moteur de stockage » | ✔ | ChromaManager ne contient aucune logique métier : il persiste et interroge par similarité. |
| « Générez une base et envoyez-la au binôme » | ✔ | `demo_database.py` génère la base ; `BaseExporter` produit l'archive ; le binôme la charge avec 3 lignes (section 7). |
| Compatibilité binôme | ✔ | Même collection, même modèle d'embedding, dossier autonome — aucun format propriétaire. |
| Aucune régression | ✔ | 145 tests verts ; pipeline L2/L3 intacts ; imports différés (aucun coût de démarrage pour l'existant). |

---

## 9. Vérification SOLID / Clean Architecture (sans sur-ingénierie)

| Principe | Application |
|---|---|
| SRP | 3 raisons de changement disjointes : vectorisation (embedding_generator), accès ChromaDB (chroma_manager), emballage (base_export). |
| DIP | Aucun composant ne dépend d'un autre : `demo_database.py` **injecte** les données entre EmbeddingGenerator et ChromaManager ; ChromaManager ne connaît pas EmbeddingGenerator (il reçoit des vecteurs). |
| OCP | Changer de modèle = changer la constante `model_name` ; changer de stockage = remplacer chroma_manager (la forme des données d'entrée/sortie ne change pas). |
| Infra/Domaine | `app/storage/` = infrastructure ; le domaine (chunks) ne connaît ni chromadb ni sentence-transformers. |
| Testabilité | `client_factory` éphémère ; EmbeddingGenerator testable sur le modèle en cache ; BaseExporter testable sur dossier temporaire. |
| Extensibilité | `page_range` réservé (None) pour la future capture de pages PDF ; `client_factory` prêt. |
| Rétrocompatibilité | Imports différés ; zéro fichier existant modifié ; schéma de chunk conservé dans les métadonnées. |

> **Note honnête (DIP)** : aucun fichier `interfaces.py`/Protocol n'est créé — décision volontaire de
> simplicité (un seul backend). Si un second stockage vectoriel devenait nécessaire, un Protocol
> serait introduit à ce moment-là (YAGNI). La dépendance est déjà inversée structurellement.

---

## 10. Fichiers

**Créer (10)**

| Fichier | Rôle |
|---|---|
| `app/storage/__init__.py` | package Storage |
| `app/storage/embedding_generator.py` | wrapper sentence-transformers (all-MiniLM-L6-v2) |
| `app/storage/chroma_manager.py` | **seul import chromadb** du projet |
| `app/storage/base_export.py` | zip/unzip de `storage/chroma/` |
| `demo_database.py` | orchestration complète + livraison de l'archive |
| `docs/livrable4.md` | documentation du livrable |
| `tests/test_embedding_generator.py` | déterminisme, dimension 384, type float32 |
| `tests/test_chroma_manager.py` | build/search/count/delete, round-trip métadonnées |
| `tests/test_base_export.py` | zip → unzip → base réutilisable, validation sqlite |
| `tests/test_database_integration.py` | pipeline réel → requête retrouve son chunk |

**Modifier (3 — configuration/documentation uniquement)**

| Fichier | Changement |
|---|---|
| `requirements.txt` | + `chromadb` + `sentence-transformers` |
| `.gitignore` | + `storage/` + `index/` + `logs/` |
| `README.md` | sous-module Storage + technologies |

**Ne jamais toucher** : tout `app/retrieval/*`, `app/guardrails/*`, `app/evaluation/*`,
`app/executor/*`, `app/registry/*`, `app/api/*`, `app/telemetry/*`, `app/schemas/*`,
`app/utils/*`, `app/streamlit_app.py`, `demo.py`, `demo_evaluation.py`, `test_chunks.py`,
tous les `tests/test_*.py` existants.

---

## 11. Tests et non-régression

- Nouveaux tests : 4 fichiers (section 10), client ChromaDB **éphémère** en mémoire pour les tests.
- Non-régression à chaque étape : `pytest tests/ -q` → **145 + nouveaux, tous verts** ;
  `demo_evaluation.py` sur corpus TXT seul → **17 chunks, Recall@5 = 0.8917, MRR = 1.0000** ;
  grep d'encapsulation : aucun `import chromadb` hors de `app/storage/chroma_manager.py`.

---

## 12. Limites et points d'attention (signalés, pas masqués)

1. **Version de ChromaDB** : le dossier persistant (sqlite + HNSW) est lié à la version majeure
   de chromadb (1.5.9 ici). Mitigation : l'archive reste **strictement le dossier `storage/chroma/`
   tel quel** (aucun fichier ajouté) ; la contrainte `chromadb==1.5.9` est documentée dans
   `docs/livrable4.md` et le rapport de stage, et doit être **communiquée au binôme**
   (l'archive brute ne contient aucun fichier explicatif).
2. **Modèle d'embedding** : le binôme doit disposer du modèle `all-MiniLM-L6-v2` (90 Mo,
   téléchargement unique ou partage du cache). Le modèle est déjà en cache local ici.
3. **Écart ONNX/PyTorch** : si le binôme utilise l'embedding function par défaut de ChromaDB
   (ONNX), les vecteurs sont quasi-identiques mais pas bit-à-bit. Voie recommandée : même
   bibliothèque (sentence-transformers) pour requêtes et base.
4. **Verrouillage Windows** : fermer le client (`close()`) avant `BaseExporter.zip()`.
5. **Imports lents** : chromadb et sentence-transformers importés en différé dans leurs
   composants pour préserver le démarrage de l'API et des démos.

---

## 13. Plan d'implémentation (après validation)

| Étape | Contenu | Validation |
|---|---|---|
| 4.1 | `embedding_generator.py` + tests | tests + revue (modèle en cache → hors-ligne) |
| 4.2 | `chroma_manager.py` + tests (client éphémère) | tests + revue |
| 4.3 | `base_export.py` + tests (round-trip zip) | tests + revue |
| 4.4 | `demo_database.py` + test d'intégration + génération réelle de l'archive | archive générée + revue |
| 4.5 | `livrable4.md` + requirements/.gitignore/README + validation finale | 145+ tests + 0.8917/1.0000 + grep encapsulation |

---

## 14. Conclusion

L'architecture finale est **volontairement simple** : trois composants dans `app/storage/`
(EmbeddingGenerator adossé à sentence-transformers, ChromaManager unique point d'accès à
ChromaDB, BaseExporter zip), orchestrés par `demo_database.py`. ChromaDB reste un simple
moteur de stockage, tous les outils sont développés par l'étudiante, et le binôme charge la
base en trois lignes sans format propriétaire ni couche superflue. Aucune régression :
145 tests verts, Livrables 1-3 intacts.
