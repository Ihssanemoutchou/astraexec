# Contrat d'Intégration — Module Action ↔ Module Reasoning

> Document d'ingénierie — AstraExec · Projet de fin d'études · EMSI
> Version : 1.0 · Date : 2026-08-02
> Auteur : Ihssane MOUTCHOU (Module Action)
> Destinataire : Étudiant(e) du Module Reasoning (binôme)
>
> **Ce document est un CONTRAT TECHNIQUE, pas une documentation utilisateur.**
> Il décrit exactement ce que le Module Reasoning doit envoyer au Module Action,
> ce qu'il recevra en retour, et comment fusionner les deux modules sans crash.
> Tous les contrats ci-dessous proviennent de l'implémentation réelle du code
> (vérifiée par lecture du code source et tests), aucune donnée inventée.

---

## 0. Résumé exécutif

- Le Module Action est appelé via **`Executor.run(action)`** (Python) ou via
  l'**API REST FastAPI** (`POST /execute`).
- **Entrée unique** : un dictionnaire JSON `{"tool": "<nom_outil>", "parameters": {…}}`.
- **Sortie unique** : un dictionnaire JSON avec `status` (`"success"` ou `"error"`),
  toujours présent, avec `execution_time`.
- **`run()` ne lève JAMAIS d'exception** : toute erreur est convertie en objet
  `{"status": "error", …}`.
- **Un seul outil est enregistré par défaut** : `fusion_search` (recherche hybride).
- **Sécurité** : validation de structure + détection de Prompt Injection
  (seuil ≥ 2 motifs) intégrées dans le pipeline. Un filtre éthique optionnel
  (`EthicalFilter`) existe et peut être appelé séparément.
- **Base documentaire** : ChromaDB (`storage/chroma/`), collection `astra_docs`,
  espace `cosine`, modèle `all-MiniLM-L6-v2` (384 dims). Livrée via
  `exports/base_documentaire_v1.zip`. Le binôme doit utiliser **le même modèle**
  et `query_embeddings` (jamais `query_texts`).

---

## 1. Architecture Globale

### 1.1 Vue d'ensemble

```
AstraExec (Module Action)
│
├── app/api/          → API REST FastAPI (points d'entrée réseau)
├── app/executor/     → Executor (pipeline d'exécution, point d'entrée métier)
├── app/registry/     → ToolRegistry + BaseTool (contrat des outils)
├── app/guardrails/   → Validator, InjectionGuard, EthicalFilter (sécurité)
├── app/retrieval/    → DocumentManager, SmartSeg, FusionSearch, LexiRank,
│                       EvidenceRank, QueryProfiler, ReaderFactory, lecteurs
├── app/storage/      → EmbeddingGenerator, ChromaManager, BaseExporter (L4)
├── app/schemas/      → ActionInterface (dataclass du contrat d'action)
├── app/telemetry/    → Logger (journalisation)
├── app/utils/        → Helpers (utilitaires)
├── app/evaluation/   → metrics.py (Recall@K, MRR)
└── app/streamlit_app.py → interface de démonstration (non contractuelle)
```

### 1.2 Responsabilités de chaque paquet

| Paquet | Responsabilité | Dépend de |
|---|---|---|
| `app/api/` | Exposition REST : `/execute`, `/search`, `/tools`, `/health` | executor, registry, retrieval |
| `app/executor/` | Orchestration : valider → filtrer → résoudre → exécuter → journaliser | registry, guardrails, telemetry |
| `app/registry/` | Contrat des outils (`BaseTool`) + registre nom → outil | — |
| `app/guardrails/` | Validation de structure, anti-injection, filtre éthique | — |
| `app/retrieval/` | Segmentation, indexation hybride, recherche, re-ranking | numpy |
| `app/storage/` | Embeddings (sentence-transformers), ChromaDB, export zip | chromadb, sentence_transformers, numpy |
| `app/schemas/` | Dataclass `ActionInterface` (validation optionnelle côté appelant) | — |
| `app/telemetry/` | Journalisation des exécutions (`logs/astra_exec.log`) | — |
| `app/evaluation/` | Métriques Recall@K / MRR (campagnes) | — |

### 1.3 Graphe des dépendances

```
                   ┌──────────────────────────────┐
                   │  Module Reasoning (binôme)   │
                   └──────────────┬───────────────┘
                                  │ JSON {"tool", "parameters"}
                                  ▼
              ┌───────────────────────────────────────┐
              │  API REST  (app/api/main.py)          │
              │  POST /execute · GET /search          │
              └──────────────────┬────────────────────┘
                                 │ ActionRequest (Pydantic)
                                 ▼
              ┌───────────────────────────────────────┐
              │  Executor.run(action)                 │  app/executor/executor.py
              │  1. validate_action                   │
              │  2. resolve_tool                      │
              │  3. validate_schema (si parameter_schema)
              │  4. execute_tool                      │
              └──────────────────┬────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌──────────────┐        ┌────────────────┐        ┌─────────────────┐
│ ToolRegistry │        │  Validator +   │        │   Tool (outil)  │
│ registry/    │        │  InjectionGuard│        │   registry/     │
└──────────────┘        └────────────────┘        └────────┬────────┘
                                                           ▼
                                                    ┌─────────────────┐
                                                    │  Retrieval      │
                                                    │  FusionSearch → │
                                                    │  EvidenceRank   │
                                                    │  QueryProfiler  │
                                                    └─────────────────┘
```

### 1.4 Pipeline d'exécution (flux de la requête)

```
Reasoning
   │  envoie {"tool": "fusion_search", "parameters": {"query": "..."}}
   ▼
Executor.run()
   │  1. prepare_action()      → structure minimale (tool, parameters)
   │  2. Validator.validate()  → types et champs obligatoires
   │  3. InjectionGuard.inspect() → détection prompt injection (score ≥ 2 → blocage)
   │  4. resolve_tool()        → outil existe-t-il dans le registre ?
   │  5. validate_schema()     → si l'outil expose parameter_schema
   │  6. tool.execute(**parameters)  → exécution réelle
   │       FusionSearch.search(query, profile) → EvidenceRank.rerank(...)
   ▼
Réponse structurée {"status": "success"|"error", "execution_time", ...}
   ▼
Retour au Reasoning
```

**Transitions clés** :

1. **Validation structurelle** — l'action doit être un dict avec `tool` (str) et `parameters` (dict). Sinon `status="error"`.
2. **Anti-injection** — tous les paramètres sont concaténés et scannés contre 16 motifs interdits ; si ≥ 2 motifs détectés → `"Prompt Injection détectée."`.
3. **Résolution de l'outil** — le nom doit exister dans le registre ; sinon `"L'outil 'X' n'existe pas."`.
4. **Validation du schéma** — si l'outil définit `parameter_schema`, chaque paramètre est vérifié (type, obligatoire, valeurs autorisées, longueur).
5. **Exécution** — les paramètres sont passés en `**kwargs` à `execute()`.
6. **Journalisation** — succès ou erreur enregistrés dans `logs/astra_exec.log`.

---

## 2. Points d'entrée publics

### 2.1 `Executor.run(action)` — entrée métier principale

| Élément | Valeur |
|---|---|
| Emplacement | `app/executor/executor.py` |
| Signature | `run(self, action: Dict[str, Any]) -> Dict[str, Any]` |
| Entrée | `action` : dictionnaire `{"tool": str, "parameters": dict}` |
| Sortie (succès) | `{"status": "success", "tool": str, "execution_time": float, "result": <sortie outil>}` |
| Sortie (erreur) | `{"status": "error", "execution_time": float, "message": str}` |
| Exceptions | **Aucune** : tout est capturé et converti en objet d'erreur |

### 2.2 API REST (`app/api/main.py`) — entrée réseau

| Endpoint | Méthode | Entrée | Sortie |
|---|---|---|---|
| `/execute` | POST | `{"tool": str, "parameters": dict}` (body JSON) | résultat de `executor.run()` ; HTTP 200 si erreur métier, 400 en dernier recours, 422 si body invalide (Pydantic) |
| `/search` | GET | `?query=<str>` | résultat de `executor.run({"tool": "fusion_search", "parameters": {"query": query}})` |
| `/tools` | GET | — | `[{name, description}, …]` (liste des outils enregistrés) |
| `/health` | GET | — | `{"status": "healthy", "executor": "running", "tool_count": n}` |
| `/` | GET | — | `{"project": "AstraExec", "version": "2.0.0", "status": "running", "engine": "100% custom tools"}` |

### 2.3 Scripts de démonstration (non contractuels, mais utiles au binôme)

| Script | Rôle |
|---|---|
| `demo.py` | Démo automatique de tous les modules (SmartSeg, LexiRank, FusionSearch, EvidenceRank, Guardrails, Executor, Telemetry, API) |
| `demo_database.py` | Génération complète de la base documentaire : `python demo_database.py` |
| `demo_evaluation.py` | Campagne d'évaluation retrieval (Recall@K, MRR) |
| `demo_performance.py` | Benchmark des performances |
| `app/streamlit_app.py` | Interface de recherche (démo) |

---

## 3. Workflow d'exécution complet (étape par étape)

### Étape 1 — Réception de la requête

Le Reasoning envoie une action JSON, par exemple :

```json
{
  "tool": "fusion_search",
  "parameters": { "query": "Qu'est-ce que BM25 ?" }
}
```

Via API : `POST /execute` → le body est validé par le modèle Pydantic
`ActionRequest(tool: str, parameters: dict)` (erreur 422 si structure invalide).
Via code : appel direct `executor.run(action)`.

### Étape 2 — `prepare_action()` (structure minimale)

Dans `Executor.run()`, `validate_action()` appelle d'abord `prepare_action()` :
- l'action **doit être un dictionnaire** → sinon `TypeError` → `message = "Une action doit être un dictionnaire."`
- les clés **`tool`** et **`parameters`** doivent exister → sinon `ValueError` → `message = "Champ obligatoire manquant : tool"` (ou `: parameters`)

### Étape 3 — `Validator.validate()` (types)

- `tool` doit être une **chaîne** → sinon `message = "Le nom de l'outil doit être une chaîne."`
- `parameters` doit être un **dictionnaire** → sinon `message = "Les paramètres doivent être un dictionnaire."`

### Étape 4 — `InjectionGuard.inspect()` (détection d'injection)

- Concatène **toutes les valeurs** de `parameters` en une chaîne unique.
- Normalise (minuscules, espaces réduits).
- Scanne contre **16 motifs regex interdits** (ex. `ignore\s+previous`, `system\s+prompt`, `sudo`, `rm\s+-rf`, `drop\s+table`, `bypass`, …).
- `score` = nombre de motifs détectés.
- **Si `score >= 2`** → `ValueError` → `message = "Prompt Injection détectée."`
- Si sûr → retourne `{"risk_score": int, "patterns": [...], "safe": true}` (non utilisé par `run()`).

### Étape 5 — `resolve_tool()` (existence de l'outil)

- `tool_name = action["tool"]`.
- `registry.exists(tool_name)` → sinon `ValueError` → `message = "L'outil 'X' n'existe pas."`
- Retourne l'instance de l'outil.

### Étape 6 — `validate_schema()` (si l'outil définit un schéma)

- `schema = getattr(tool, "parameter_schema", None)`.
- Si non nul → `Validator.validate_schema(parameters, schema)`.
- Vérifie pour chaque paramètre : obligatoire, type, valeurs autorisées, longueur min/max.
- Erreurs possibles (voir Section 11).

### Étape 7 — `execute_tool()` (exécution)

- `tool.execute(**parameters)` → les clés de `parameters` deviennent des arguments nommés.
- L'outil retourne un dict (ex. `{"results": [...], "profile": {...}}` pour `fusion_search`).
- Si l'outil lève une exception → capturée par le `except Exception` de `run()`.

### Étape 8 — Calcul du temps et journalisation

- `execution_time = round(time.perf_counter() - start, 4)`.
- Succès : `Logger.log_success(tool.name, elapsed)` → ligne `SUCCESS | Tool=X | Time=…s` dans `logs/astra_exec.log`.
- Erreur : `Logger.log_error(str(error), elapsed)` → ligne `ERROR | Time=…s | <message>`.

### Étape 9 — Formatage de la réponse

Succès :

```json
{
  "status": "success",
  "tool": "fusion_search",
  "execution_time": 0.0234,
  "result": { "results": [...], "profile": {...} }
}
```

Erreur :

```json
{
  "status": "error",
  "execution_time": 0.0012,
  "message": "L'outil 'X' n'existe pas."
}
```

### Étape 10 — Retour au Reasoning

Le dict est sérialisé en JSON (via l'API) ou retourné directement (via code).
**Le Reasoning doit TOUJOURS vérifier `status` avant d'utiliser `result`.**

---

## 4. Pipeline de validation

### 4.1 Ordre exact des contrôles (dans `Executor.run()`)

| # | Contrôle | Mécanisme | Échec → message |
|---|---|---|---|
| 1 | Action = dictionnaire | `prepare_action` | `Une action doit être un dictionnaire.` |
| 2 | `tool` présent | `prepare_action` | `Champ obligatoire manquant : tool` |
| 3 | `parameters` présent | `prepare_action` | `Champ obligatoire manquant : parameters` |
| 4 | `tool` est une chaîne | `Validator.validate` | `Le nom de l'outil doit être une chaîne.` |
| 5 | `parameters` est un dict | `Validator.validate` | `Les paramètres doivent être un dictionnaire.` |
| 6 | Anti-injection | `InjectionGuard.inspect` | `Prompt Injection détectée.` |
| 7 | Outil existe | `resolve_tool` | `L'outil 'X' n'existe pas.` |
| 8 | Schéma des paramètres | `validate_schema` | messages dédiés (Section 11) |

### 4.2 `Validator.validate_schema(parameters, schema)`

Format du schéma (attribut `parameter_schema` d'un outil) :

```python
{
    "query": {
        "type": "string",        # string | integer | number | boolean | list | dict
        "required": True,         # bool (défaut False)
        "allowed": ["a", "b"],    # optionnel : valeurs autorisées
        "min_length": 2,          # optionnel (string)
        "max_length": 500,        # optionnel (string)
        "description": "..."      # optionnel
    }
}
```

Contrôles dans l'ordre, pour chaque paramètre du schéma :
1. **Obligatoire manquant** → `Paramètre obligatoire manquant : <nom>` (si `required=True` et absent).
2. **Type** → `Le paramètre '<nom>' doit être de type <type>, reçu : <type réel>`.
   - Carte des types : `string→str`, `integer→int`, `number→(int,float)`, `boolean→bool`, `list→list`, `dict→dict`.
3. **Valeurs autorisées** → `Le paramètre '<nom>' doit être l'une des valeurs : [...], reçu : <valeur>` (si `allowed` défini et valeur absente).
4. **Longueur min** (string) → `Le paramètre '<nom>' doit contenir au moins <n> caractère(s)`.
5. **Longueur max** (string) → `Le paramètre '<nom>' doit contenir au maximum <n> caractère(s)`.

### 4.3 Outils utilitaires du Validator (non appelés par `run()`)

| Méthode | Rôle |
|---|---|
| `validate_parameters(parameters, required=[...])` | Vérifie la présence de paramètres obligatoires |
| `validate_tool_name(name)` | Rejette un nom vide |
| `full_validation(action, required_params=None, schema=None)` | Validation complète en une passe |

### 4.4 `ActionInterface` (`app/schemas/action_interface.py`)

Dataclass optionnelle côté appelant (le pipeline n'en dépend pas, `run()` accepte un dict) :

```json
{
  "tool": "fusion_search",
  "parameters": {"query": "..."},
  "priority": "normal",      // low | normal | high
  "confidence": 1.0,          // 0.0 - 1.0
  "request_id": "",
  "metadata": {}
}
```

`validate()` : `tool` non vide, `parameters` dict, `priority` dans `[low, normal, high]`,
`confidence` dans `[0, 1]`. `to_dict()` / `from_dict()` pour conversion.

---

## 5. Pipeline de sécurité

### 5.1 InjectionGuard (intégré au pipeline — obligatoire)

- **Emplacement** : `app/guardrails/injection_guard.py`.
- **Déclencheur** : appelé automatiquement à chaque `Executor.run()` (étape 4).
- **Mécanisme** : concaténation des valeurs de `parameters` → normalisation →
  scan regex de 16 motifs interdits → score = nombre de correspondances.
- **Politique de blocage** : `score >= 2` → rejet.
- **Retour** : `ValueError("Prompt Injection détectée.")` → converti par `run()` en
  `{"status": "error", "execution_time": …, "message": "Prompt Injection détectée."}`.
- **Ce que le Reasoning doit attendre** : une requête injectée ne produit **jamais**
  une exécution ; elle produit un objet d'erreur `status="error"`.

Motifs interdits (normalisés en minuscules) :
`ignore previous`, `ignore instructions`, `system prompt`, `developer message`,
`reveal prompt`, `show hidden`, `print system`, `bypass`, `override`,
`forget everything`, `disable guard`, `sudo`, `rm -rf`, `drop table`,
`shutdown`, `format c:`.

### 5.2 EthicalFilter (filtre éthique — OPTIONNEL, hors pipeline `run()`)

- **Emplacement** : `app/guardrails/ethical_filter.py` + `ethical_filter_config.json`.
- **Important** : l'EthicalFilter **n'est pas appelé automatiquement** par
  `Executor.run()`. C'est un composant de sécurité supplémentaire que le binôme
  **peut** invoquer avant d'envoyer l'action, s'il veut un filtrage éthique plus fin.
- **API** :
  - `evaluate(text: str) -> dict` :
    ```json
    {
      "decision": "ALLOW" | "BLOCK",
      "justification": "…",
      "risk_score": 0.0,
      "max_weight": 0.0,
      "primary_category": "Prompt Injection" | "Instruction Bypass" | "Hidden Instructions" | "Malicious Command" | "Suspicious Pattern" | null,
      "matches": [{"category": "...", "pattern": "...", "weight": int, "effective_weight": float}]
    }
    ```
  - `inspect(action: dict) -> dict` : même résultat + clé `tool`.
  - `is_allowed(text) -> bool`.
  - `get_statistics() -> {"total_queries", "allowed", "blocked", "block_rate"}`.
- **Configuration** (`ethical_filter_config.json`) : `threshold: 3`,
  poids des catégories `1.0` chacune, `disabled_rules: []`, logging activé
  (`logs/ethical_filter.jsonl`).
- **Politique** : `score >= seuil (3)` → `BLOCK`. Requête vide/invalide → `BLOCK`.

### 5.3 Logger (télémétrie)

- **Emplacement** : `app/telemetry/logger.py`.
- **Fichier** : `logs/astra_exec.log` (créé automatiquement).
- **Méthodes** : `log_success(tool, time)`, `log_error(message, time)`,
  `log_info(msg)`, `log_warning(msg)`, `log_action_start(tool)`,
  `log_action_end(tool)`, `log_event(event, details)`, `now()`.
- Format des lignes : `2026-08-02 12:00:00,000 | INFO | SUCCESS | Tool=X | Time=0.0012s`.

### 5.4 Résumé du flux de sécurité

```
Action entrante
   │
   ├─ Validator.validate        (structure + types)          → bloqué si invalide
   ├─ InjectionGuard.inspect    (anti prompt injection)      → bloqué si score ≥ 2
   │
   │  ── optionnel, côté Reasoning ──
   ├─ EthicalFilter.evaluate    (filtre éthique, seuil 3)    → décision ALLOW/BLOCK
   │
   ▼
Exécution ou erreur structurée
```

---

## 6. Pipeline de retrieval

### 6.1 Segmentation — `DocumentManager` + `SmartSeg`

```
app/api/data/  (.txt + .pdf)
   │  ReaderFactory (extension → lecteur)
   ▼
TXTDocumentReader / PDFDocumentReader  → texte brut
   ▼
SmartSeg.process_text(text, source)
   │  clean_text → remove_noise → split_sentences → split_sections → split_into_chunks
   ▼
chunks: [ {chunk_id, source, length, word_count, content}, … ]
```

- **`DocumentManager(data_folder)`** : `load_documents() -> List[Dict]` (chunk_id global
  unique par lot, `source` = nom du fichier), `available_sources()`, `source_count()`.
  Lève `FileNotFoundError` si aucun fichier supporté.
- **`SmartSeg`** : `chunk_size=500`, `overlap=50`, `min_chunk_size=100`.
  `process(file_path)` (lecture .txt) et `process_text(text, source)` (entrée générique).
- **Schéma d'un chunk** :
  ```json
  { "chunk_id": 0, "source": "machine_learning.txt", "length": 412, "word_count": 61, "content": "…" }
  ```

### 6.2 Recherche hybride — `FusionSearch`

- `build_index(chunks: List[Dict])` : index TF-IDF maison (numpy) + index BM25 (LexiRank).
- `search(query: str, top_k: int = 5, profile: dict = None) -> List[Dict]` :
  - recherche vectorielle (cosinus) sur `top_k*2` candidats ;
  - recherche lexicale BM25 sur `top_k*2` candidats ;
  - fusion pondérée (poids ajustés selon `profile` : keyword 25/75, definition 75/25,
    comparative 60/40, explanatory 65/35, défaut 50/50) ;
  - normalisation min-max, tri descendant.
  - Retour par résultat :
    ```json
    {
      "chunk": { "chunk_id": 0, "source": "…", "length": 412, "word_count": 61, "content": "…" },
      "score": 0.8234, "semantic": 0.9123, "lexical": 0.0,
      "semantic_raw": 0.45, "lexical_raw": 0.0
    }
    ```
- **Erreur** : `ValueError("Index non construit.")` si `search()` avant `build_index()`.
- **Persistance** : `save(path)` / `load(path)` → pickle + numpy (`index/fusion_search.pkl`).

### 6.3 Recherche lexicale — `LexiRank` (BM25 maison)

- `build_index(chunks)`, `search(query, top_k=5)` → `[{chunk, score, raw_bm25, bonus}]`.
- Paramètres BM25 : `k1=1.5`, `b=0.75`. Bonus exact-match (0.15/mot) + bonus taille (0.10).

### 6.4 Re-ranking — `EvidenceRank`

- `rerank(results) -> List[Dict]` : ajoute `final_score` à chaque résultat et trie
  par ordre décroissant.
- Score composite (poids par défaut) :
  `final_score = semantic*0.50 + lexical*0.30 + quality*0.10 + position*0.10`
  où `quality` dépend de `length` (300–800 → 1.0) et `position` de `chunk_id` (≤3 → 1.0).

### 6.5 Profilage — `QueryProfiler`

- `profile(query: str) -> {"query", "length", "type"}`.
- Types : `keyword` (≤2 mots), `comparative` (compare/vs/…), `definition`
  (qu'est-ce/définition/…), `explanatory` (pourquoi/comment/explique…), sinon `semantic`.

### 6.6 Embeddings — `EmbeddingGenerator`

- **Emplacement** : `app/storage/embedding_generator.py` (seul import de `sentence_transformers`).
- Modèle : `SentenceTransformer("all-MiniLM-L6-v2")` — **384 dimensions**, vecteurs `float32`.
- API : `embed_text(text) -> np.ndarray (384,)` · `embed_texts(texts) -> np.ndarray (N×384)`
  · `info()` · attributs `model_name`, `dimension`.

### 6.7 ChromaDB — `ChromaManager`

- **Emplacement** : `app/storage/chroma_manager.py` (seul import de `chromadb`).
- Constantes : `DEFAULT_COLLECTION = "astra_docs"`, `DEFAULT_SPACE = "cosine"`.
- Constructeur : `ChromaManager(path="storage/chroma", collection_name="astra_docs", client_factory=None)`.
- API :
  - `build(chunks, embeddings) -> None` : **idempotent** (supprime puis recrée la collection).
  - `add_chunks(chunks, embeddings) -> None` : ingestion incrémentale (upsert sur `source::chunk_id`).
  - `search(embedding, top_k=5) -> List[Dict]` : recherche **par vecteur uniquement**
    (`query_embeddings`, jamais `query_texts`). Retour :
    ```json
    [
      { "source": "machine_learning.txt", "chunk_id": 3, "length": 412,
        "word_count": 61, "content": "…", "distance": 0.2345 }
    ]
    ```
    (trié du plus proche au plus éloigné ; liste vide si base vide ou collection absente)
  - `count() -> int` · `close() -> None` (libère les verrous) · `delete_collection()`
  - `info() -> {"engine", "path", "collection_name", "space", "count", "client_loaded"}`
- IDs stockés : `"{source}::{chunk_id}"` (ex. `machine_learning.txt::3`).
- Métadonnées stockées : uniquement les clés non nulles parmi
  `source`, `chunk_id`, `length`, `word_count`.
- Validation : `chunks` non vide, clés `content/source/chunk_id` requises,
  `embeddings` matrice `(N, dim)` float32 avec `N = len(chunks)`.

### 6.8 Export — `BaseExporter`

- **Emplacement** : `app/storage/base_export.py`.
- `export(source_dir="storage/chroma", output_zip="exports/base_documentaire_v1.zip") -> str`
  : compresse le dossier Chroma tel quel (aucun format propriétaire).
- `unzip(zip_path, target_dir=".") -> str` : décompresse et **valide la présence de
  `chroma.sqlite3`**.
- Contraintes : le client Chroma doit être **fermé** (`close()`) avant l'export
  (verrous de fichiers sous Windows).

---

## 7. Outils

### 7.1 Contrat `BaseTool` (`app/registry/base_tool.py`)

```python
class BaseTool(ABC):
    def __init__(self, name: str, description: str): ...
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]: ...
    def info(self) -> dict:  # {"name": ..., "description": ...}
    # optionnel : attribut `parameter_schema` (dict) pour la validation
```

Toute action adressée au Reasoning doit cibler un outil **enregistré** dans le
registre de l'Executor. L'API enregistre un seul outil par défaut (ci-dessous).

### 7.2 Outil enregistré par défaut : `fusion_search`

| Champ | Valeur |
|---|---|
| **name** | `"fusion_search"` |
| **description** | `"Recherche hybride AstraExec (TF-IDF custom + BM25 custom)"` |
| **parameter_schema** | `{"query": {"type": "string", "required": true, "description": "Requete de recherche"}}` |
| **paramètre requis** | `query` (chaîne non vide) |
| **paramètres optionnels** | aucun |

**Comportement d'exécution** :
1. `query = kwargs.get("query", "")` ; si vide → `ValueError("Le parametre 'query' est obligatoire.")`.
2. `profile = QueryProfiler().profile(query)`.
3. `results = FusionSearch().search(query, profile=profile)` (top_k par défaut = 5).
4. `ranked = EvidenceRank().rerank(results)` (ajoute `final_score`).
5. Retour :
   ```json
   {
     "results": [
       {
         "chunk": {"chunk_id": 0, "source": "…", "length": 412, "word_count": 61, "content": "…"},
         "score": 0.8234, "semantic": 0.9123, "lexical": 0.0,
         "semantic_raw": 0.45, "lexical_raw": 0.0,
         "final_score": 0.8345
       }
     ],
     "profile": {"query": "…", "length": 4, "type": "definition"}
   }
   ```

**Erreurs possibles** : `Le parametre 'query' est obligatoire.` (query absente/vide) ;
`Index non construit.` (si l'index n'a pas pu être chargé/construit au démarrage).

### 7.3 Registre — `ToolRegistry` (`app/registry/tool_registry.py`)

| Méthode | Rôle | Erreur |
|---|---|---|
| `register(tool)` | Ajoute un outil (clé = `tool.name`) | — |
| `get(name)` | Récupère l'instance | `ValueError("Outil 'X' introuvable.")` |
| `exists(name)` | Vérifie l'existence | — |
| `list_tools()` | `[{name, description}, …]` | — |

---

## 8. Base documentaire (ChromaDB)

### 8.1 Génération

```
app/api/data/  (corpus : .txt + .pdf)
   │  DocumentManager.load_documents()
   ▼
chunks (List[Dict])
   │  EmbeddingGenerator.embed_texts(contents)  → matrice (N × 384) float32
   ▼
ChromaManager.build(chunks, embeddings)
   ▼
storage/chroma/   (PersistentClient ChromaDB)
   │  ChromaManager.close()
   ▼
BaseExporter.export()  →  exports/base_documentaire_v1.zip
```

Lancement : `python demo_database.py` (script d'orchestration complet).

### 8.2 Emplacement et structure

```
storage/
├── chroma/
│   ├── chroma.sqlite3          ← catalogue, documents, métadonnées, ids
│   └── <uuid>/data_level0/     ← index HNSW (vecteurs binaires)
└── exports/
    └── base_documentaire_v1.zip  ← archive livrée au binôme
```

- `storage/` et `exports/` sont **gitignorés** : ils ne sont pas dans le dépôt,
  ils sont générés ou reçus localement.

### 8.3 Paramètres contractuels (à respecter absolument)

| Paramètre | Valeur |
|---|---|
| Moteur | ChromaDB **== 1.5.9** (contrainte de version majeure, voir §8.6) |
| Client | `chromadb.PersistentClient(path="storage/chroma")` |
| Nom de collection | `astra_docs` |
| Espace de distance | `cosine` (stocké dans la config de la collection) |
| Modèle d'embedding | `all-MiniLM-L6-v2` (sentence-transformers) |
| Dimension des vecteurs | 384 (float32) |
| IDs | `"{source}::{chunk_id}"` (ex. `machine_learning.txt::3`) |
| Métadonnées | `source` (str), `chunk_id` (int), `length` (int), `word_count` (int) |
| Contenu | texte brut du chunk (`content`) |
| Méthode de requête | **`query_embeddings` uniquement** — jamais `query_texts` |

### 8.4 Lecture côté binôme (3 lignes)

```python
import chromadb
client = chromadb.PersistentClient(path="storage/chroma")   # après dézip de l'archive
collection = client.get_collection("astra_docs")
```

### 8.5 Requête côté binôme (doit utiliser LE MÊME modèle)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")          # même modèle que la base
query_vec = model.encode("Qu'est-ce que BM25 ?")          # (384,) float32
result = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

# result["documents"][0]   → textes des chunks
# result["metadatas"][0]   → [{"source", "chunk_id", "length", "word_count"}, ...]
# result["distances"][0]   → distances cosinus (plus petit = plus proche)
```

### 8.6 Contraintes de version et compatibilité

1. **Version ChromaDB** : le dossier persistant (sqlite + HNSW) est lié à la
   version majeure de chromadb. Le binôme doit installer **`chromadb==1.5.9`**
   (documenté dans `requirements.txt` du projet Action).
2. **Modèle d'embedding** : le binôme doit disposer du modèle `all-MiniLM-L6-v2`
   (~90 Mo, téléchargement unique ou partage du cache Hugging Face). Les vecteurs
   de requête doivent être produits par **sentence-transformers** (PyTorch) pour
   être identiques à ceux de la base. L'embedding function ONNX par défaut de
   ChromaDB produit des vecteurs quasi-identiques mais pas bit-à-bit.
3. **Espace cosine** : stocké dans la configuration de la collection → rien à
   configurer côté binôme.
4. **`query_texts` interdit** : si le binôme utilisait `query_texts`, ChromaDB
   utiliserait sa propre fonction d'embedding (ONNX), ce qui introduit un écart
   numérique. Le contrat impose `query_embeddings` avec le même modèle.

---

## 9. Contrat d'entrée (ce que le Reasoning doit ENVOYER)

### 9.1 Format minimal obligatoire

```json
{
  "tool": "fusion_search",
  "parameters": { "query": "Qu'est-ce que BM25 ?" }
}
```

| Champ | Type | Obligatoire | Contrainte |
|---|---|---|---|
| `tool` | string | **oui** | Doit exister dans le registre (`fusion_search` par défaut) |
| `parameters` | object (dict) | **oui** | Peut être vide `{}` si l'outil n'a pas de paramètres |

### 9.2 Champs optionnels (non utilisés par `run()`, acceptés par l'API)

`priority` (`low|normal|high`), `confidence` (0–1), `request_id`, `metadata`.
Ils sont **ignorés** par `Executor.run()` (seuls `tool` et `parameters` sont lus),
mais peuvent être portés par l'`ActionInterface` côté Reasoning.

### 9.3 Exemples valides

```json
{ "tool": "fusion_search", "parameters": { "query": "machine learning" } }
{ "tool": "fusion_search", "parameters": { "query": "différence entre BM25 et TF-IDF" } }
```

### 9.4 Contraintes

- `tool` doit être une chaîne **exacte** (sensibilité à la casse).
- `parameters` doit être un **objet JSON** (dict), jamais une liste ni une chaîne.
- La requête `query` doit être une chaîne **non vide** pour `fusion_search`.
- Tout contenu malveillant dans `parameters` (2 motifs interdits ou plus) → rejet.

---

## 10. Contrat de sortie (ce que le Reasoning RECEVRA)

### 10.1 Objet de succès

```json
{
  "status": "success",
  "tool": "fusion_search",
  "execution_time": 0.0234,
  "result": {
    "results": [
      {
        "chunk": {
          "chunk_id": 3,
          "source": "machine_learning.txt",
          "length": 412,
          "word_count": 61,
          "content": "Le machine learning est une branche de l'intelligence artificielle…"
        },
        "score": 0.8234,
        "semantic": 0.9123,
        "lexical": 0.0,
        "semantic_raw": 0.45,
        "lexical_raw": 0.0,
        "final_score": 0.8345
      }
    ],
    "profile": { "query": "…", "length": 4, "type": "definition" }
  }
}
```

| Champ | Type | Description |
|---|---|---|
| `status` | string | `"success"` |
| `tool` | string | nom de l'outil exécuté |
| `execution_time` | float | secondes, arrondi à 4 décimales |
| `result` | object | sortie de l'outil (varie selon l'outil) |
| `result.results[]` | array | résultats de recherche triés (meilleur d'abord) |
| `result.results[i].chunk` | object | `{chunk_id, source, length, word_count, content}` |
| `result.results[i].score` | float | score de fusion normalisé |
| `result.results[i].final_score` | float | score final après re-ranking |
| `result.profile` | object | `{query, length, type}` |

### 10.2 Objet d'erreur

```json
{
  "status": "error",
  "execution_time": 0.0012,
  "message": "L'outil 'inexistant' n'existe pas."
}
```

| Champ | Type | Description |
|---|---|---|
| `status` | string | `"error"` |
| `execution_time` | float | secondes, arrondi à 4 décimales |
| `message` | string | message d'erreur français (exact, voir Section 11) |

### 10.3 Règles d'exploitation côté Reasoning

1. **Toujours vérifier `status` en premier.** Ne jamais lire `result` si `status != "success"`.
2. `execution_time` est **toujours présent** (succès comme erreur).
3. En cas d'erreur, `result` est **absent** ; seule `message` est disponible.
4. Les résultats de `fusion_search` sont triés du plus pertinent au moins pertinent
   (`final_score` décroissant).

---

## 11. Contrat d'erreur (catalogue complet)

### 11.1 Erreurs du pipeline (converties en `{"status": "error", "execution_time", "message"}`)

| # | Condition | Message exact (français) |
|---|---|---|
| 1 | Action non-dict | `Une action doit être un dictionnaire.` |
| 2 | Clé `tool` absente | `Champ obligatoire manquant : tool` |
| 3 | Clé `parameters` absente | `Champ obligatoire manquant : parameters` |
| 4 | `tool` non-chaîne | `Le nom de l'outil doit être une chaîne.` |
| 5 | `parameters` non-dict | `Les paramètres doivent être un dictionnaire.` |
| 6 | Prompt injection (score ≥ 2) | `Prompt Injection détectée.` |
| 7 | Outil inexistant | `L'outil '<nom>' n'existe pas.` |
| 8 | Paramètre obligatoire manquant (schéma) | `Paramètre obligatoire manquant : <nom>` |
| 9 | Mauvais type (schéma) | `Le paramètre '<nom>' doit être de type <type>, reçu : <type réel>` |
| 10 | Valeur non autorisée (schéma) | `Le paramètre '<nom>' doit être l'une des valeurs : [<allowed>], reçu : <valeur>` |
| 11 | Longueur minimale (schéma) | `Le paramètre '<nom>' doit contenir au moins <n> caractère(s)` |
| 12 | Longueur maximale (schéma) | `Le paramètre '<nom>' doit contenir au maximum <n> caractère(s)` |
| 13 | Exception levée par l'outil | texte de l'exception (`str(error)`) — ex. `Le parametre 'query' est obligatoire.` |

### 11.2 Erreurs de l'outil `fusion_search`

| Condition | Message |
|---|---|
| `query` absente ou vide | `Le parametre 'query' est obligatoire.` |
| Index non construit | `Index non construit.` (peut survenir si l'index n'a pas pu être construit au démarrage de l'API) |

### 11.3 Erreurs API (HTTP)

| Code | Condition | Corps |
|---|---|---|
| 422 | body JSON invalide (Pydantic : `tool`/`parameters` manquants ou mauvais type) | erreur de validation FastAPI |
| 400 | exception inattendue dans `/execute` (dernier recours, rare car `run()` ne lève pas) | `{"status": "error", "message": "<str>"}` |

### 11.4 Erreurs `EthicalFilter` (si utilisé séparément)

- `decision: "BLOCK"` avec `justification` commençant par `Requête bloquée : …`
  (score ≥ seuil 3) ou `Requête vide ou invalide : rien à exécuter.`
- Catégories : `prompt_injection`, `instruction_bypass`, `hidden_instructions`,
  `malicious`, `suspicious`.

### 11.5 Comportement garanti

- **`Executor.run()` ne lève JAMAIS d'exception** : toutes les erreurs ci-dessus
  sont retournées comme objets `status="error"`. Le moteur reste fonctionnel après
  n'importe quelle erreur (tests de robustesse : 50 exécutions, 10 erreurs
  consécutives, récupération vérifiée).

---

## 12. Ce que le binôme (Module Reasoning) DOIT implémenter

### Checklist obligatoire

- [ ] **Générer une action JSON valide** : objet avec `tool` (string) et `parameters` (objet), sérialisable en JSON.
- [ ] **Utiliser le nom d'outil exact** : `fusion_search` (sensibilité à la casse).
- [ ] **Respecter le `parameter_schema`** : fournir `query` (string non vide) pour `fusion_search`.
- [ ] **Vérifier `status` systématiquement** après chaque appel : `if result["status"] == "success"` avant toute lecture de `result`.
- [ ] **Ne jamais supposer `result` présent en cas d'erreur** : n'utiliser que `message`.
- [ ] **Ne jamais contourner la validation** : envoyer les actions brutes, laisser l'Executor valider. Ne pas envoyer de contenu contenant ≥ 2 motifs interdits.
- [ ] **Interpréter les erreurs** : en cas de `status="error"`, soit reformuler l'action, soit renvoyer le message à l'utilisateur, soit passer à l'outil suivant.
- [ ] **Traiter les résultats de recherche** : `results` est trié (le premier élément est le plus pertinent) ; chaque `chunk` contient `content`, `source`, `chunk_id`, `length`, `word_count` ; `final_score` indique la pertinence.
- [ ] **Gérer l'absence de résultats** : `results` peut être vide (aucune correspondance) — ce n'est pas une erreur.
- [ ] **Appeler l'API via `POST /execute`** (ou `Executor.run` en intégration directe).
- [ ] **Utiliser le même modèle d'embedding** (`all-MiniLM-L6-v2`, 384 dims) et `query_embeddings` pour toute requête à la base Chroma (`astra_docs`).
- [ ] **Toujours vérifier `status` avant `result`** (répété ici volontairement : c'est la règle n°1 du contrat).

### En-têtes / préconditions

- [ ] Avoir installé `chromadb==1.5.9` et `sentence-transformers` (pour la base).
- [ ] Avoir dézippé `base_documentaire_v1.zip` → dossier `storage/chroma/` à la racine du projet Reasoning (ou chemin adapté passé à `PersistentClient`).
- [ ] Avoir testé la connexion à l'API : `GET /health` → `{"status": "healthy", ...}`.

---

## 13. Ce que le Module Action DOIT envoyer au binôme

### Checklist de livraison

- [ ] **Le présent contrat** (`docs/contrat_integration.md`).
- [ ] **L'archive de la base** : `exports/base_documentaire_v1.zip`
  (contenu : `storage/chroma/` brut — `chroma.sqlite3` + index HNSW).
- [ ] **Le README** (`README.md`) — architecture et technologies.
- [ ] **`docs/livrable4.md`** — documentation de la base documentaire.
- [ ] **`docs/livrable5.md`** — rapport final (sécurité, robustesse, performances).
- [ ] **`requirements.txt`** — versions exactes (`chromadb==1.5.9`, `sentence-transformers`).
- [ ] **Le contrat d'outil `fusion_search`** : nom exact, `parameter_schema`, exemple de résultat (Section 7 de ce document).
- [ ] **Le modèle d'embedding** : info `all-MiniLM-L6-v2`, 384 dims, ou partage du cache Hugging Face (optionnel).
- [ ] **Exemples d'appels** : `curl -X POST http://<host>:<port>/execute -H "Content-Type: application/json" -d '{"tool":"fusion_search","parameters":{"query":"machine learning"}}'`.
- [ ] **Les constantes de la base** : collection `astra_docs`, espace `cosine`, ids `source::chunk_id`, métadonnées `{source, chunk_id, length, word_count}`.

### Comment lancer le module Action (côté Action)

```
# API REST
uvicorn app.api.main:app --reload --port 8000

# ou démo console
python demo.py
```

---

## 14. Ce que le binôme DOIT renvoyer au Module Action

### Checklist de retour (pour une fusion sans crash)

- [ ] **Le JSON d'action** qu'il compte envoyer (structure exacte, Section 9).
- [ ] **Les plans générés** par son module Reasoning (exemples de plans contenant des appels d'outils).
- [ ] **Les noms d'outils** qu'il prévoit d'appeler (doivent correspondre au registre : `fusion_search`, ou accord sur de nouveaux outils).
- [ ] **Le format de réponse attendu** de son côté (pour vérifier la compatibilité, Section 10).
- [ ] **Exemples de prompts** utilisés pour générer les plans (pour valider qu'aucun ne déclenche l'anti-injection).
- [ ] **La version de Python** et des bibliothèques de son environnement.
- [ ] **Son besoin en nouveaux outils** (s'il veut d'autres actions que `fusion_search`, ex. outils de calcul, de stockage) → à coordonner avec le Module Action.

---

## 15. Checklist de fusion finale

Avant de fusionner les deux modules, vérifier point par point :

### 15.1 Compatibilité d'entrée

- [ ] Le Reasoning produit un JSON objet `{"tool", "parameters"}` (jamais liste/chaîne).
- [ ] `tool` est une chaîne exacte existant dans le registre Action.
- [ ] `parameters` est un objet JSON.
- [ ] Les paramètres requis par `parameter_schema` sont fournis (ex. `query`).

### 15.2 Compatibilité de sortie

- [ ] Le Reasoning lit `status` avant `result`.
- [ ] Le Reasoning gère `status == "error"` (message affichable, pas de crash).
- [ ] Le Reasoning exploite `result.results[].chunk.{content, source, chunk_id, length, word_count}` et `final_score`.
- [ ] Le Reasoning accepte `results` vide (pas d'erreur).

### 15.3 Compatibilité base documentaire

- [ ] Le binôme utilise `chromadb==1.5.9`.
- [ ] Le binôme ouvre `storage/chroma/` (dézippé) avec `PersistentClient`.
- [ ] Le binôme utilise la collection `astra_docs`.
- [ ] Le binôme interroge avec `query_embeddings` + `all-MiniLM-L6-v2` (384 dims).
- [ ] Aucun `query_texts` dans le code du binôme.

### 15.4 Noms d'outils et schémas

- [ ] Les noms d'outils du Reasoning correspondent au registre Action.
- [ ] Les paramètres envoyés respectent `parameter_schema` (type, allowed, longueurs).
- [ ] Toute demande de nouvel outil est documentée et validée des deux côtés.

### 15.5 Erreurs et JSON

- [ ] Le Reasoning gère chaque message d'erreur du catalogue (Section 11) sans crash.
- [ ] La sérialisation JSON est correcte des deux côtés (UTF-8, `ensure_ascii=False` côté Action).

### 15.6 Pipeline d'exécution

- [ ] Test de bout en bout : Reasoning → `POST /execute` → `fusion_search` → réponse parsée.
- [ ] Test avec requête malveillante : le Reasoning reçoit `status="error"` + `"Prompt Injection détectée."`.
- [ ] Test avec outil inconnu : `status="error"` + `"L'outil 'X' n'existe pas."`.
- [ ] Test de non-régression : `python -m pytest tests/ -q` → **294 tests verts** côté Action.
- [ ] Test de démo : `python demo.py` (8 modules) fonctionne.

### 15.7 Validation finale

- [ ] Fusion sans crash (exigence de l'encadrante).
- [ ] Les deux étudiants peuvent lancer leur module indépendamment et ensemble.
- [ ] Le rapport de stage documente le contrat d'intégration (ce document).

---

## Annexe A — Références des contrats dans le code

| Contrat | Fichier source |
|---|---|
| Pipeline d'exécution | `app/executor/executor.py` |
| Validation | `app/guardrails/validator.py` |
| Anti-injection | `app/guardrails/injection_guard.py` |
| Filtre éthique | `app/guardrails/ethical_filter.py` + `ethical_filter_config.json` |
| Registre / outils | `app/registry/tool_registry.py`, `app/registry/base_tool.py` |
| Recherche hybride | `app/retrieval/fusion_search.py`, `app/retrieval/lexi_rank.py`, `app/retrieval/evidence_rank.py`, `app/retrieval/query_profiler.py` |
| Segmentation | `app/retrieval/smart_seg.py`, `app/retrieval/document_manager.py` |
| Lecteurs | `app/retrieval/reader_factory.py`, `app/retrieval/txt_reader.py`, `app/retrieval/pdf_reader.py` |
| Embeddings | `app/storage/embedding_generator.py` |
| Base Chroma | `app/storage/chroma_manager.py` |
| Export | `app/storage/base_export.py` |
| API | `app/api/main.py` |
| Télémétrie | `app/telemetry/logger.py` |
| Schéma d'action | `app/schemas/action_interface.py` |

## Annexe B — Constantes critiques à partager

```python
# app/storage/chroma_manager.py
DEFAULT_COLLECTION = "astra_docs"
DEFAULT_SPACE = "cosine"

# app/storage/embedding_generator.py
EmbeddingGenerator.model_name  # "all-MiniLM-L6-v2" (attribut)
EmbeddingGenerator.dimension   # 384 (attribut)

# app/storage/base_export.py
DEFAULT_OUTPUT = "exports/base_documentaire_v1.zip"
DEFAULT_SOURCE = "storage/chroma"
```

## Annexe C — Exemple minimal de test d'intégration (côté binôme)

```python
import requests

# 1. Vérifier que le module Action est vivant
health = requests.get("http://localhost:8000/health").json()
assert health["status"] == "healthy"

# 2. Exécuter une action
r = requests.post(
    "http://localhost:8000/execute",
    json={"tool": "fusion_search", "parameters": {"query": "machine learning"}},
)
data = r.json()

# 3. Règle n°1 : vérifier status avant result
if data["status"] == "success":
    for item in data["result"]["results"]:
        print(item["final_score"], item["chunk"]["source"], item["chunk"]["content"][:60])
else:
    print("Erreur :", data["message"])
```

---

*Fin du contrat d'intégration. En cas de divergence entre ce document et le code,
le code fait foi (références en Annexe A).*
