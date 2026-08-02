# Étude d'architecture — Support PDF (Livrable 3, Étape 2)

> Document d'architecture — à valider avant toute implémentation.
> Aucun code n'est produit à cette étape.

---

## 1. Étude des bibliothèques PDF

Trois bibliothèques candidates ont été comparées pour l'extraction de texte
dans un pipeline RAG :

| Critère | PyMuPDF (`fitz`) | pdfplumber | pypdf |
|---|---|---|---|
| **Moteur** | C (MuPDF) | Python (pdfminer.six) | Python pur |
| **Vitesse** | 10–50× plus rapide que les libs pures (benchmarks py-pdf, 2025) | Lente (analyse caractère par caractère) | Modérée |
| **Qualité extraction** | Excellente (coordonnées, polices, blocs) | Excellente, la meilleure pour les tableaux | Correcte sur PDF numériques simples, limitée sur mises en page complexes |
| **Tableaux** | De base | Avancée (`extract_tables()`) | Manuelle / heuristique |
| **Mémoire** | Efficace (C optimisé) | Élevée (modèle objet complet par caractère) | Faible |
| **Maintenance** | Très active (wrapper de l'industrie MuPDF) | Active (standard communautaire tableaux) | Très active (successeur de PyPDF2) |
| **Dépendances** | Une wheel pip (C-bindings) | pdfminer.six | Aucune |
| **PDF protégés** | Oui (avec mot de passe) | Oui | Oui |
| **PDF scannés (OCR)** | Non natif (rendu d'images possible) | Non natif | Non natif |
| **Fichiers corrompus** | Réparation automatique robuste | Fragile | Fragile |
| **Licence** | AGPL-3.0 / commerciale | MIT | BSD-3 |

*Sources : PyMuPDF docs (about/features), py-pdf/benchmarks (2025-06),*
*nutrient.io « Best Python PDF Libraries » (2026), pypdf docs (comparisons).*

### Analyse détaillée

**PyMuPDF (fitz)**
- **Performances** : moteur C MuPDF, benchmarks py-pdf montrent un gain de
  10 à 50× vs bibliothèques pure-Python. Traitement de milliers de pages en
  quelques secondes.
- **Qualité** : extraction avec coordonnées et métadonnées de police ;
  excellent rendu du texte numérique, y compris les caractères spéciaux
  (accents, symboles, unicode).
- **Intégration** : une seule dépendance pip (`PyMuPDF`), API simple
  (`fitz.open()` → `page.get_text()`).
- **Robustesse** : réparation automatique des fichiers endommagés — un atout
  majeur pour un pipeline d'indexation.
- **Limitations** : licence AGPL-3.0 (à vérifier selon le contexte du stage) ;
  pas d'OCR natif (mais rendu d'images possible pour une future extension) ;
  extraction de tableaux basique.

**pdfplumber**
- **Qualité** : référence pour les tableaux et la mise en page spatiale.
- **Performances** : la plus lente (parsing caractère par caractère) ;
  consommation mémoire élevée.
- **Limitations** : hérite des dépendances de pdfminer.six ; surdimensionné
  pour une recherche hybride où le texte est segmenté en chunks.

**pypdf**
- **Avantages** : pur Python, zéro dépendance, très simple.
- **Limitations** : qualité moindre sur les mises en page complexes, colonnes
  multiples et encodages non standard ; pas de réparation des fichiers
  corrompus ; plus lent que PyMuPDF.

### Justification technique du choix : **PyMuPDF (fitz)**

Le besoin d'AstraExec n'est **pas** l'extraction de tableaux structurés
(définition de pdfplumber) mais l'extraction fiable et rapide de texte brut
destiné à la segmentation (SmartSeg). Dans ce contexte :

1. **Vitesse** : gain de 10–50× — déterminant pour un corpus appelé à grandir.
2. **Robustesse** : réparation automatique des fichiers corrompus — critique
   pour l'indexation en lot sans interruption.
3. **Qualité unicode** : les documents du corpus sont en français (accents,
   caractères spéciaux) — PyMuPDF gère correctement l'unicode.
4. **Intégration** : une seule dépendance, wheels officielles, aucune
   dépendance système à compiler.
5. **Évolutivité** : le rendu de pages en images (fonctionnalité exclusive de
   PyMuPDF parmi les trois) ouvre une piste future (OCR/VLM) sans changer de
   bibliothèque.

**Réserve à documenter** : la licence AGPL-3.0. Pour un projet de stage
d'ingénieur à usage académique/démonstration, elle est acceptable ; si la
contrainte licences devenait bloquante, pdfplumber (MIT) serait le repli —
avec l'impact de performance documenté ci-dessus. Le choix PyMuPDF est fait
pour les qualités techniques ; la couche d'architecture proposée (Section 2)
isole l'extraction derrière une interface, ce qui permet de changer de
bibliothèque sans toucher au reste du pipeline.

---

## 2. Architecture proposée

### Principe directeur

Pas de `if extension == ".pdf"` éparpillé dans le code. Le pattern
**Strategy + Factory** est appliqué : chaque format est pris en charge par un
lecteur dédié, sélectionné par une fabrique selon l'extension. Le reste du
pipeline (nettoyage, segmentation, chunking, métadonnées) est **totalement
indépendant du format d'entrée**.

### Composants et responsabilités

| Composant | Responsabilité | Statut |
|---|---|---|
| **DocumentManager** | Orchestration : parcours du dossier, choix du lecteur, numérotation globale des chunks | Modifié (extension) |
| **PDFDocumentReader** | Extraction du texte brut + pages depuis un PDF (PyMuPDF) | **Nouveau** |
| **TXTDocumentReader** | Extraction du texte brut depuis un .txt (encodages multiples) | **Nouveau** |
| **TextCleaner** | Nettoyage (encodage, espaces, bruit, caractères de contrôle) | **Nouveau** (logique extraite de SmartSeg, sans changement de comportement) |
| **ChunkGenerator** | Segmentation : phrases → sections → chunks avec overlap | **Nouveau** (logique extraite de SmartSeg, sans changement de comportement) |
| **MetadataBuilder** | Construction des métadonnées (chunk_id, source, longueur, mots, page) | **Nouveau** (logique extraite de SmartSeg, enrichie `page` pour PDF) |
| **SmartSeg** | Composant central : réutilise TextCleaner, ChunkGenerator, MetadataBuilder en interne | API publique et comportement inchangés (composition interne) |

### Diagramme d'architecture

```
                       ┌──────────────────────────────┐
                       │        DocumentManager       │
                       │   (orchestration, extension) │
                       └──────────────┬───────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │     ReaderFactory         │
                        │  .txt → TXTDocumentReader │
                        │  .pdf → PDFDocumentReader │
                        └─────────────┬─────────────┘
                                      │  texte brut (+ pages pour PDF)
                                      ▼
                       ┌──────────────────────────────┐
                       │           SmartSeg           │  (central, inchangé)
                       │   ┌───────────┐              │
                       │   │TextCleaner│              │
                       │   └─────┬─────┘              │
                       │         ▼                    │
                       │   ┌──────────────┐           │
                       │   │ChunkGenerator│           │
                       │   └─────┬────────┘           │
                       │         ▼                    │
                       │   ┌────────────────┐         │
                       │   │MetadataBuilder │         │
                       │   └────────────────┘         │
                       └──────────────┬───────────────┘
        (API publique et comportement inchangés)
                                      │  chunks : List[Dict]
                                      ▼
                       ┌──────────────────────────────┐
                       │        FusionSearch          │   (inchangé)
                       └──────────────────────────────┘
```

### Relations entre composants

- **DocumentManager → ReaderFactory** : délègue le choix du lecteur (ouvert/
  fermé : ajouter un format = ajouter un lecteur, sans modifier le manager).
- **ReaderFactory → TXT/PDFDocumentReader** : instancie le lecteur adapté.
- **Reader → SmartSeg** : les lecteurs retournent du texte brut ; SmartSeg
  reçoit `(texte, source)` via une méthode d'entrée générique
  (par ex. `process_text`) sans jamais connaître le format source.
- **SmartSeg → TextCleaner / ChunkGenerator / MetadataBuilder** : composition
  interne (DIP) ; SmartSeg conserve son API publique actuelle
  (`process(file_path)`, `load_text`, `clean_text`, …) pour ne rien casser.
- **SmartSeg → FusionSearch** : contrat de données inchangé (liste de dicts
  `chunk_id`, `content`, `source`, `length`, `word_count`).

### Pourquoi cette répartition respecte SOLID

- **S** : chaque classe a une seule raison de changer (lecture, nettoyage,
  découpage, métadonnées, orchestration).
- **O** : nouveaux formats par ajout de lecteurs, sans modification de
  DocumentManager ni de SmartSeg.
- **L** : les lecteurs implémentent la même interface (`read(path) → texte`),
  substituables.
- **I** : les lecteurs n'exposent que la méthode utile à l'appelant.
- **D** : SmartSeg et DocumentManager dépendent d'abstractions
  (interface de lecteur), pas de PyMuPDF directement.

---

## 3. Pipeline complet

```
PDF / TXT
   │
   ▼
1. Lecture (PDFDocumentReader / TXTDocumentReader)
   │  ouverture, décodage, vérification d'intégrité
   ▼
2. Extraction du texte (PDF uniquement : page.get_text() par page)
   │  texte brut + numéros de pages
   ▼
3. Nettoyage (TextCleaner)
   │  unicode, espaces insécables, retours chariot, caractères de contrôle
   ▼
4. Segmentation en phrases (ChunkGenerator)
   │  découpage phrase par phrase (regex existante de SmartSeg)
   ▼
5. Groupement en sections (ChunkGenerator)
   │  sections de ~chunk_size caractères
   ▼
6. Chunking avec overlap (ChunkGenerator)
   │  fenêtres glissantes de 500c, overlap 50c, min 100c
   ▼
7. Création des métadonnées (MetadataBuilder)
   │  chunk_id, source, length, word_count, page (PDF)
   ▼
8. Retour des chunks (DocumentManager → FusionSearch)
   │  List[Dict] au contrat de données existant
```

**Détail des étapes :**

1. **Lecture** — le lecteur sélectionné ouvre le fichier. Pour TXT : boucle
   d'encodages (utf-8, latin-1, cp1252, iso-8859-1) déjà en place. Pour PDF :
   ouverture PyMuPDF, gestion du mot de passe, itération sur les pages.
2. **Extraction** — PDF : `page.get_text("text")` par page (texte numérique) ;
   les numéros de page sont conservés pour les métadonnées. TXT : retour direct.
3. **Nettoyage** — mêmes règles que SmartSeg actuelles (aucun changement de
   comportement) : `\xa0` → espace, suppression `\r`, espaces multiples,
   `\n{3,}` → `\n\n`, caractères de contrôle remplacés.
4–6. **Segmentation / Chunking** — reproduction exacte de la logique actuelle
   de SmartSeg (`split_sentences`, `split_sections`, `split_into_chunks`).
7. **Métadonnées** — mêmes champs qu'aujourd'hui ; ajout additif de `page`
   pour les chunks issus de PDF (absent pour TXT, donc aucun impact sur le
   contrat actuel).
8. **Retour** — liste de dicts identique à aujourd'hui pour le TXT ;
   FusionSearch est totalement transparent au format source.

---

## 4. Cas particuliers

| Cas | Problème | Stratégie | Comportement attendu |
|---|---|---|---|
| **PDF vide** | 0 page ou aucun texte extrait | Détection texte vide après extraction | Avertissement console + chunk(s) non généré(s) ; l'indexation continue |
| **PDF protégé par mot de passe** | Ouverture impossible sans mot de passe | Tentative d'ouverture ; si `needs_pass`, documenter le comportement (mot de passe non fourni dans ce périmètre) | Exception propre levée par le lecteur, interceptée par DocumentManager → fichier ignoré avec message clair |
| **PDF scanné (image)** | Aucun texte numérique extractible | Détection : texte extrait quasi vide alors que le PDF a des pages | Avertissement « PDF scanné : OCR non pris en charge (hors périmètre) » ; fichier ignoré ; piste OCR documentée en perspectives |
| **PDF multi-pages** | Texte et numéros de page | Extraction page par page ; association du numéro de page aux chunks | Chunks avec métadonnée `page` ; le texte est segmenté globalement (flux continu), la page est attribuée selon la position |
| **PDF avec tableaux** | Structure tabulaire | Extraction texte brute (`get_text`) — les tableaux deviennent du texte linéarisé | Texte indexable ; les tableaux structurés ne sont pas reconstruits (hors périmètre, piste pdfplumber documentée) |
| **Caractères spéciaux** | Accents, symboles, unicode | Extraction native PyMuPDF (unicode) + TextCleaner existant | Texte correctement décodé, tokens analysés sans erreur |
| **PDF très volumineux** | Consommation mémoire | Extraction page par page, pas de chargement entier en mémoire | Pipeline stable ; les chunks sont produits au fil de l'eau |
| **Erreurs de lecture** | Fichier illisible, PDF mal formé | `try/except` au niveau du lecteur, exception typée propagée à DocumentManager | DocumentManager ignore le fichier fautif, logge l'erreur et continue (résilience du lot) |
| **Fichiers corrompus** | Données PDF invalides | Réparation automatique PyMuPDF + interception des exceptions | Fichier ignoré avec message clair ; l'indexation des autres fichiers n'est pas interrompue |

**Règle transversale** : une erreur sur un fichier ne doit **jamais** bloquer
l'indexation des autres fichiers (résilience par lot).

---

## 5. Compatibilité (rétrocompatibilité TXT)

Le comportement actuel sur les fichiers `.txt` doit rester strictement
identique. Mesures :

1. **SmartSeg conserve son API publique actuelle** : `process(file_path)`,
   `load_text`, `clean_text`, `remove_noise`, `split_sentences`,
   `split_sections`, `split_into_chunks`, `build_metadata`, `info`.
   Les méthodes gardent les mêmes signatures et le même comportement.
2. **Extraction de logique sans changement de comportement** :
   TextCleaner, ChunkGenerator et MetadataBuilder sont extraits **à
   l'identique** (copie exacte de la logique existante) puis SmartSeg les
   compose en interne. Les résultats produits sont bit-à-bit identiques.
3. **Contrat de données inchangé** : les chunks TXT contiennent exactement
   les mêmes clés qu'aujourd'hui (`chunk_id`, `source`, `length`,
   `word_count`, `content`). La clé `page` n'est ajoutée que pour les PDF.
4. **DocumentManager** : le parcours `.txt` produit la même numérotation
   globale et le même ordre (fichiers triés alphabétiquement).
5. **Garde de non-régression** : la suite de tests existante (114 tests,
   dont `test_smart_seg.py`, `test_fusion_search.py`) doit passer à
   l'identique, et `demo_evaluation.py` doit toujours produire
   **Mean Recall@5 = 0.8917** et **MRR = 1.0000**.
6. **Contrainte de routage (issue du code actuel)** : `SmartSeg.load_text`
   lève une `ValueError` pour toute extension ≠ `.txt`, et `process(path)`
   appelle `load_text`. Par conséquent, **le chemin PDF doit passer
   exclusivement par la nouvelle méthode `process_text(text, source)`**
   et jamais par `process(path)` sur un PDF.
7. **Dépendance optionnelle** : PyMuPDF est ajouté à `requirements.txt` ;
   le code TXT n'importe pas PyMuPDF (import uniquement dans
   `PDFDocumentReader`), donc le chemin TXT fonctionne même sans la lib.

---

## 6. Modifications du projet

### Nouveaux fichiers

| Fichier | Contenu |
|---|---|
| `app/retrieval/pdf_reader.py` | `PDFDocumentReader` (PyMuPDF) |
| `app/retrieval/txt_reader.py` | `TXTDocumentReader` (boucle d'encodages) |
| `app/retrieval/reader_factory.py` | `ReaderFactory` (sélection par extension) |
| `app/retrieval/text_cleaner.py` | `TextCleaner` (logique extraite de SmartSeg) |
| `app/retrieval/chunk_generator.py` | `ChunkGenerator` (phrases/sections/chunks) |
| `app/retrieval/metadata_builder.py` | `MetadataBuilder` (métadonnées + `page`) |
| `tests/test_pdf.py` | Tests du lecteur PDF |
| `tests/test_readers.py` | Tests des lecteurs TXT/PDF + factory |

*(Alternative plus minimaliste : regrouper les 3 classes de pipeline dans un
seul fichier `app/retrieval/text_pipeline.py`. Un fichier par classe a été
privilégié pour la clarté du soutenance et le principe de responsabilité
unique.)*

### Fichiers à modifier

| Fichier | Modification |
|---|---|
| `app/retrieval/document_manager.py` | Utiliser `ReaderFactory` ; accepter `.txt` + `.pdf` ; résilience par lot ; `available_sources` étendu |
| `app/retrieval/smart_seg.py` | Composition interne des nouveaux composants ; **API publique et comportement inchangés** |
| `requirements.txt` | Ajout de `PyMuPDF` |
| `docs/livrable3.md` (étape 6) | Documenter l'architecture PDF |

### Classes à modifier

- `DocumentManager` (orchestration, extension de formats)
- `SmartSeg` (uniquement l'implémentation interne, par composition)

### Méthodes à ajouter

- `PDFDocumentReader.read(path) → Dict` (texte + pages + source)
- `TXTDocumentReader.read(path) → Dict`
- `ReaderFactory.create(path) → lecteur` (+ `SUPPORTED_EXTENSIONS`)
- `TextCleaner.clean(text) → str`
- `ChunkGenerator.generate(text) → List[str]`
- `MetadataBuilder.build(chunks, source, pages=None) → List[Dict]`
- `DocumentManager._index_file(path) → List[Dict]` (extraction propre par fichier)
- `SmartSeg.process_text(text, source) → List[Dict]` (entrée générique pour PDF)

### Méthodes qui resteront inchangées

- `SmartSeg` : `process`, `load_text`, `clean_text`, `remove_noise`,
  `split_sentences`, `split_sections`, `split_into_chunks`, `build_metadata`,
  `info`
- `FusionSearch`, `LexiRank`, `QueryProfiler`, `EvidenceRank`,
  `Evaluator`, `demo_evaluation.py` : **aucune modification**

---

## 7. Plan d'implémentation (ordre recommandé)

> Aucun code n'est écrit à cette étape ; l'ordre ci-dessous minimise les
> risques de régression.

1. **Extraction à l'identique** : créer `TextCleaner`, `ChunkGenerator`,
   `MetadataBuilder` en copiant exactement la logique de SmartSeg.
2. **Composition interne** : faire composer SmartSeg par ces classes.
   **Point de contrôle :** lancer `pytest` (114 tests) + `demo_evaluation.py`
   → résultats identiques (0.8917 / 1.0000). C'est l'étape la plus risquée.
3. **Lecteur TXT** : extraire la boucle d'encodages dans
   `TXTDocumentReader`, derrière l'interface commune.
4. **Lecteur PDF** : implémenter `PDFDocumentReader` avec PyMuPDF
   (extraction page par page, détection vide/scanné, erreurs typées).
5. **ReaderFactory** : sélection par extension, `SUPPORTED_EXTENSIONS`.
6. **DocumentManager** : basculer sur la factory + résilience par lot +
   `available_sources` étendu. **Point de contrôle :** `demo_evaluation.py`
   inchangé + test d'un corpus mixte txt/pdf.
7. **Métadonnées PDF** : ajout `page` (additif, uniquement PDF).
8. **Tests** : `tests/test_pdf.py`, `tests/test_readers.py`.
9. **Documentation** : mettre à jour `docs/livrable3.md`.

### Risques et mitigations

| Risque | Gravité | Mitigation |
|---|---|---|
| Régression TXT lors de l'extraction des composants | Élevée | Copie exacte de la logique + point de contrôle 2 (tests + demo_evaluation) |
| Licence AGPL de PyMuPDF | Moyenne | Documenté ; lecteur isolé derrière l'interface (bascule possible vers pdfplumber) |
| PDF scannés non traités | Moyenne | Détection + message clair ; piste OCR en perspectives |
| PDF volumineux / mémoire | Faible | Extraction page par page |
| Erreur d'un fichier bloquant le lot | Faible | Résilience par fichier (try/except typé) |

### Tests à prévoir

- **Régression** : les 114 tests existants + `demo_evaluation.py`
  (0.8917 / 1.0000).
- **Lecteurs** : TXT (4 encodages), PDF texte numérique, PDF multi-pages
  (métadonnée `page`), PDF vide, PDF protégé, PDF scanné (texte quasi vide),
  fichier corrompu → message clair sans interruption du lot.
- **Fixtures** : le corpus ne contient aucun PDF — les fichiers de test
  (y compris PDF protégé / scanné) seront **générés programmatiquement avec
  PyMuPDF lui-même**, sans dépendre de fichiers externes.
- **Factory** : extension inconnue → erreur explicite.
- **SmartSeg** : `process` (chemin TXT) et `process_text` (texte) produisent
  des chunks équivalents.
- **DocumentManager** : dossier mixte txt/pdf → numérotation globale
  continue, sources correctes.

---

## Conclusion

L'architecture proposée ajoute le support PDF **sans modifier le contrat
public** de SmartSeg ni DocumentManager, **sans toucher au pipeline de
recherche** (FusionSearch, LexiRank, EvidenceRank) et **sans altérer les
résultats du Livrable 2**. PyMuPDF est choisi pour ses performances, sa
robustesse et sa qualité d'extraction, derrière une interface qui rend le
choix de bibliothèque réversible. Chaque nouvelle responsabilité est portée
par un composant dédié, conformément aux principes SOLID, et l'ajout d'un
nouveau format futur (docx, odt…) se réduira à l'écriture d'un nouveau
lecteur.
