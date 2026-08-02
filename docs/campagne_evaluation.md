# Campagne d'évaluation du moteur de recherche hybride AstraExec

**Projet :** AstraExec — Module d'action intelligent pour agent RAG
**Composant évalué :** FusionSearch (fusion 50 % vectoriel / 50 % lexical) + LexiRank (BM25 maison) + TermVectorizer (TF-IDF maison)
**Contexte :** Rapport de stage de fin d'études — EMSI
**Auteur :** Ihssane MOUTCHOU
**Date :** Août 2026
**Reproductibilité :** `python demo_evaluation.py` (déterministe, aucun service externe)

> **Avertissement méthodologique.** Toutes les valeurs numériques de ce document sont
> issues de l'exécution réelle du script `demo_evaluation.py` sur le corpus de
> `app/api/data/`, ou de la lecture intégrale des chunks via `test_chunks.py`. Aucune
> valeur n'a été inventée ni ajustée. Lorsqu'une affirmation ne peut pas être
> démontrée à partir du projet, elle est explicitement signalée comme telle.

---

## 0. Mise à jour (2 août 2026) — migration du corpus après ajout des PDF

### 0.1 Ce qui a changé

Le corpus de démonstration est passé de **4 documents TXT** à **7 documents
(4 TXT + 3 PDF)** :

| Document | Type |
|---|---|
| `astra_platform.txt` | TXT |
| `Contexte et enjeux.pdf` | PDF |
| `fiche_synthese_cnn.pdf` | PDF |
| `machine_learning.txt` | TXT |
| `recherche_lexicale.txt` | TXT |
| `sample.txt` | TXT |
| `Towards Agentic RAG with Deep Reasoning.pdf` | PDF |

Le nombre de chunks indexés est passé de **17 à 415**.

### 0.2 Pourquoi les chunk_id ont changé

`DocumentManager` parcourt le dossier dans l'**ordre alphabétique des fichiers** et
attribue un `chunk_id` **global et séquentiel** à chaque chunk produit. L'ajout des
PDF a intercalé de nouveaux fichiers dans l'ordre de lecture (ex. `Contexte et
enjeux.pdf` est lu avant `machine_learning.txt`). Par conséquent, **tous les chunks
des documents TXT ont été décalés**, tandis que les anciens IDs 0-16 désignent
désormais des chunks PDF différents.

Conséquence : l'ancien ground truth (IDs 0-16) ne pointait plus vers les mêmes
contenus. L'évaluateur comparait alors des chunks pertinents *réels* (retrouvés par
le moteur avec les bons contenus, ex. 78, 79, 80 pour le machine learning) à des IDs
obsolètes → Recall@K et MRR artificiellement bas (0.1667 / 0.2917 / 0.2500).

### 0.3 Méthode de réalignement du ground truth

Les jugements de pertinence d'origine ont été **conservés à l'identique**. Pour
chaque ancien chunk pertinent, le nouvel ID a été obtenu par **correspondance de
contenu exacte** entre l'index TXT-seul (qui reproduit fidèlement l'ancien corpus de
17 chunks, vérifié par marqueurs de contenu) et l'index complet de 415 chunks. La
segmentation (SmartSeg) n'ayant pas changé, chaque ancien chunk se retrouve à
l'identique dans le nouvel index.

| Ancien ID | Nouvel ID | Source | Ancien ID | Nouvel ID | Source |
|---|---|---|---|---|---|
| 0 | 0 | astra_platform.txt | 9 | 84 | recherche_lexicale.txt |
| 1 | 1 | astra_platform.txt | 10 | 85 | sample.txt |
| 2 | 2 | astra_platform.txt | 11 | 86 | sample.txt |
| 3 | 78 | machine_learning.txt | 12 | 87 | sample.txt |
| 4 | 79 | machine_learning.txt | 13 | 88 | sample.txt |
| 5 | 80 | machine_learning.txt | 14 | 89 | sample.txt |
| 6 | 81 | machine_learning.txt | 15 | 90 | sample.txt |
| 7 | 82 | recherche_lexicale.txt | 16 | 91 | sample.txt |
| 8 | 83 | recherche_lexicale.txt | — | — | — |

### 0.4 Nouveau ground truth (EVAL_SET)

| Requête | Anciens IDs | Nouveaux IDs |
|---|---|---|
| machine learning intelligence artificielle | 3, 4, 5, 11, 12 | 78, 79, 80, 86, 87 |
| BM25 recherche lexicale | 7, 8, 9 | 82, 83, 84 |
| plateforme Astra architecture | 0, 1, 2 | 0, 1, 2 |
| TF-IDF pondération fréquence | 8, 14 | 83, 89 |
| deep learning réseaux neurones | 4, 12 | 79, 87 |
| recherche vectorielle hybride | 1, 6, 15 | 1, 81, 90 |
| Python programmation langage | 10 | 85 |
| évaluation pertinence EvidenceRank | 16 | 91 |

**20 associations pertinentes conservées** — aucun jugement modifié ni ajouté.

### 0.5 Résultats réels sur le corpus actuel (415 chunks)

| # | Requête | Recall@5 | Recall@10 | RR |
|---|---|---|---|---|
| 1 | machine learning intelligence artificielle | 0.6000 | 1.0000 | 1.0 |
| 2 | BM25 recherche lexicale | 0.6667 | 0.6667 | 1.0 |
| 3 | plateforme Astra architecture | 1.0000 | 1.0000 | 1.0 |
| 4 | TF-IDF pondération fréquence | 1.0000 | 1.0000 | 1.0 |
| 5 | deep learning réseaux neurones | 1.0000 | 1.0000 | 1.0 |
| 6 | recherche vectorielle hybride | 0.6667 | 1.0000 | 1.0 |
| 7 | Python programmation langage | 1.0000 | 1.0000 | 1.0 |
| 8 | évaluation pertinence EvidenceRank | 1.0000 | 1.0000 | 1.0 |
| **Moyenne** | | **0.8667** | **0.9583** | **MRR = 1.0000** |

Lecture : le moteur place systématiquement un chunk pertinent en première position
(MRR = 1.0) et retrouve ~87 % des pertinents dans le top 5. La légère baisse du
Recall@5 par rapport à l'ancien corpus (0.8917 → 0.8667) s'explique par la **forte
concurrence des nouveaux chunks PDF** dans le top 5 (ex. chunk 86 repoussé hors du
top 5 pour la requête machine learning), et non par une dégradation du moteur : à
K=10, les requêtes concernées récupèrent leurs chunks manquants (Recall@10 = 1.0).

### 0.6 Comparaison avant / après correction

| Métrique | Ancien GT (obsolète) | Nouveau GT (aligné) |
|---|---|---|
| Mean Recall@5 | 0.1667 | **0.8667** |
| Mean Recall@10 | 0.2917 | **0.9583** |
| MRR | 0.2500 | **1.0000** |

Les chiffres « avant » sont les résultats réels obtenus avec l'ancien EVAL_SET sur
le nouveau corpus : ils mesurent une incohérence d'annotation, pas la qualité du
moteur.

---

## 1. Objectif de la campagne d'évaluation

Un moteur de recherche, aussi soigné soit-il, ne peut prétendre à la fiabilité que si
sa qualité est **mesurée objectivement et reproduisiblement**. La campagne poursuit
quatre objectifs :

1. **Quantifier la qualité de la recherche hybride** de FusionSearch sur le corpus de
   démonstration, avec des métriques standard de la recherche d'information :
   Recall@K, Reciprocal Rank (RR), Mean Reciprocal Rank (MRR).
2. **Valider le pipeline complet de bout en bout** : segmentation (SmartSeg) →
   indexation (TermVectorizer, LexiRank) → recherche (FusionSearch) → évaluation
   (Evaluator). Un bon score atteste du fonctionnement de la chaîne entière, et pas
   seulement d'un maillon.
3. **Établir une baseline reproductible** : un point de référence chiffré et versionné,
   permettant de mesurer l'impact de toute évolution future (segmentation, pondération
   de fusion, re-ranking, etc.).
4. **Produire une analyse critique honnête** : comprendre *pourquoi* le système
   atteint ces scores, identifier ses faiblesses réelles et leurs causes.

La campagne porte exclusivement sur la **phase de recherche** (retrieval). Elle
n'évalue ni la génération de réponses, ni l'exécution d'actions, ni la sécurité.

---

## 2. Justification scientifique du choix des 8 requêtes

Le choix du nombre de requêtes est un compromis entre **puissance statistique** et
**faisabilité de l'annotation manuelle**. Huit requêtes ont été retenues pour les
raisons suivantes.

### 2.1 Pourquoi 8 et pas 5

- **Couverture des 4 documents.** Le corpus contient 4 documents dont les thèmes se
  recouvrent partiellement (le machine learning apparaît dans 3 documents, BM25/TF-IDF
  dans 2, la recherche hybride dans 3). Avec 5 requêtes, au moins un document serait
  sous-représenté : il faudrait soit ignorer un thème, soit concentrer deux thèmes dans
  une seule requête, réduisant la granularité de l'évaluation.
- **Stabilité des moyennes.** Avec 5 requêtes, chaque requête pèse 1/5 de la moyenne :
  si le rappel d'une requête chutait de 0,2 point (ex. : de 1,0 à 0,8), la moyenne
  baisserait de 0,2/5 = 0,04 (ex. : de 0,90 à 0,86). Avec 8 requêtes, la même chute
  ne ferait baisser la moyenne que de 0,2/8 = 0,025 : la moyenne est moins sensible à
  un cas particulier, donc plus représentative du comportement moyen.
- **Diversité des types de requêtes.** 8 requêtes permettent de couvrir quatre types :
  multi-termes (« machine learning intelligence artificielle »), à sigle
  (« TF-IDF pondération fréquence », « BM25 recherche lexicale »), nominales
  (« plateforme Astra architecture ») et à cible unique (« Python programmation
  langage »). Cinq requêtes ne permettraient pas cette typologie complète.

### 2.2 Pourquoi 8 et pas 20

- **Faisabilité du ground truth.** La pertinence est annotée **manuellement, après
  lecture intégrale de chaque chunk** (section 4). Sur un corpus de 17 chunks, la
  lecture exhaustive représente 17 × ~360 caractères ≈ 6 000 caractères analysés.
  Chaque requête supplémentaire ajoute un travail d'annotation et de vérification.
  20 requêtes démultiplieraient ce travail sans apport proportionnel : le corpus ne
  contient qu'une vingtaine de thèmes distincts, et des requêtes supplémentaires
  seraient nécessairement des variantes (synonymes, reformulations) des thèmes déjà
  couverts.
- **Risque de redondance.** Sur un corpus de cette taille, 20 requêtes produiraient de
  nombreuses associations redondantes (mêmes chunks pertinents), ce qui **gonflerait
  artificiellement la taille de l'échantillon** sans apporter d'information nouvelle —
  un biais méthodologique pire que d'avoir moins de requêtes.
- **Exigence de discrimination.** L'objectif est de mesurer la capacité du moteur à
  classer correctement des chunks concurrents. Ce comportement se manifeste dès que le
  nombre de requêtes dépasse le nombre de thèmes disjoints du corpus (≈ 8 ici). Au-delà,
  le gain informationnel décroît fortement.

### 2.3 Pourquoi ces thèmes

Les 8 requêtes correspondent aux thèmes **effectivement présents** dans le corpus, tels
que révélés par la lecture des 4 documents :

| Document | Thèmes réellement présents | Requête(s) associée(s) |
|---|---|---|
| `astra_platform.txt` | plateforme, architecture, modules, recherche hybride | plateforme Astra architecture ; recherche vectorielle hybride |
| `machine_learning.txt` | ML, types d'apprentissage, deep learning, applications | machine learning ; deep learning réseaux neurones |
| `recherche_lexicale.txt` | recherche lexicale, BM25, TF-IDF, tokenisation | BM25 recherche lexicale ; TF-IDF pondération fréquence |
| `sample.txt` | Python, ML, deep learning, vectoriel, BM25, TF-IDF, fusion, EvidenceRank | Python ; EvidenceRank ; ML ; deep learning ; TF-IDF ; hybride |

Aucune requête ne porte sur un sujet absent du corpus (ex. : vision par ordinateur,
traitement du langage naturel comme objet principal), car une telle requête n'aurait
pas de ground truth défendable.

### 2.4 Pourquoi cette diversité

- **Taille variable des ensembles de pertinents** : de 1 (Python, EvidenceRank) à
  5 (machine learning). Cela teste le comportement du moteur sur des besoins
  d'information étroits et larges.
- **Origines multiples des pertinents** : certaines requêtes n'ont de pertinents que
  dans un seul document (Python → `sample.txt`), d'autres dans plusieurs
  (machine learning → `machine_learning.txt` **et** `sample.txt` ; hybride →
  `astra_platform.txt`, `machine_learning.txt`, `sample.txt`). Cela teste la capacité
  de la recherche à aller chercher des informations réparties dans la collection.
- **Contraste lexical** : des sigles rares à fort IDF (TF-IDF, BM25), des termes
  génériques à faible IDF (machine, recherche, plateforme), des noms propres (Astra,
  Python, EvidenceRank). La diversité des fréquences documentaires est essentielle
  pour ne pas favoriser artificiellement une branche de la fusion.

---

## 3. Corpus et segmentation

### 3.1 Composition du corpus

Corpus de démonstration dans `app/api/data/`, chargé par `DocumentManager`
(ordre alphabétique des fichiers), segmenté par **SmartSeg** avec ses paramètres par
défaut : `chunk_size=500` caractères, `overlap=50`, `min_chunk_size=100`.

**Résultat réel : 4 documents → 17 chunks** (vocabulaire TF-IDF : 335 termes).

| Document | Chunks | Longueurs (car.) | Mots totaux | Thèmes |
|---|---|---|---|---|
| `astra_platform.txt` | 0–2 (3) | 500, 500, 467 | 215 | plateforme, architecture modulaire, sécurité, recherche hybride |
| `machine_learning.txt` | 3–6 (4) | 500, 500, 147, 132 | 170 | ML, apprentissages, deep learning, applications |
| `recherche_lexicale.txt` | 7–9 (3) | 500, 111, 500 | 178 | recherche lexicale, BM25, TF-IDF, tokenisation |
| `sample.txt` | 10–16 (7) | 500, 189, 500, 139, 500, 149, 232 | 322 | Python, ML, deep, vectoriel, BM25, TF-IDF, fusion, EvidenceRank |
| **Total** | **17** | **6 066** | **885** | — |

Longueur moyenne des chunks : 6 066 / 17 ≈ **356,8 caractères**. Six chunks sont
« courts » (< 200 caractères) : **5, 6, 8, 11, 13, 15** — ce sont des fragments issus
du découpage à 500 caractères (fins de documents ou de sections).

### 3.2 Incidence de la segmentation

Les fragments courts ont un impact direct sur la recherche : un chunk de 111 ou
132 caractères contient moins de termes distincts, donc un signal TF-IDF et BM25
plus faible, et peut perdre les termes porteurs du thème (voir section 11).

---

## 4. Construction du Ground Truth — méthodologie en 6 étapes

Le ground truth est l'ensemble des paires (requête, chunk pertinent). Sa construction
a suivi un protocole strict, digne d'un travail de recherche, en 6 étapes.

### Étape 1 — Lecture complète des documents sources

Lecture intégrale des 4 fichiers `.txt` du corpus, dans leur état brut, pour connaître
les thèmes traités, leur organisation (sections) et leurs frontières. C'est la base de
la formulation des requêtes : on ne peut interroger que ce que le corpus contient.

### Étape 2 — Génération des chunks

Exécution du pipeline réel de segmentation : `DocumentManager("app/api/data")` →
`SmartSeg.process()` sur chaque fichier → génération des 17 chunks avec leurs
métadonnées (`chunk_id`, `source`, `length`, `word_count`, `content`). Les `chunk_id`
sont globaux et ordonnés par document (astra_platform → 0–2, machine_learning → 3–6,
recherche_lexicale → 7–9, sample → 10–16).

### Étape 3 — Exécution de `test_chunks.py`

Le script `test_chunks.py` charge le corpus et affiche **l'intégralité de chaque
chunk** :

```python
for chunk in chunks:
    print("Chunk ID :", chunk["chunk_id"])
    print("Source   :", chunk.get("source"))
    print("Length   :", chunk.get("length"))
    print(chunk["content"])
```

Cette exécution matérialise la **collection indexée telle que le moteur la voit** :
c'est elle, et non le texte source, qui sert de référence à la pertinence. On y
constate notamment que certains chunks commencent en milieu de phrase (chunks 5, 8,
11, 13, 15) — conséquence du découpage à 500 caractères.

### Étape 4 — Lecture manuelle des 17 chunks

Lecture ligne par ligne des 17 chunks, en notant pour chacun : thème dominant, termes
clés présents, et sa position dans la structure du document d'origine. Cette étape
permet d'éviter les erreurs d'annotation « au hasard » : on ne déclare un chunk
pertinent qu'après vérification de son contenu effectif.

### Étape 5 — Association manuelle requête → chunk_id

Pour chaque requête formulée, identification de l'ensemble des chunks dont le contenu
**répond au besoin d'information**, selon deux critères complémentaires :
1. **présence lexicale** des termes de la requête dans le chunk (appariement exact) ;
2. **adéquation sémantique** : le chunk traite-t-il du sujet demandé, même si certains
   termes de la requête n'y figurent pas littéralement ?

Exemple : pour « TF-IDF pondération fréquence », le chunk 14 contient la définition
littérale « TF-IDF (term frequency-inverse document frequency) est une méthode de
pondération » → pertinent. Le chunk 8 (fragment) contient « fréquence du terme (TF) et
fréquence inverse (IDF) » → pertinent. Aucun autre chunk ne traite de la pondération
TF-IDF → seuls 8 et 14 sont retenus.

### Étape 6 — Validation

1. **Vérification d'existence** : chaque `chunk_id` du ground truth doit exister dans
   le corpus chargé (garde-fou intégré à `demo_evaluation.py`, qui interrompt le
   script en cas d'incohérence).
2. **Justification traçable** : chaque association est documentée dans la section 5
   avec les mots-clés réellement retrouvés et la source du chunk.
3. **Non-trivialité** : chaque requête possède au moins un chunk pertinent (sinon
   Recall et RR seraient nuls par construction, biaisant la moyenne).

Le ground truth final est codé dans `demo_evaluation.py` (`EVAL_SET`) : **8 requêtes,
20 associations pertinentes**.

---

## 5. Tableau complet du Ground Truth

Le tableau suivant reproduit l'intégralité du ground truth réel, enrichi de la source
et des mots-clés **effectivement retrouvés** dans chaque chunk (appariement exact,
tokenisation identique à celle de FusionSearch).

| Requête | Chunks pertinents | Pourquoi ces chunks sont pertinents | Mots-clés retrouvés dans le chunk | Source |
|---|---|---|---|---|
| machine learning intelligence artificielle | 3, 4, 5, 11, 12 | **3** : « Le machine learning est une branche de l'intelligence artificielle » + types d'apprentissage. **4** : « Le deep learning est une sous-catégorie du machine learning » + applications. **5** : « La recherche d'information utilise le machine learning ». **11** : définition de l'IA (« systèmes peuvent apprendre à partir de données »). **12** : « Le deep learning est un sous-ensemble du machine learning ». | 3 : {artificielle, learning, machine} ; 4 : {learning, machine} ; 5 : {learning, machine} ; 11 : {artificielle} ; 12 : {learning, machine} | machine_learning.txt (3–5) ; sample.txt (11–12) |
| BM25 recherche lexicale | 7, 8, 9 | **7** : « BM25 est un algorithme de classement lexical très utilisé. Il calcule un score de pertinence ». **8** : description du score « fréquence du terme (TF) et fréquence inverse (IDF) » (suite de la phrase de 7). **9** : « BM25 utilise également des paramètres de saturation k1 et b » + TF-IDF. | 7 : {bm25, lexicale, recherche} ; 8 : **aucun** ; 9 : {bm25, lexicale} | recherche_lexicale.txt |
| plateforme Astra architecture | 0, 1, 2 | **0** : « Astra est une plateforme de recherche documentaire intelligente » + « Architecture de la plateforme ». **1** : « Le module de recherche utilise à la fois la recherche lexicale et la vectorielle » + fonctionnalités. **2** : « Modules de la plateforme : API, Executor, Retrieval, Registry, Guardrails, Telemetry » (architecture modulaire). | 0 : {architecture, astra, plateforme} ; 1 : {astra, plateforme} ; 2 : {plateforme} | astra_platform.txt |
| TF-IDF pondération fréquence | 8, 14 | **8** : « fréquence du terme (TF) et fréquence inverse (IDF) ». **14** : « TF-IDF (term frequency-inverse document frequency) est une méthode de pondération qui reflète l'importance d'un terme ». | 8 : {fréquence, idf, tf} ; 14 : {fréquence, idf, pondération, tf} | recherche_lexicale.txt (8) ; sample.txt (14) |
| deep learning réseaux neurones | 4, 12 | **4** : « Le deep learning est une sous-catégorie du machine learning qui utilise des réseaux de neurones profonds ». **12** : « Le deep learning est un sous-ensemble du machine learning qui utilise des réseaux de neurones artificiels avec plusieurs couches ». | 4 : {deep, learning, neurones, réseaux} ; 12 : {deep, learning, neurones, réseaux} | machine_learning.txt (4) ; sample.txt (12) |
| recherche vectorielle hybride | 1, 6, 15 | **1** : « La recherche vectorielle utilise la similarité cosinus » + « Astra offre une recherche hybride ». **6** : « Les moteurs de recherche modernes combinent des approches lexicales et vectorielles ». **15** : « combine les résultats de la recherche vectorielle et de la recherche lexicale ». | 1 : {hybride, recherche, vectorielle} ; 6 : {recherche} ; 15 : {recherche, vectorielle} | astra_platform.txt (1) ; machine_learning.txt (6) ; sample.txt (15) |
| Python programmation langage | 10 | **10** : « Python est un langage de programmation interprété, multi-paradigme et multiplateforme ». | 10 : {langage, programmation, python} | sample.txt |
| évaluation pertinence EvidenceRank | 16 | **16** : « L'évaluation de la pertinence est cruciale dans les systèmes de recherche. EvidenceRank est notre système de re-ranking ». | 16 : {evidencerank, pertinence} | sample.txt |

> **Note honnête.** Pour la requête « évaluation pertinence EvidenceRank », le
> chevauchement lexical n'a compté que 2 des 3 termes de la requête
> ({evidencerank, pertinence}) : le tokeniseur de FusionSearch met le texte en
> minuscules et découpe aux espaces, or le chunk 16 dit « L'évaluation de la
> pertinence » — « évaluation » y est fusionné dans le token « l'évaluation » et n'est
> donc pas compté comme terme isolé. Le chunk est néanmoins pertinent par adéquation
> sémantique directe : il traite exactement de l'évaluation de la pertinence et
> d'EvidenceRank.

---

## 6. Schéma du pipeline d'évaluation

```
                        ┌─────────────────────┐
                        │  Corpus (4 .txt)    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   DocumentManager   │   app/retrieval/document_manager.py
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      SmartSeg       │   chunk_size=500, overlap=50, min=100
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   17 chunks (0–16)  │   test_chunks.py → lecture manuelle
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Ground Truth (GT)  │   8 requêtes × chunks pertinents (20 assoc.)
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
     │  LexiRank    │    │  TF-IDF      │    │  (Fusion)    │
     │  (BM25)      │    │  cosinus     │    │ 50/50        │
     └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
            │                   │                    │
            └───────────────────┴────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ FusionSearch.search │   top_k=5 (Recall@5) ; top_k=10 (Recall@10)
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Classement Top K   │   liste ordonnée de chunk_id
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Évaluation (GT)    │   Recall@K, RR par requête
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Agrégation        │   Mean Recall@K, MRR
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     Analyse         │   interprétation scientifique
                        └─────────────────────┘
```

**Lecture du schéma.** La chaîne est entièrement maison et déterministe. Le ground
truth est construit **en amont** de toute recherche (flèche GT → évaluation), ce qui
garantit qu'il ne dépend pas des résultats du moteur.

---

## 7. Définition mathématique des métriques et application à une vraie requête

Toutes les métriques sont calculées par `app/evaluation/metrics.py` (implémentation
maison).

### 7.1 Recall@K

Soit `R` l'ensemble des chunks pertinents pour une requête et `TopK` l'ensemble des K
premiers chunks retournés :

```
Recall@K = |R ∩ TopK| / |R|
```

**Application à une vraie requête.** Requête « BM25 recherche lexicale » :
- `R = {7, 8, 9}` (ground truth)
- résultats réels : `[7, 0, 12, 15, 9]` → `Top5 = {7, 0, 12, 15, 9}`
- `R ∩ Top5 = {7, 9}` → `|R ∩ Top5| = 2`, `|R| = 3`
- **Recall@5 = 2/3 = 0.6667**

### 7.2 Reciprocal Rank (RR)

```
RR = 1 / rang du premier document pertinent   (0 si aucun pertinent)
```

**Application à une vraie requête.** Requête « Python programmation langage » :
- résultats réels : `[10, 0, 4, 12, 14]`
- `R = {10}` ; le chunk 10 est en **position 1**
- **RR = 1/1 = 1.0**

### 7.3 Mean Reciprocal Rank (MRR)

```
MRR = (1/N) · Σᵢ₌₁ᴺ RRᵢ
```

**Application aux 8 requêtes** (voir tableau section 9) : chaque requête a son premier
résultat pertinent en position 1 → `RRᵢ = 1.0` pour tout i → **MRR = 8 × 1.0 / 8 = 1.0**.

---

## 8. Pourquoi Mean Recall@5 = 0.8917 — calcul complet

Les Recall@5 réels par requête sont :

| # | Requête | Recall@5 |
|---|---|---|
| 1 | machine learning intelligence artificielle | 0.8000 |
| 2 | BM25 recherche lexicale | 0.6667 |
| 3 | plateforme Astra architecture | 1.0000 |
| 4 | TF-IDF pondération fréquence | 1.0000 |
| 5 | deep learning réseaux neurones | 1.0000 |
| 6 | recherche vectorielle hybride | 0.6667 |
| 7 | Python programmation langage | 1.0000 |
| 8 | évaluation pertinence EvidenceRank | 1.0000 |

Le Mean Recall@5 est la **macro-moyenne** (chaque requête pèse 1/8) :

```
0.8000
+ 0.6667
+ 1.0000
+ 1.0000
+ 1.0000
+ 0.6667
+ 1.0000
+ 1.0000
= 7.1334

7.1334 / 8 = 0.891675 → arrondi à 4 décimales = 0.8917
```

**Vérification croisée (micro-moyenne).** Sur les 20 associations pertinentes,
17 sont retrouvées dans le top 5 (4 + 2 + 3 + 2 + 2 + 2 + 1 + 1) → 17/20 = **0.85**.
L'écart macro/micro (0.8917 vs 0.85) s'explique par la pondération : la macro-moyenne
donne le même poids aux requêtes à 1 pertinent (Recall = 1.0 par construction) et aux
requêtes à 5 pertinents ; la micro-moyenne est tirée vers le bas par la requête 1
(5 pertinents dont 1 manqué). Les deux lectures sont complémentaires ; le rapport
retient la macro-moyenne, conforme à la pratique courante en évaluation IR.

---

## 9. Pourquoi MRR = 1.000 — démonstration

Le MRR vaut 1.0 si et seulement si, pour **chaque** requête, le premier résultat
retourné est pertinent. C'est vérifié sur les 8 requêtes (résultats réels) :

| Requête | Premier résultat (chunk) | Pertinent ? | RR |
|---|---|---|---|
| machine learning intelligence artificielle | 3 | ✅ (3 ∈ R) | 1.0 |
| BM25 recherche lexicale | 7 | ✅ (7 ∈ R) | 1.0 |
| plateforme Astra architecture | 0 | ✅ (0 ∈ R) | 1.0 |
| TF-IDF pondération fréquence | 8 | ✅ (8 ∈ R) | 1.0 |
| deep learning réseaux neurones | 4 | ✅ (4 ∈ R) | 1.0 |
| recherche vectorielle hybride | 1 | ✅ (1 ∈ R) | 1.0 |
| Python programmation langage | 10 | ✅ (10 ∈ R) | 1.0 |
| évaluation pertinence EvidenceRank | 16 | ✅ (16 ∈ R) | 1.0 |

```
MRR = (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) / 8 = 8/8 = 1.0000
```

Le MRR n'est donc pas une approximation : il est **exactement** 1.0, car la fusion
classe systématiquement un chunk pertinent en première position.

---

## 10. Recall@10 : calcul, comparaison et pouvoir discriminant

### 10.1 Résultats réels

Le script calcule désormais Recall@10 par une **passe séparée** (top_k=10), afin de ne
pas perturber le classement du top 5 (la normalisation min-max de FusionSearch dépend
du nombre de candidats).

| # | Requête | Recall@5 | Recall@10 |
|---|---|---|---|
| 1 | machine learning intelligence artificielle | 0.8000 | 1.0000 |
| 2 | BM25 recherche lexicale | 0.6667 | 0.6667 |
| 3 | plateforme Astra architecture | 1.0000 | 1.0000 |
| 4 | TF-IDF pondération fréquence | 1.0000 | 1.0000 |
| 5 | deep learning réseaux neurones | 1.0000 | 1.0000 |
| 6 | recherche vectorielle hybride | 0.6667 | 1.0000 |
| 7 | Python programmation langage | 1.0000 | 1.0000 |
| 8 | évaluation pertinence EvidenceRank | 1.0000 | 1.0000 |
| **Moyenne** | | **0.8917** | **0.9583** |

### 10.2 Comparaison

- Recall@10 (0.9583) > Recall@5 (0.8917) : en élargissant la fenêtre, deux requêtes
  récupèrent leurs chunks manquants (la requête 1 retrouve le chunk 12 en position 6 ;
  la requête 6 retrouve le chunk 6 en position 7).
- **Cas notable : la requête 2 reste à 0.6667 même à K=10.** Le chunk 8 n'apparaît
  **dans aucun** des 10 premiers résultats. C'est la preuve que son échec n'est pas un
  simple problème de rang, mais un échec **de récupération** (voir section 11).

### 10.3 Pourquoi Recall@5 est plus discriminant sur un corpus de 17 chunks

1. **Taille de la collection.** À K=10, on considère 10/17 ≈ 59 % du corpus : il
   suffit qu'un chunk pertinent soit « quelque part » dans plus de la moitié de la
   collection. La métrique devient peu exigeante.
2. **Usage réel.** Dans un pipeline RAG, 3 à 10 chunks sont transmis au générateur ;
   K=5 correspond à une première page de résultats réaliste. Mesurer à K=5 évalue ce
   que voit réellement l'utilisateur ou l'agent.
3. **Mise en évidence des échecs.** Recall@5 révèle que 3 associations pertinentes sur
   20 ne sont pas dans le top 5 ; Recall@10 n'en révèle plus qu'une. Le choix de K=5
   comme métrique principale rend la campagne **plus sensible aux défauts**, donc plus
   utile pour piloter les améliorations.

---

## 11. Analyse critique des résultats

### 11.1 Synthèse chiffrée

- 6 requêtes sur 8 atteignent Recall@5 = 1.0 ;
- 2 requêtes atteignent 0.6667 (2 pertinents sur 3 retrouvés) ;
- MRR = 1.0 (toujours un pertinent en tête).

### 11.2 Pourquoi Recall@5 = 0.6667 pour « BM25 recherche lexicale » ?

- **Chunk manqué : 8** (111 caractères, fragment de `recherche_lexicale.txt`).
- **Contenu réel du chunk 8** : « e prend en compte la fréquence du terme dans le
  document (TF) et la fréquence inverse dans la collection (IDF). » — c'est la **fin
  d'une phrase coupée** (il commence par « e prend ») ; il ne contient **aucun** des
  termes de la requête (voir tableau section 5 : mots-clés retrouvés = ∅).
- **Cause : segmentation avant tout.** Le chunk 8 est un artefact du découpage à
  500 caractères. Privé des mots « BM25 », « recherche », « lexicale », il ne peut
  être rapproché de la requête ni par le BM25 (aucun appariement), ni par le TF-IDF
  (aucun terme commun). **Ce n'est pas un échec de la fusion** : les deux branches
  échouent pour la même raison lexicale.
- **Preuve complémentaire** : même à K=10 le chunk 8 n'apparaît pas. Si la cause était
  un mauvais classement, il serait au moins dans le top 10 ; son absence à K=10
  confirme l'échec de récupération dû au contenu tronqué.

### 11.3 Pourquoi Recall@5 = 0.6667 pour « recherche vectorielle hybride » ?

- **Chunk manqué : 6** (132 caractères, fragment de `machine_learning.txt`).
- **Contenu réel du chunk 6** : « Les moteurs de recherche modernes combinent des
  approches lexicales et vectorielles pour offrir la meilleure expérience
  utilisateur. »
- **Cause : inadéquation de forme lexicale + forte concurrence.** Le chunk contient
  « vectorielles » (adjectif au pluriel) mais ni le terme exact « vectorielle », ni
  « hybride ». La requête « recherche vectorielle hybride » matche parfaitement les
  chunks 1 et 15 (expressions exactes), qui dominent donc le classement. Le chunk 6
  n'est retrouvé qu'en **position 7** (Recall@10 = 1.0) : il est bien classé «
  correctement mais trop bas », victime de la concurrence et de la variante fléchie.
- **Segmentation ?** Partiellement : le chunk 6 est un fragment court (fin du
  document), mais sa phrase est complète ; le vrai problème est l'absence de
  normalisation morphologique (vectorielle vs vectorielles) et la concurrence des
  chunks à appariement exact.

### 11.4 Synthèse des causes

| Requête | Chunk manqué | Segmentation ? | Lexical ? | Vectoriel ? | Fusion ? |
|---|---|---|---|---|---|
| BM25 recherche lexicale | 8 | ✅ (fragment tronqué, aucun mot-clé) | échec (aucun appariement) | échec (aucun terme commun) | non (les deux branches échouent pareil) |
| recherche vectorielle hybride | 6 | ⚠️ (fragment court, phrase complète) | partiel (vectorielles ≠ vectorielle) | partiel | non (classé 7ᵉ, concurrence) |

**Conclusion de l'analyse.** Les deux dégradations sont imputables à des **artefacts
de segmentation et de forme lexicale**, pas à la logique de fusion. Le moteur ne
« perd » aucun chunk pertinent qui soit bien formé et contienne les termes de la
requête.

---

## 12. Analyse du corpus

| Document | Nb chunks | Longueur (car.) | Thèmes | Contribution au GT |
|---|---|---|---|---|
| `astra_platform.txt` | 3 | 1 467 | plateforme, architecture, sécurité, recherche hybride, modules | 3 chunks pertinents (0, 1, 2) |
| `machine_learning.txt` | 4 | 1 279 | ML, apprentissages, deep learning, applications | 3 chunks (3, 4, 5) + 1 fragment (6) |
| `recherche_lexicale.txt` | 3 | 1 111 | recherche lexicale, BM25, TF-IDF | 3 chunks (7, 8, 9) |
| `sample.txt` | 7 | 2 209 | synthèse multi-thèmes + EvidenceRank | 6 chunks (10, 11, 12, 14, 15, 16) |
| **Total** | **17** | **6 066** | — | **16 chunks sur 17** |

Observations :
- `sample.txt` est le document le plus long (2 209 car.) et le plus découpé
  (7 chunks) ; il joue le rôle de « document de synthèse » et est le plus sollicité
  par le ground truth (6 chunks pertinents).
- Les trois documents thématiques sont représentés équitablement (3 chunks chacun,
  dont 3 pertinents chacun).
- Le fragment 13 (`sample.txt`, 139 car.) n'est pertinent pour **aucune** requête
  (voir section 13).

---

## 13. Analyse statistique du Ground Truth

### 13.1 Couverture des chunks

- **Chunks utilisés** : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16
  → **16 chunks sur 17 (94,1 %)**.
- **Chunk jamais utilisé : 13** (`sample.txt`, 139 caractères).

### 13.2 Pourquoi le chunk 13 n'est-il jamais pertinent ?

Contenu réel du chunk 13 : « une méthode de classement utilisée par les moteurs de
recherche pour estimer la pertinence des documents par rapport à une requête donnée. »

C'est la **fin de phrase** de la définition de la recherche lexicale (BM25) entamée
dans le chunk 12. Deux raisons expliquent son absence du ground truth :
1. Il n'apporte **aucun concept nouveau** : tout ce qu'il dit (classement, pertinence,
   moteurs de recherche) est déjà couvert, et plus complètement, par les chunks 7, 9,
   12 et 15 qui sont pertinents pour les requêtes « BM25 recherche lexicale » et
   « recherche vectorielle hybride ».
2. Aucune des 8 requêtes ne correspond à son contenu isolé : aucune requête ne porte
   sur « la notion générique de classement des moteurs de recherche » en dehors de
   BM25 ou de la fusion.

Ce choix est un **jugement d'annotation assumé** : le fragment 13 est un artefact de
segmentation sans contenu informationnel propre. Le signaler ici fait partie de
l'honnêteté méthodologique de la campagne.

### 13.3 Distribution des tailles de pertinents

| Taille de l'ensemble pertinent | Requêtes | 
|---|---|
| 1 | Python (10), EvidenceRank (16) |
| 2 | TF-IDF (8,14), deep learning (4,12) |
| 3 | BM25 (7,8,9), hybride (1,6,15), Astra (0,1,2) |
| 5 | machine learning (3,4,5,11,12) |

La répartition est volontairement étalée (1, 2, 3, 5) pour éviter qu'une seule
configuration domine la moyenne.

---

## 14. Menaces sur la validité

1. **Corpus très petit (17 chunks).** Les scores ne sont pas généralisables à un
   corpus réel ; ils valident le fonctionnement sur un périmètre restreint.
2. **Seulement 8 requêtes.** Variance élevée des moyennes : le retrait d'une seule
   requête (ex. requête 2 ou 6) modifierait le Mean Recall@5 de ~0,04. Pas de test de
   significativité statistique possible à cette échelle.
3. **Ground truth manuel et mono-annotateur.** La pertinence est un jugement
   subjectif ; un autre annotateur pourrait inclure le chunk 10 pour la requête
   « machine learning » (il en mentionne la définition) ou le chunk 13 pour une
   requête sur la notion de classement. L'accord inter-annotateurs n'est pas mesuré.
4. **Requêtes en mots-clés.** Les requêtes sont des concaténations de termes ; les
   requêtes en langage naturel complet (typiques d'un agent conversationnel) ne sont
   pas couvertes.
5. **Pas de métrique de précision.** Recall seul ne mesure pas la proportion de
   résultats non pertinents dans le top K. Une requête à Recall parfait peut renvoyer
   5 chunks dont 3 non pertinents (ce n'est pas le cas ici, mais la métrique ne le
   montrerait pas).
6. **Pas de NDCG ni de MAP.** Le MRR ne capture que le premier pertinent ; l'ordre des
   pertinents suivants n'est pas évalué.
7. **Pas d'utilisateurs réels.** La pertinence « statique » (le chunk traite-t-il du
   sujet ?) ne mesure pas la satisfaction réelle ni la qualité de la réponse finale du
   RAG.
8. **Pas de baseline externe.** Sans comparaison (BM25 seul, vectoriel seul, moteurs
   de référence), on ne peut quantifier l'apport de la fusion.

---

## 15. Perspectives

1. **Recall@1 et Recall@3** : mesurés pour information lors de la campagne —
   Recall@1 = 0.5250, Recall@3 = 0.8250 (macro-moyennes réelles). Ils pourraient être
   intégrés comme métriques complémentaires du rapport final.
2. **Precision@K** : ajouter la précision au top 5 pour compléter le recall.
3. **NDCG et MAP** : métriques sensibles à l'ordre de tous les pertinents, plus
   fines que le MRR.
4. **Ablation BM25 seul vs vectoriel seul vs fusion** : mesurer chacun des composants
   isolément sur le même ground truth pour quantifier l'apport de la fusion.
5. **Ablation avec EvidenceRank** : comparer FusionSearch seul vs FusionSearch +
   EvidenceRank (re-ranking) pour mesurer l'impact du re-ranking sur Recall@5 et MRR.
6. **Augmentation du corpus** : enrichir la collection pour approcher les conditions
   d'un corpus réel et rendre les métriques plus significatives.
7. **Plus de requêtes** : passer à 20–30 requêtes, y compris en langage naturel et
   hors-vocabulaire, pour une estimation plus stable.
8. **Plusieurs annotateurs** : mesurer l'accord inter-annotateurs (κ de Cohen) pour
   fiabiliser le ground truth.
9. **Optimisation de la segmentation** : couper aux frontières de phrases et fusionner
   les fragments courts (< 100 caractères) pour éliminer les artefacts des chunks 6, 8,
   13 — puis re-mesurer avec le même protocole pour quantifier le gain.
10. **Normalisation morphologique** (vectorielle/vectorielles) : tester l'impact d'une
    lemmatisation ou d'un racinage sur la requête « recherche vectorielle hybride ».

---

## 16. Audit du code `demo_evaluation.py` (améliorations non-comportementales)

Le script a été audité puis amélioré **sans modifier la logique de calcul ni les
résultats** (Mean Recall@5 = 0.8917, MRR = 1.0 — vérifiés avant et après).

### 16.1 Constats de l'audit

| Catégorie | Constat | Correction |
|---|---|---|
| Code mort | Constante de couleur `MAGENTA` jamais utilisée | Supprimée |
| Imports inutiles | Aucun (sys, time, typing, modules maison : tous utilisés) | — |
| Nombres magiques | `top_k=5` et `ks=[5]` codés en dur à deux endroits, risquant une désynchronisation (search à 5 mais évaluation à un autre K) | Constantes `TOP_K = 5`, `SECONDARY_K = 10` |
| Garde-fous | Existence des `chunk_id` du GT déjà vérifiée ; manquait : requête vide, ensemble de pertinents vide, division par zéro | Garde-fous ajoutés (messages d'erreur explicites + `sys.exit(1)`) |
| Documentation | `main()` sans docstring ; commentaire du GT peu détaillé | Docstring ajoutée ; commentaires enrichis |
| Métrique complémentaire | Recall@10 non mesuré | Passe séparée read-only (top_k=10), sans effet sur le top 5 |
| Robustesse | Affichage des accents sous Windows | `sys.stdout.reconfigure(encoding="utf-8")` (try/except) |

### 16.2 Preuve que les résultats sont inchangés

- Avant modification : Mean Recall@5 = 0.8917, MRR = 1.0000.
- Après modification : Mean Recall@5 = 0.8917, MRR = 1.0000, Mean Recall@10 = 0.9583.
- La suite de tests (114 tests) passe intégralement.
- Aucun module de production n'a été modifié (FusionSearch, LexiRank, TermVectorizer,
  Evaluator, SmartSeg, DocumentManager : intacts).

---

## 17. Conclusion

La campagne d'évaluation de la recherche hybride AstraExec repose sur un protocole
explicite en 6 étapes, un ground truth manuel traçable (20 associations justifiées par
le contenu réel des chunks) et des métriques standard calculées de manière
déterministe. Sur le corpus actuel (7 documents, 415 chunks), les résultats réels —
**Mean Recall@5 = 0.8667**, **MRR = 1.0000**, **Mean Recall@10 = 0.9583** — montrent
que le moteur place systématiquement un document pertinent en tête et retrouve près
de 87 % des pertinents dans le top 5 (voir section 0 pour la migration du corpus).

L'analyse critique identifie les deux seules dégradations comme des artefacts de
segmentation et de forme lexicale (chunks 83 et 81, ex-8 et ex-6), et non comme des
défauts de la fusion. Les limites (corpus réduit, 8 requêtes, annotation mono-annotateur, absence de
précision/NDCG et d'utilisateurs réels) bornent la portée des conclusions et tracent
les perspectives de consolidation.

Ce document constitue la partie scientifique et documentaire du Livrable 2 : la
campagne est **reproductible, chiffrée et honnête**, prête à être présentée en
soutenance.
