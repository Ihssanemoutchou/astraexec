"""
AstraExec — Script d'Évaluation Automatique
============================================

Évalue la qualité de la recherche hybride (FusionSearch) sur le corpus
de démonstration, en calculant Recall@K et Mean Reciprocal Rank.

Usage :
    python demo_evaluation.py
"""

import sys
import time
from typing import List, Dict, Set

# ── Modules maison ──────────────────────────────────────────────────
from app.retrieval.document_manager import DocumentManager
from app.retrieval.fusion_search import FusionSearch
from app.evaluation.metrics import Evaluator, recall_at_k, reciprocal_rank

# ── Paramètres de la campagne ─────────────────────────────────────
TOP_K = 5        # Rang principal : Recall@5 (première page de résultats)
SECONDARY_K = 10 # Rang secondaire : Recall@10 (analyse de sensibilité)

# ── Couleurs ANSI ──────────────────────────────────────────────────
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
SEP = f"{BOLD}{'=' * 72}{RESET}"


def print_step(label: str):
    print(f"  {BLUE}>{RESET}  {label}")


def print_ok():
    print(f"  {GREEN}OK{RESET}")


def print_info(text: str):
    print(f"    {DIM}{text}{RESET}")


# ════════════════════════════════════════════════════════════════════
#  Définition du jeu d'évaluation
# ════════════════════════════════════════════════════════════════════

# Queries et IDs pertinents attendus (établis à partir du corpus réel).
# Les chunk_id ci-dessous correspondent aux chunks générés par
# DocumentManager depuis le dossier app/api/data/.
# Chaque association est justifiée par le contenu du chunk dans
# docs/campagne_evaluation.md (section « Construction du Ground Truth »).
#
# Sources et thèmes des chunks :
#   astra_platform.txt  → chunks 0, 1, 2   (plateforme, vectoriel, sécurité)
#   machine_learning.txt → chunks 3, 4, 5, 6 (ML intro, RL, applications)
#   recherche_lexicale.txt → chunks 7, 8, 9 (lexical, TF/IDF, BM25)
#   sample.txt  → chunks 10-16 (Python, AI, deep, BM25, TF-IDF, hybride, EvidenceRank)

EVAL_SET: List[Dict] = [
    {
        "query": "machine learning intelligence artificielle",
        "relevant": {3, 4, 5, 11, 12},
    },
    {
        "query": "BM25 recherche lexicale",
        "relevant": {7, 8, 9},
    },
    {
        "query": "plateforme Astra architecture",
        "relevant": {0, 1, 2},
    },
    {
        "query": "TF-IDF pondération fréquence",
        "relevant": {8, 14},
    },
    {
        "query": "deep learning réseaux neurones",
        "relevant": {12, 4},
    },
    {
        "query": "recherche vectorielle hybride",
        "relevant": {1, 15, 6},
    },
    {
        "query": "Python programmation langage",
        "relevant": {10},
    },
    {
        "query": "évaluation pertinence EvidenceRank",
        "relevant": {16},
    },
]


# ════════════════════════════════════════════════════════════════════
#  Affichage détaillé par requête
# ════════════════════════════════════════════════════════════════════

def print_query_report(
    query: str,
    expected: Set[int],
    predicted_ids: List[int],
    recall_k: float,
    recall_10: float,
    rr: float,
):
    """Affiche un rapport lisible pour une requête."""
    print(SEP)
    print(f"  {BOLD}Requête   :{RESET} {query}")
    print(f"  {BOLD}Attendus  :{RESET} {sorted(expected)}")
    print(f"  {BOLD}Prédits   :{RESET} {predicted_ids}")
    print(f"  {BOLD}Recall@5  :{RESET} {recall_k:.4f}")
    print(f"  {BOLD}Recall@10 :{RESET} {recall_10:.4f}")
    print(f"  {BOLD}RR        :{RESET} {rr:.4f}")


def print_final_report(
    total_queries: int,
    mean_recall_5: float,
    mean_recall_10: float,
    mrr: float,
    elapsed: float,
):
    """Affiche le récapitulatif final."""
    print()
    print(SEP)
    print(f"  {BOLD}{CYAN}RÉSULTATS GLOBAUX{RESET}")
    print(f"  {DIM}{'-' * 68}{RESET}")
    print(f"  Requêtes évaluées  : {total_queries}")
    print(f"  Mean Recall@5      : {mean_recall_5:.4f}")
    print(f"  Mean Recall@10     : {mean_recall_10:.4f}")
    print(f"  MRR                : {mrr:.4f}")
    print(f"  Temps total        : {elapsed:.2f}s")
    print(SEP)


# ════════════════════════════════════════════════════════════════════
#  Pipeline d'évaluation
# ════════════════════════════════════════════════════════════════════

def main():
    """Exécute la campagne d'évaluation (corpus → Recall@5 / Recall@10 / MRR)."""
    # Force l'UTF-8 sur la sortie console (affichage correct des accents
    # sous Windows, dont la console par défaut utilise cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # certains environnements ne permettent pas la reconfiguration

    print(f"""
  {BOLD}{CYAN}+----------------------------------------------------+{RESET}
  {BOLD}{CYAN}|  AstraExec - Évaluation de la Recherche Hybride    |{RESET}
  {BOLD}{CYAN}|  Recall@K & Mean Reciprocal Rank                   |{RESET}
  {BOLD}{CYAN}+----------------------------------------------------+{RESET}
""")

    start_total = time.time()

    # ── 1. Chargement du corpus ─────────────────────────────────────
    print_step("Chargement du corpus depuis app/api/data/ ...")
    dm = DocumentManager("app/api/data")
    chunks = dm.load_documents()
    print_ok()
    print_info(f"{len(chunks)} chunks chargés")

    # Garde-fou : tout chunk_id du ground truth doit exister dans le corpus.
    # Sinon, le Recall@5 serait artificiellement bas (pertinents introuvables).
    existing_ids = {c["chunk_id"] for c in chunks}
    unknown_ids = sorted(
        {rid for item in EVAL_SET for rid in item["relevant"]} - existing_ids
    )
    if unknown_ids:
        print(f"  {RED}{BOLD}ERREUR{RESET} : chunk_id inconnus dans le corpus : "
              f"{unknown_ids}")
        print(f"  {RED}Le ground truth doit être aligné sur le corpus chargé.{RESET}")
        sys.exit(1)

    # Garde-fou : chaque requête doit être non vide et posséder au moins
    # un chunk pertinent, sinon Recall@K et RR seraient nuls par défaut.
    for item in EVAL_SET:
        if not item["query"].strip():
            print(f"  {RED}{BOLD}ERREUR{RESET} : requête vide dans EVAL_SET.{RESET}")
            sys.exit(1)
        if not item["relevant"]:
            print(f"  {RED}{BOLD}ERREUR{RESET} : aucun chunk pertinent pour : "
                  f"{item['query']}")
            sys.exit(1)

    # ── 2. Construction de l'index hybride ──────────────────────────
    print()
    print_step("Construction de l'index FusionSearch ...")
    fusion = FusionSearch()
    fusion.build_index(chunks)
    print_ok()

    info = fusion.info()
    print_info(f"Vocabulaire : {info['vocabulary']} termes")
    print_info(f"Documents   : {info['documents']} chunks")

    # ── 3. Évaluation requête par requête ───────────────────────────
    print()
    print_step("Exécution des requêtes d'évaluation ...")
    print()

    evaluator = Evaluator()
    total_queries = len(EVAL_SET)
    recall_10s: List[float] = []

    for item in EVAL_SET:
        query = item["query"]
        relevant = item["relevant"]

        # Recherche hybride au rang principal (Recall@TOP_K)
        results = fusion.search(query, top_k=TOP_K)

        # Extraction des chunk_id prédits dans l'ordre
        predicted_ids = [r["chunk"]["chunk_id"] for r in results]

        # Recherche élargie au rang secondaire (Recall@SECONDARY_K).
        # Une passe séparée est nécessaire : FusionSearch normalise les
        # scores (min-max) sur un pool de top_k*2 candidats, donc changer
        # top_k modifierait le classement du top 5. Cette passe est
        # read-only : elle n'affecte ni Recall@5 ni RR.
        results10 = fusion.search(query, top_k=SECONDARY_K)
        predicted_ids_10 = [r["chunk"]["chunk_id"] for r in results10]

        # Ajout à l'évaluateur
        evaluator.add_query(
            query=query,
            retrieved_ids=predicted_ids,
            relevant_ids=list(relevant),
        )

        recall5 = recall_at_k(predicted_ids, list(relevant), TOP_K)
        recall10 = recall_at_k(predicted_ids_10, list(relevant), SECONDARY_K)
        recall_10s.append(recall10)
        rr = reciprocal_rank(predicted_ids, list(relevant))

        # Rapport pour cette requête
        print_query_report(query, relevant, predicted_ids, recall5, recall10, rr)

    # ── 4. Vue d'ensemble avec l'Evaluator ──────────────────────────
    rapport = evaluator.evaluate(ks=[TOP_K])

    mean_recall_5 = rapport["recall_at_k"][TOP_K]
    # Garde-fou : division par zéro si EVAL_SET était vide (cas théorique,
    # la constante est définie en tête de module avec 8 requêtes).
    mean_recall_10 = sum(recall_10s) / total_queries if total_queries else 0.0
    mrr = rapport["mrr"]

    # Affichage du résumé textuel produit par l'Evaluator
    print()
    print(SEP)
    print(f"  {BOLD}{CYAN}Rapport Evaluator{RESET}")
    print(f"  {DIM}{'-' * 68}{RESET}")
    for d in rapport["details"]:
        r_at_k = ", ".join(f"R@{k}={v:.4f}" for k, v in d["recall_at_k"].items())
        print(f"  {d['query']}")
        print(f"    {r_at_k} | RR={d['reciprocal_rank']:.4f}")

    # ── 5. Résultats globaux ────────────────────────────────────────
    elapsed = time.time() - start_total
    print_final_report(total_queries, mean_recall_5, mean_recall_10, mrr, elapsed)

    # ── Bilan ────────────────────────────────────────────────────────
    print()
    if mrr >= 0.6:
        print(f"  {GREEN}{BOLD}  Bilan : Bonne performance — la recherche hybride répond bien.{RESET}")
    elif mrr >= 0.3:
        print(f"  {YELLOW}{BOLD}  Bilan : Performance moyenne — la recherche peut être améliorée.{RESET}")
    else:
        print(f"  {RED}{BOLD}  Bilan : Performance faible — revoir la stratégie de fusion.{RESET}")
    print()


if __name__ == "__main__":
    main()
