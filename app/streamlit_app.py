"""
AstraExec — Interface Streamlit
================================

Lancement :
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import time

from app.retrieval.fusion_search import FusionSearch
from app.retrieval.evidence_rank import EvidenceRank
from app.retrieval.document_manager import DocumentManager
from app.retrieval.lexi_rank import LexiRank

# =============================================================
# Configuration de la page
# =============================================================

st.set_page_config(
    page_title="AstraExec — Recherche Hybride",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =============================================================
# CSS personnalisé
# =============================================================

st.markdown("""
<style>
    /* Reset Streamlit defaults */
    #root > div:first-child > div > div > div > div > section {
        padding-top: 1rem;
    }
    .stApp {
        background: #0a0e1a;
    }
    .stApp header {
        background: transparent;
    }

    /* Logo */
    .logo-container {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .logo-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
    }
    .logo-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; font-weight: 800; color: #fff;
        box-shadow: 0 0 24px rgba(59,130,246,0.25);
    }
    .logo-text {
        font-size: 1.6rem; font-weight: 800;
        background: linear-gradient(135deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .logo-sub {
        color: #94a3b8;
        font-size: 0.85rem; text-align: center;
        margin-top: 0.25rem;
    }
    .logo-sub span {
        color: #3b82f6; font-weight: 500;
    }

    /* Search input */
    .stTextInput>div>div>input {
        background: #111827 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1rem !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(59,130,246,0.3) !important;
    }

    /* Metric cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem; font-weight: 700; color: #3b82f6;
    }
    .metric-label {
        font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Result cards */
    .result-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    .result-card:hover {
        border-color: rgba(59,130,246,0.3);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .result-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;
    }
    .result-badge {
        font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
        padding: 0.2rem 0.6rem; border-radius: 6px;
        background: rgba(59,130,246,0.12); color: #3b82f6;
        border: 1px solid rgba(59,130,246,0.15);
    }
    .result-score {
        display: flex; align-items: center; gap: 0.75rem;
    }
    .score-main {
        font-size: 1rem; font-weight: 700; color: #3b82f6;
    }
    .score-detail {
        font-size: 0.7rem; color: #94a3b8; display: flex; gap: 0.5rem;
    }
    .score-detail span {
        padding: 0.1rem 0.4rem; border-radius: 4px; background: #1e293b;
    }
    .score-detail .sem { color: #60a5fa; }
    .score-detail .lex { color: #f59e0b; }
    .result-content {
        font-size: 0.88rem; line-height: 1.6; color: #e2e8f0;
    }
    .result-footer {
        margin-top: 0.5rem; font-size: 0.72rem; color: #94a3b8;
        display: flex; gap: 1rem;
    }
    .result-footer .src { color: #3b82f6; font-weight: 500; }

    /* Summary bar */
    .summary-bar {
        display: flex; align-items: center; justify-content: space-between;
        margin: 1rem 0; flex-wrap: wrap; gap: 0.5rem;
    }
    .summary-left {
        font-size: 0.85rem; color: #94a3b8;
    }
    .summary-left strong { color: #e2e8f0; }
    .summary-right {
        font-size: 0.8rem; color: #94a3b8;
    }
    .summary-right .q { color: #3b82f6; font-weight: 500; }

    /* Footer */
    .footer {
        text-align: center; padding: 2rem 0 1rem;
        font-size: 0.72rem; color: #94a3b8; opacity: 0.5;
    }
    .footer span { color: #3b82f6; }

    /* Divider */
    hr {
        border-color: #1e3a5f !important;
        margin: 1.5rem 0 !important;
    }

    /* StMetric style fix */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div {
        color: #3b82f6 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    @media (max-width: 640px) {
        .result-header { flex-direction: column; align-items: flex-start; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# Initialisation du moteur de recherche (caché)
# =============================================================

@st.cache_resource
def init_search_engine():
    engine = FusionSearch()
    ranker = EvidenceRank()
    manager = DocumentManager("app/api/data")
    chunks = manager.load_documents()
    engine.build_index(chunks)
    return engine, ranker


try:
    with st.spinner("Initialisation du moteur de recherche..."):
        search_engine, ranker = init_search_engine()
    engine_ready = True
except Exception as e:
    st.error(f"Erreur d'initialisation : {e}")
    engine_ready = False


# =============================================================
# Sidebar
# =============================================================

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    top_k = st.slider("Nombre de résultats", 1, 10, 5)

    st.markdown("---")
    st.markdown("### 💡 Suggestions")
    suggestions = ["machine learning", "deep learning", "recherche lexicale", "BM25", "TF-IDF"]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state["query"] = s
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Stats")
    if engine_ready:
        st.metric("Documents indexes", search_engine.vectorizer.dimension)
        st.metric("Chunks", len(search_engine.documents))
    else:
        st.warning("Moteur non initialise")


# =============================================================
# En-tête
# =============================================================

st.markdown('<div class="logo-container">'
            '<div class="logo-badge">'
            '<div class="logo-icon">A</div>'
            '<span class="logo-text">AstraExec</span>'
            '</div>'
            '<div class="logo-sub">Moteur de recherche hybride — '
            '<span>TF-IDF custom</span> + <span>BM25 custom</span></div>'
            '</div>', unsafe_allow_html=True)


# =============================================================
# Recherche
# =============================================================

# Récupérer la query depuis session state si définie
default_query = st.session_state.get("query", "")
query = st.text_input(
    "Rechercher",
    value=default_query,
    placeholder="Entrez votre requête...",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([3, 1, 1])
with col2:
    search_clicked = st.button("🔍 Rechercher", use_container_width=True)
with col3:
    clear_clicked = st.button("🗑️ Effacer", use_container_width=True)

if clear_clicked:
    st.session_state["query"] = ""
    st.rerun()

# =============================================================
# Résultats
# =============================================================

if engine_ready and (search_clicked or default_query) and query:
    with st.spinner("Recherche en cours..."):
        start_time = time.time()
        results = search_engine.search(query, top_k=top_k * 2)
        ranked = ranker.rerank(results)
        elapsed = time.time() - start_time

    if not ranked:
        st.info("Aucun résultat trouvé. Essayez d'autres termes.")
    else:
        # Barre de résumé
        st.markdown(
            f'<div class="summary-bar">'
            f'<div class="summary-left"><strong>{len(ranked[:top_k])}</strong> résultats · '
            f'{elapsed:.4f}s</div>'
            f'<div class="summary-right">Requête : <span class="q">“{query}”</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Cartes de résultat
        for i, r in enumerate(ranked[:top_k]):
            chunk = r.get("chunk", {})
            content = chunk.get("content", "")
            source = chunk.get("source", "inconnu")
            cid = chunk.get("chunk_id", i)
            fs = r.get("final_score", 0)
            sem = r.get("semantic", 0)
            lex = r.get("lexical", 0)

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-header">'
                f'<span class="result-badge">#{i + 1}</span>'
                f'<div class="result-score">'
                f'<span class="score-main">{fs:.4f}</span>'
                f'<div class="score-detail">'
                f'<span class="sem">sem {sem:.4f}</span>'
                f'<span class="lex">lex {lex:.4f}</span>'
                f'</div>'
                f'</div>'
                f'</div>'
                f'<div class="result-content">{content}</div>'
                f'<div class="result-footer">'
                f'<span class="src">{source}</span>'
                f'<span>chunk #{cid}</span>'
                f'<span>{chunk.get("length", "?")} car.</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

elif not engine_ready:
    st.warning("Le moteur de recherche n'a pas pu etre initialise. Verifiez le dossier app/api/data/")
elif query == "" and not search_clicked:
    st.markdown('<div style="text-align:center;padding:3rem 1rem;color:#94a3b8;">'
                '<div style="font-size:3rem;margin-bottom:1rem;opacity:0.3;">🔍</div>'
                '<h3 style="color:#e2e8f0;font-size:1.1rem;margin-bottom:0.4rem;">Prêt à chercher</h3>'
                '<p style="font-size:0.88rem;max-width:360px;margin:0 auto;line-height:1.6;">'
                'Tapez une requête ci-dessus ou utilisez les suggestions '
                'pour explorer les documents avec la recherche hybride.</p>'
                '</div>', unsafe_allow_html=True)


# =============================================================
# Footer
# =============================================================

st.markdown("---")
st.markdown(
    '<div class="footer">'
    'AstraExec — Outils 100% custom · <span>v2.0.0</span>'
    '</div>',
    unsafe_allow_html=True,
)
