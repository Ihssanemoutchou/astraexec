from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.executor.executor import Executor
from app.registry.base_tool import BaseTool
from app.retrieval.fusion_search import FusionSearch
from app.retrieval.evidence_rank import EvidenceRank
from app.retrieval.document_manager import DocumentManager
from app.retrieval.query_profiler import QueryProfiler

# --- Couche d'intégration Module REASONING (additive, contrat v1.0) ---
from app.schemas.retrieval_contract import RetrievalRequest, RetrievalResponse
from app.integration.retrieval_adapter import RetrievalAdapter, RetrievalError

app = FastAPI(
    title="AstraExec",
    version="2.0.0",
    description="Module d'action intelligent -- Outils 100% custom",
)

executor = Executor()


class ActionRequest(BaseModel):
    tool: str
    parameters: dict


class FusionTool(BaseTool):

    def __init__(self):
        super().__init__(
            "fusion_search",
            "Recherche hybride AstraExec (TF-IDF custom + BM25 custom)"
        )

        self.search_engine = FusionSearch()
        self.ranker = EvidenceRank()
        self.profiler = QueryProfiler()

        # Chargement ou construction de l'index
        try:
            self.search_engine.load()
        except (FileNotFoundError, Exception):
            manager = DocumentManager("app/api/data")
            chunks = manager.load_documents()
            self.search_engine.build_index(chunks)
            self.search_engine.save()

    def execute(self, **kwargs):
        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Le parametre 'query' est obligatoire.")

        # QueryProfiler adapte les poids de la fusion
        profile = self.profiler.profile(query)
        results = self.search_engine.search(query, profile=profile)
        ranked = self.ranker.rerank(results)

        return {
            "results": ranked,
            "profile": profile,
        }

    @property
    def parameter_schema(self) -> dict:
        return {
            "query": {
                "type": "string",
                "required": True,
                "description": "Requete de recherche"
            }
        }


executor.register_tool(FusionTool())

# Façade /retrieve : utilise le même Executor (et donc le même fusion_search)
# que /execute — aucun composant métier n'est modifié.
retrieval_adapter = RetrievalAdapter(executor)


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """
    Point d'entrée du module REASONING (contrat RetrievalRequest/RetrievalResponse).

    Endpoint volontairement minimal : toute la logique de mapping vit dans
    RetrievalAdapter (app/integration/retrieval_adapter.py).
    """
    return retrieval_adapter.retrieve(request)


@app.exception_handler(RetrievalError)
def retrieval_error_handler(request, exc: RetrievalError):
    """Erreur de retrieval → HTTP 400, avec query_id pour corrélation multi-hop."""
    body = {"detail": str(exc)}
    if exc.query_id:
        body["query_id"] = exc.query_id
    return JSONResponse(status_code=400, content=body)


@app.get("/")
def home():
    return {
        "project": "AstraExec",
        "version": "2.0.0",
        "status": "running",
        "engine": "100% custom tools"
    }


@app.get("/tools")
def tools():
    return executor.available_tools()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "executor": "running",
        "tool_count": len(executor.available_tools())
    }


@app.post("/execute")
def execute(action: ActionRequest):
    try:
        return executor.run(action.model_dump())
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )


@app.get("/search-ui", response_class=HTMLResponse)
def search_ui():
    return SEARCH_HTML


@app.get("/search")
def search(query: str = ""):
    if not query:
        return {"status": "error", "message": "Le parametre 'query' est obligatoire."}

    result = executor.run({"tool": "fusion_search", "parameters": {"query": query}})
    return result


SEARCH_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AstraExec — Recherche Hybride</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface-2: #1e293b;
    --border: #1e3a5f;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.25);
    --success: #22c55e;
    --orange: #f59e0b;
  }

  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .bg-gradient {
    position: fixed;
    top: -40%;
    left: -20%;
    width: 140%;
    height: 140%;
    background: radial-gradient(ellipse at 30% 20%, rgba(59,130,246,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 80%, rgba(139,92,246,0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 860px;
    padding: 2.5rem 1.5rem;
  }

  /* Header */
  header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .logo {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
  }

  .logo-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 800;
    color: #fff;
    box-shadow: 0 0 20px var(--accent-glow);
  }

  .logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
  }

  .subtitle {
    color: var(--text-dim);
    font-size: 0.9rem;
    font-weight: 400;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.5;
  }

  .subtitle span {
    color: var(--accent);
    font-weight: 500;
  }

  /* Search Box */
  .search-wrapper {
    position: relative;
    margin-bottom: 2.5rem;
  }

  .search-box {
    position: relative;
    display: flex;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.5rem 0.5rem 0.5rem 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }

  .search-box:focus-within {
    border-color: var(--accent);
    box-shadow: 0 4px 24px var(--accent-glow), 0 0 0 1px var(--accent);
  }

  .search-box .icon {
    color: var(--text-dim);
    font-size: 1.1rem;
    flex-shrink: 0;
    transition: color 0.3s;
  }

  .search-box:focus-within .icon {
    color: var(--accent);
  }

  .search-box input {
    flex: 1;
    background: none;
    border: none;
    outline: none;
    color: var(--text);
    font-size: 1.05rem;
    font-family: inherit;
    padding: 0.7rem 0.8rem;
    min-width: 0;
  }

  .search-box input::placeholder {
    color: var(--text-dim);
    opacity: 0.6;
  }

  .search-btn {
    background: linear-gradient(135deg, var(--accent), #2563eb);
    color: #fff;
    border: none;
    border-radius: 11px;
    padding: 0.65rem 1.4rem;
    font-size: 0.9rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    transition: all 0.25s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .search-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px var(--accent-glow);
  }

  .search-btn:active {
    transform: translateY(0);
  }

  .search-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  /* Suggestions */
  .suggestions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 0.8rem;
    opacity: 0.7;
    transition: opacity 0.2s;
  }

  .suggestions:hover {
    opacity: 1;
  }

  .suggestions span {
    font-size: 0.78rem;
    color: var(--text-dim);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    cursor: pointer;
    transition: all 0.2s;
  }

  .suggestions span:hover {
    background: var(--surface-2);
    border-color: var(--accent);
    color: var(--accent);
  }

  /* Status */
  .status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    opacity: 0;
    transition: opacity 0.4s;
  }

  .status-bar.visible {
    opacity: 1;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  .results {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    transition: all 0.25s ease;
    animation: cardIn 0.35s ease-out both;
  }

  .result-card:hover {
    border-color: rgba(59,130,246,0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }

  @keyframes cardIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .result-card:nth-child(1) { animation-delay: 0.02s; }
  .result-card:nth-child(2) { animation-delay: 0.06s; }
  .result-card:nth-child(3) { animation-delay: 0.10s; }
  .result-card:nth-child(4) { animation-delay: 0.14s; }
  .result-card:nth-child(5) { animation-delay: 0.18s; }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .card-badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    background: rgba(59,130,246,0.12);
    color: var(--accent);
    border: 1px solid rgba(59,130,246,0.15);
  }

  .card-score {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .score-main {
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
  }

  .score-detail {
    font-size: 0.72rem;
    color: var(--text-dim);
    display: flex;
    gap: 0.6rem;
  }

  .score-detail span {
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: var(--surface-2);
  }

  .score-detail .sem { color: #60a5fa; }
  .score-detail .lex { color: #f59e0b; }

  .card-content {
    font-size: 0.88rem;
    line-height: 1.6;
    color: var(--text);
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .card-footer {
    margin-top: 0.6rem;
    font-size: 0.75rem;
    color: var(--text-dim);
    display: flex;
    gap: 1rem;
  }

  .card-footer .source {
    color: var(--accent);
    font-weight: 500;
  }

  .card-footer .chunk-id {
    opacity: 0.5;
  }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: var(--text-dim);
  }

  .empty-state .icon-big {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.3;
  }

  .empty-state h3 {
    color: var(--text);
    font-size: 1.1rem;
    margin-bottom: 0.4rem;
  }

  .empty-state p {
    font-size: 0.88rem;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.6;
  }

  /* Error state */
  .error-state {
    text-align: center;
    padding: 2rem;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 12px;
    color: #f87171;
  }

  /* Footer */
  footer {
    margin-top: 3rem;
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-dim);
    opacity: 0.5;
  }

  footer span {
    color: var(--accent);
    opacity: 0.7;
  }

  @media (max-width: 600px) {
    .container { padding: 1.5rem 1rem; }
    .search-box { flex-wrap: wrap; padding: 0.5rem; }
    .search-box input { width: 100%; }
    .search-btn { width: 100%; text-align: center; margin-top: 0.3rem; }
    header { margin-bottom: 2rem; }
    .result-card { padding: 1rem; }
  }
</style>
</head>
<body>
<div class="bg-gradient"></div>
<div class="container">

  <header>
    <div class="logo">
      <div class="logo-icon">A</div>
      <span class="logo-text">AstraExec</span>
    </div>
    <p class="subtitle">
      Moteur de recherche hybride &mdash; <span>TF-IDF custom</span> + <span>BM25 custom</span>
    </p>
  </header>

  <div class="search-wrapper">
    <div class="search-box">
      <span class="icon">&#x1F50D;</span>
      <input
        id="searchInput"
        type="text"
        placeholder="Rechercher dans les documents..."
        autofocus
      />
      <button class="search-btn" id="searchBtn">Rechercher</button>
    </div>
    <div class="suggestions" id="suggestions">
      <span onclick="doSearch(this.textContent)">machine learning</span>
      <span onclick="doSearch(this.textContent)">recherche lexicale</span>
      <span onclick="doSearch(this.textContent)">deep learning</span>
      <span onclick="doSearch(this.textContent)">BM25</span>
    </div>
  </div>

  <div class="status-bar" id="statusBar">
    <div class="spinner"></div>
    <span id="statusText">Recherche en cours...</span>
  </div>

  <div id="resultsContainer">
    <div class="empty-state">
      <div class="icon-big">&#x1F50D;</div>
      <h3>Pr&ecirc;t &agrave; chercher</h3>
      <p>
        Tapez une requ&ecirc;te ci-dessus ou cliquez sur un exemple
        pour explorer les documents avec la recherche hybride.
      </p>
    </div>
  </div>

  <footer>
    AstraExec &mdash; Outils 100% custom &middot; <span>v2.0.0</span>
  </footer>
</div>

<script>
  const input = document.getElementById('searchInput');
  const btn = document.getElementById('searchBtn');
  const statusBar = document.getElementById('statusBar');
  const statusText = document.getElementById('statusText');
  const resultsContainer = document.getElementById('resultsContainer');

  let requestId = 0;

  async function doSearch(query) {
    if (!query || !query.trim()) return;
    query = query.trim();
    input.value = query;

    const id = ++requestId;

    statusBar.classList.add('visible');
    statusText.textContent = 'Recherche en cours...';
    btn.disabled = true;

    resultsContainer.innerHTML = '';

    try {
      const res = await fetch(`/search?query=${encodeURIComponent(query)}`);
      const data = await res.json();

      if (id !== requestId) return;

      if (data.status === 'error') {
        showError(data.message || 'Erreur lors de la recherche.');
        return;
      }

      const resultData = data.result || {};
      const results = resultData.results || [];
      const profile = resultData.profile || {};

      if (results.length === 0) {
        showEmpty(query);
      } else {
        renderResults(query, results, data.execution_time, profile);
      }
    } catch (err) {
      if (id === requestId) {
        showError('Erreur de connexion au serveur.');
      }
    } finally {
      if (id === requestId) {
        statusBar.classList.remove('visible');
        btn.disabled = false;
      }
    }
  }

  function renderResults(query, results, time, profile) {
    let html = `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;flex-wrap:wrap;gap:0.5rem;">
      <div style="font-size:0.85rem;color:var(--text-dim);">
        <strong style="color:var(--text);">${results.length}</strong> r&eacute;sultat${results.length > 1 ? 's' : ''}
        &middot; ${(time || 0).toFixed(4)}s
      </div>
      <div style="font-size:0.8rem;color:var(--text-dim);">
        Requ&ecirc;te : &ldquo;<span style="color:var(--accent);font-weight:500;">${escapeHtml(query)}</span>&rdquo;
      </div>
    </div>`;

    if (profile && profile.type) {
      html += '<div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:0.8rem;text-align:center;">Profil : <span style="color:var(--orange);font-weight:500;">' + profile.type + '</span></div>';
    }

    html += '<div class="results">';

    results.forEach((r, i) => {
      const chunk = r.chunk || {};
      const content = chunk.content || '';
      const source = chunk.source || 'inconnu';
      const cid = chunk.chunk_id ?? i;
      const finalScore = r.final_score ? r.final_score.toFixed(4) : '—';
      const sem = r.semantic ? r.semantic.toFixed(4) : '—';
      const lex = r.lexical ? r.lexical.toFixed(4) : '—';

      html += `<div class="result-card" style="animation-delay:${(i * 0.04).toFixed(2)}s">
        <div class="card-header">
          <span class="card-badge">#${i + 1}</span>
          <div class="card-score">
            <span class="score-main">${finalScore}</span>
            <div class="score-detail">
              <span class="sem">sem ${sem}</span>
              <span class="lex">lex ${lex}</span>
            </div>
          </div>
        </div>
        <div class="card-content">${escapeHtml(content)}</div>
        <div class="card-footer">
          <span class="source">${escapeHtml(source)}</span>
          <span class="chunk-id">chunk #${cid}</span>
          <span>${chunk.length || '?'} car.</span>
        </div>
      </div>`;
    });

    html += '</div>';
    resultsContainer.innerHTML = html;
  }

  function showEmpty(query) {
    resultsContainer.innerHTML = `<div class="empty-state">
      <div class="icon-big">&#x1F50E;</div>
      <h3>Aucun r&eacute;sultat</h3>
      <p>Aucun document trouv&eacute; pour &ldquo;${escapeHtml(query)}&rdquo;. Essayez d'autres termes.</p>
    </div>`;
  }

  function showError(msg) {
    resultsContainer.innerHTML = `<div class="error-state">&#x26A0; ${escapeHtml(msg)}</div>`;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Event listeners
  btn.addEventListener('click', () => doSearch(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doSearch(input.value);
  });

  // Trigger initial suggestion click simulation — actually just focus
  input.focus();
</script>
</body>
</html>"""