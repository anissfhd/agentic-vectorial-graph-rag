import { useEffect, useMemo, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { getJson, postJson } from './api'

const TABS = ['Query', 'Vectorial RAG', 'Graph RAG', 'Agentic RAG']
const COMMUNITY_COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#fb7185', '#fbbf24', '#60a5fa', '#f472b6', '#2dd4bf', '#c084fc', '#f97316', '#84cc16', '#22d3ee', '#e879f9', '#facc15', '#4ade80', '#818cf8']
const DEMO_QUESTIONS = [
  "Comment l'OMM définit-elle une vague de froid ?",
  'Quel lien existe entre le blocage scandinave et les vagues de froid en Europe occidentale ?',
  'Quelle relation existe entre le SWG et les analogues de circulation, et quel objectif cette méthode sert-elle ?',
  'Comment fonctionne la rétropropagation dans un réseau neuronal ?',
]

function useLoad(path) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    getJson(path).then(setData).catch((reason) => setError(reason.message))
  }, [path])
  return { data, error }
}

function Badge({ children, tone = 'blue' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

function Panel({ title, eyebrow, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      {title && <h2>{title}</h2>}
      {children}
    </section>
  )
}

function Metric({ label, value, detail }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  )
}

function Loading({ error }) {
  return <div className={error ? 'error' : 'loading'}>{error || 'Chargement des artefacts calculés…'}</div>
}

function QueryPage({ health }) {
  const [question, setQuestion] = useState(DEMO_QUESTIONS[0])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async (value = question) => {
    setQuestion(value)
    setLoading(true)
    setError('')
    try {
      setResult(await postJson('/query', { question: value }))
    } catch (reason) {
      setError(reason.response?.data?.detail || reason.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-grid query-layout">
      <Panel title="Interroger la thèse" eyebrow="Pipeline agentique complet" className="query-compose">
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} />
        <button className="primary" onClick={() => run()} disabled={loading}>{loading ? 'Exécution…' : 'Exécuter la requête'}</button>
        {error && <div className="error">{error}</div>}
        <div className="sample-list">
          <span>Questions de démonstration</span>
          {DEMO_QUESTIONS.map((item, index) => (
            <button key={item} onClick={() => run(item)}><b>0{index + 1}</b>{item}</button>
          ))}
        </div>
      </Panel>

      <Panel title="Décision & réponse" eyebrow="LangGraph · Q-Learning · RAG" className="answer-panel">
        {!result ? <div className="empty-state">Lancez une question pour visualiser le chemin de décision, les sources et la réponse.</div> : (
          <>
            <div className="badge-row">
              <Badge>{result.detected_type}</Badge>
              <Badge tone="violet">{result.action}</Badge>
              <Badge tone={result.neo4j_status.connected ? 'green' : 'red'}>{result.neo4j_status.connected ? 'Aura connecté' : 'Aura indisponible'}</Badge>
            </div>
            <div className="answer-text">{result.answer}</div>
            <div className="metrics-row compact">
              <Metric label="Confiance" value={`${Math.round(result.confidence * 100)} %`} />
              <Metric label="Contexte" value={`${Math.round(result.context_confidence * 100)} %`} />
              <Metric label="Route" value={result.route} />
            </div>
            <div className="decision-path">
              {result.decision_path.map((step, index) => <span key={step}>{index + 1}. {step}</span>)}
            </div>
            <h3>Provenance</h3>
            <div className="source-grid">
              <div><small>Pages PDF</small><strong>{result.pages_pdf.join(', ') || '—'}</strong></div>
              <div><small>Pages thèse</small><strong>{result.pages_these.join(', ') || '—'}</strong></div>
              <div><small>Entités</small><strong>{result.entities.slice(0, 5).join(', ') || '—'}</strong></div>
            </div>
            {result.chunks?.length > 0 && <div className="chunk-list">
              {result.chunks.slice(0, 3).map((chunk) => (
                <article key={chunk.chunk_id}>
                  <div><code>{chunk.chunk_id}</code><Badge tone="green">{chunk.score.toFixed(3)}</Badge></div>
                  <p>{chunk.excerpt}</p>
                  <small>PDF {chunk.pages_pdf.join(', ')} · {chunk.section}</small>
                </article>
              ))}
            </div>}
            {result.cypher && <><h3>Cypher exécuté dans Aura</h3><pre>{result.cypher}</pre></>}
          </>
        )}
      </Panel>
      <Panel className="status-strip">
        <Metric label="Backend" value={health?.status || '—'} />
        <Metric label="FAISS" value={health ? `${health.faiss_vectors} vecteurs` : '—'} />
        <Metric label="Knowledge Graph" value={health ? `${health.graph_nodes} nœuds` : '—'} />
        <Metric label="Neo4j Aura" value={health?.neo4j?.connected ? `${health.neo4j.nodes} / ${health.neo4j.relationships}` : '—'} detail="nœuds / relations" />
      </Panel>
    </div>
  )
}

function MetricTable({ rows, columns, winner }) {
  return (
    <div className="table-wrap"><table><thead><tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead>
      <tbody>{rows.map((row) => <tr key={row.method} className={row.method === winner ? 'winner-row' : ''}>
        {columns.map((column) => <td key={column.key}>{column.key === 'method' ? row[column.key] : typeof row[column.key] === 'number' ? row[column.key].toFixed(column.digits ?? 3) : row[column.key]}</td>)}
      </tr>)}</tbody></table></div>
  )
}

function VectorialPage() {
  const chunking = useLoad('/vectorial/chunking')
  const embeddings = useLoad('/vectorial/embeddings')
  const pca = useLoad('/vectorial/pca')
  const [question, setQuestion] = useState(DEMO_QUESTIONS[0])
  const [retrieval, setRetrieval] = useState(null)
  if (!chunking.data || !embeddings.data || !pca.data) return <Loading error={chunking.error || embeddings.error || pca.error} />
  const chunkColumns = [
    { key: 'method', label: 'Méthode' }, { key: 'chunk_count', label: 'Chunks', digits: 0 },
    { key: 'f1_at_5', label: 'F1@5' }, { key: 'recall_at_5', label: 'Recall@5' },
    { key: 'intra_similarity', label: 'Intra' }, { key: 'global_score', label: 'Score' },
  ]
  const embeddingColumns = [
    { key: 'method', label: 'Représentation' }, { key: 'dimension', label: 'Dim.', digits: 0 },
    { key: 'f1_at_5', label: 'F1@5' }, { key: 'map', label: 'MAP' }, { key: 'mrr', label: 'MRR' },
    { key: 'ndcg_at_5', label: 'NDCG@5' }, { key: 'encoding_seconds', label: 'Temps (s)' },
  ]
  return (
    <div className="page-grid">
      <Panel title="Sept méthodes de chunking" eyebrow={`Gagnante · ${chunking.data.winner}`}>
        <div className="chart-box"><ResponsiveContainer width="100%" height={270}><BarChart data={chunking.data.methods}><CartesianGrid strokeDasharray="3 3" stroke="#203047"/><XAxis dataKey="method" tick={{ fill: '#9fb0c8', fontSize: 11 }}/><YAxis tick={{ fill: '#9fb0c8' }}/><Tooltip contentStyle={{ background: '#0d192a', border: '1px solid #263954' }}/><Bar dataKey="global_score" fill="#38bdf8" radius={[7,7,0,0]}/></BarChart></ResponsiveContainer></div>
        <MetricTable rows={chunking.data.methods} columns={chunkColumns} winner={chunking.data.winner} />
      </Panel>
      <Panel title="Sept représentations vectorielles" eyebrow={`Gagnante · ${embeddings.data.winner}`}>
        <div className="chart-box"><ResponsiveContainer width="100%" height={270}><BarChart data={embeddings.data.methods}><CartesianGrid strokeDasharray="3 3" stroke="#203047"/><XAxis dataKey="method" tick={{ fill: '#9fb0c8', fontSize: 10 }}/><YAxis tick={{ fill: '#9fb0c8' }}/><Tooltip contentStyle={{ background: '#0d192a', border: '1px solid #263954' }}/><Legend/><Bar dataKey="map" fill="#a78bfa"/><Bar dataKey="mrr" fill="#34d399"/><Bar dataKey="ndcg_at_5" fill="#38bdf8"/></BarChart></ResponsiveContainer></div>
        <MetricTable rows={embeddings.data.methods} columns={embeddingColumns} winner={embeddings.data.winner} />
      </Panel>
      <Panel title="Projection PCA 2D" eyebrow={`${pca.data.points.length} chunks · ${pca.data.clusters} clusters`}>
        <div className="chart-box"><ResponsiveContainer width="100%" height={420}><ScatterChart><CartesianGrid stroke="#203047"/><XAxis dataKey="x" type="number" tick={{ fill: '#9fb0c8' }}/><YAxis dataKey="y" type="number" tick={{ fill: '#9fb0c8' }}/><ZAxis dataKey="cluster" range={[45, 45]}/><Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#0d192a', border: '1px solid #263954' }}/><Scatter data={pca.data.points} fill="#38bdf8" fillOpacity={0.7}/></ScatterChart></ResponsiveContainer></div>
      </Panel>
      <Panel title="Comparer le retrieval" eyebrow="FAISS · BM25 · Reciprocal Rank Fusion">
        <div className="inline-form"><input value={question} onChange={(event) => setQuestion(event.target.value)} /><button className="primary" onClick={() => postJson('/vectorial/retrieve', { question, top_k: 5 }).then(setRetrieval)}>Comparer</button></div>
        {retrieval && <div className="retrieval-columns">{Object.entries(retrieval.methods).map(([method, items]) => <div key={method}><h3>{method}</h3>{items.slice(0,3).map((item) => <article key={item.chunk_id}><strong>{item.score.toFixed(3)} · PDF {item.pages_pdf.join(', ')}</strong><p>{item.excerpt.slice(0,220)}…</p></article>)}</div>)}</div>}
      </Panel>
    </div>
  )
}

function GraphPage() {
  const graph = useLoad('/graph')
  const metrics = useLoad('/graph/metrics')
  const status = useLoad('/graph/neo4j/status')
  const [question, setQuestion] = useState(DEMO_QUESTIONS[1])
  const [subgraph, setSubgraph] = useState(null)
  const forceData = useMemo(() => graph.data ? {
    nodes: graph.data.nodes.map((node) => ({ ...node, id: node.name })),
    links: graph.data.relations.map((relation) => ({ ...relation })),
  } : { nodes: [], links: [] }, [graph.data])
  if (!graph.data || !metrics.data || !status.data) return <Loading error={graph.error || metrics.error || status.error} />
  return (
    <div className="page-grid graph-layout">
      <Panel className="graph-canvas" title="Knowledge Graph interactif" eyebrow="NetworkX calculé · Neo4j Aura synchronisé">
        <div className={`neo4j-banner ${status.data.connected ? 'online' : 'offline'}`}><span className="pulse"/><b>{status.data.message}</b><span>{status.data.nodes} nœuds · {status.data.relationships} relations · {status.data.communities} communautés</span></div>
        <div className="force-wrap"><ForceGraph2D graphData={forceData} width={900} height={600} backgroundColor="#081321" nodeAutoColorBy="community_id" nodeLabel={(node) => `${node.name} · communauté ${node.community_id}`} linkColor={() => 'rgba(126, 154, 190, .28)'} linkDirectionalArrowLength={3} linkDirectionalArrowRelPos={1} cooldownTicks={80} /></div>
        <div className="legend">{Array.from({ length: metrics.data.community_count }, (_, index) => <span key={index}><i style={{ background: COMMUNITY_COLORS[index % COMMUNITY_COLORS.length] }}/>{index}</span>)}</div>
      </Panel>
      <Panel title="Métriques structurelles" eyebrow="Louvain & centralités">
        <div className="metrics-row"><Metric label="Nœuds" value={metrics.data.nodes}/><Metric label="Relations" value={metrics.data.edges}/><Metric label="Densité" value={metrics.data.density.toFixed(4)}/><Metric label="Modularité" value={metrics.data.modularity.toFixed(3)}/></div>
        <h3>Nœuds centraux</h3>
        <div className="rank-list">{metrics.data.top_degree_centrality.map((item, index) => <div key={item.name}><b>{String(index + 1).padStart(2,'0')}</b><span>{item.name}</span><strong>{item.value.toFixed(3)}</strong></div>)}</div>
        <h3>Communautés</h3>
        <div className="community-list">{metrics.data.communities.map((community) => <div key={community.community_id}><i style={{ background: COMMUNITY_COLORS[community.community_id % COMMUNITY_COLORS.length] }}/><b>Cluster {community.community_id}</b><span>{community.size} membres</span><small>{community.members.slice(0,4).join(', ')}</small></div>)}</div>
      </Panel>
      <Panel title="Sous-graphe d’une requête" eyebrow="Cypher réellement exécuté dans Aura" className="wide">
        <div className="inline-form"><input value={question} onChange={(event) => setQuestion(event.target.value)} /><button className="primary" onClick={() => postJson('/graph/subgraph', { question }).then(setSubgraph)}>Exécuter dans Aura</button></div>
        {subgraph && <div className="subgraph-result"><div className="metrics-row compact"><Metric label="Source" value={subgraph.source}/><Metric label="Nœuds" value={subgraph.nodes.length}/><Metric label="Relations" value={subgraph.relations.length}/></div><pre>{subgraph.cypher}</pre><div className="edge-list">{subgraph.relations.slice(0,12).map((edge) => <span key={edge.relation_id}>{edge.source} <b>—{edge.type}→</b> {edge.target}</span>)}</div></div>}
      </Panel>
    </div>
  )
}

function AgenticPage() {
  const agentic = useLoad('/agentic')
  if (!agentic.data) return <Loading error={agentic.error} />
  const { q_table: qTable, reward_history: rewards, routing_metrics: routing } = agentic.data
  return (
    <div className="page-grid">
      <Panel title="Politique Q-Learning" eyebrow="16 états · 4 actions · décision par argmax Q">
        <div className="metrics-row"><Metric label="Accuracy test" value={`${Math.round(routing.routing_accuracy * 100)} %`}/><Metric label="Abstention" value={`${Math.round(routing.correct_abstention_rate * 100)} %`}/><Metric label="Alpha" value={qTable.alpha}/><Metric label="Gamma" value={qTable.gamma}/></div>
        <div className="table-wrap"><table><thead><tr><th>État</th>{qTable.actions.map((action) => <th key={action}>{action}</th>)}<th>Politique</th></tr></thead><tbody>{qTable.states.map((state) => {
          const values = qTable.q_values[state]
          const maximum = Math.max(...values)
          return <tr key={state}><td><code>{state}</code></td>{values.map((value,index) => <td key={qTable.actions[index]} className={value === maximum ? 'q-max' : ''}>{value.toFixed(3)}</td>)}<td><Badge tone="violet">{qTable.actions[values.indexOf(maximum)]}</Badge></td></tr>
        })}</tbody></table></div>
      </Panel>
      <Panel title="Reward Monitor" eyebrow={`ε ${qTable.epsilon_start} → ${qTable.epsilon_end}`}>
        <div className="chart-box"><ResponsiveContainer width="100%" height={360}><LineChart data={rewards.history}><CartesianGrid strokeDasharray="3 3" stroke="#203047"/><XAxis dataKey="epoch" tick={{ fill: '#9fb0c8' }}/><YAxis tick={{ fill: '#9fb0c8' }}/><Tooltip contentStyle={{ background: '#0d192a', border: '1px solid #263954' }}/><Legend/><Line type="monotone" dataKey="mean_reward" stroke="#34d399" dot={false} strokeWidth={2}/><Line type="monotone" dataKey="routing_accuracy" stroke="#a78bfa" dot={false} strokeWidth={2}/><Line type="monotone" dataKey="epsilon" stroke="#fbbf24" dot={false}/></LineChart></ResponsiveContainer></div>
        <h3>Workflow LangGraph</h3><div className="workflow">{agentic.data.langgraph_nodes.map((node,index) => <span key={node}><b>{index + 1}</b>{node}</span>)}</div>
      </Panel>
      <Panel title="Évaluation sans fuite" eyebrow="8 questions test · 2 par type" className="wide">
        <div className="prediction-grid">{routing.predictions.map((item) => <article key={item.question_id}><Badge tone={item.correct ? 'green' : 'red'}>{item.correct ? 'correct' : 'incorrect'}</Badge><h3>{item.question_id}</h3><p>{item.estimated_type} → <b>{item.predicted_action}</b></p><code>{item.state}</code></article>)}</div>
      </Panel>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState(TABS[0])
  const health = useLoad('/health')
  return (
    <div className="app-shell">
      <header>
        <div className="brand-mark">VG</div>
        <div><div className="eyebrow">Projet de fin d’année · IA</div><h1>Agentic Vector–Graph RAG</h1><p>Reinforcement Learning appliqué aux vagues de froid extrêmes en Europe</p></div>
        <div className={`live-status ${health.data?.neo4j?.connected ? 'online' : ''}`}><span/><div><small>Système</small><b>{health.data?.neo4j?.connected ? 'Opérationnel' : 'Vérification…'}</b></div></div>
      </header>
      <nav>{TABS.map((item, index) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}><span>0{index + 1}</span>{item}</button>)}</nav>
      <main>
        {tab === 'Query' && <QueryPage health={health.data} />}
        {tab === 'Vectorial RAG' && <VectorialPage />}
        {tab === 'Graph RAG' && <GraphPage />}
        {tab === 'Agentic RAG' && <AgenticPage />}
      </main>
      <footer><span>Corpus unique · Thèse Cadiou 2025 · 206 pages</span><span>Calculs réels · Seed 42 · CPU</span></footer>
    </div>
  )
}

