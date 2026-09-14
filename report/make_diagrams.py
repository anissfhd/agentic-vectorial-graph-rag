from graphviz import Digraph
from pathlib import Path
import matplotlib.pyplot as plt

out=Path('/mnt/data/rapport_agentic_rag/assets')

# Common attrs
font='DejaVu Sans'
blue='#e8f4fb'
line='#173f5f'
yellow='#fff4cc'
gray='#f4f6f8'

# 1 Architecture
G=Digraph('architecture', format='png')
G.attr(rankdir='LR', bgcolor='white', dpi='180', pad='0.2', nodesep='0.45', ranksep='0.7')
G.attr('node', fontname=font, fontsize='11', shape='box', style='rounded,filled', fillcolor=blue, color=line, penwidth='1.2', margin='0.16,0.10')
G.attr('edge', color=line, penwidth='1.2', arrowsize='0.8', fontname=font, fontsize='9')
G.node('pdf','Thèse PDF\n206 pages', shape='cylinder', fillcolor=gray)
G.node('ing','Ingestion\nNettoyage')
G.node('chunk','7 chunkings\nMétriques')
G.node('emb','7 représentations\nMétriques')
G.node('vec','FAISS + BM25\nPCA', shape='cylinder', fillcolor=gray)
G.node('ent','Entités\nRelations')
G.node('graph','NetworkX\nNeo4j Aura', shape='cylinder', fillcolor=gray)
G.node('api','FastAPI\nChargement des caches')
G.node('agent','Q-Learning\nLangGraph', shape='diamond', fillcolor=yellow)
G.node('ui','React\n4 onglets')
G.edge('pdf','ing'); G.edge('ing','chunk'); G.edge('chunk','emb'); G.edge('emb','vec')
G.edge('ing','ent'); G.edge('ent','graph'); G.edge('vec','api'); G.edge('graph','api')
G.edge('api','agent'); G.edge('agent','ui'); G.edge('ui','api', label='REST JSON', constraint='false')
G.render(out/'uml_architecture', cleanup=True)

# 2 Use case
G=Digraph('usecase', format='png')
G.attr(rankdir='LR', bgcolor='white', dpi='180', pad='0.25', nodesep='0.35', ranksep='0.8')
G.attr('node', fontname=font, fontsize='10', color=line, penwidth='1.1')
G.attr('edge', color=line, penwidth='1.0', arrowsize='0.7', fontname=font, fontsize='8')
G.node('user','Utilisateur', shape='box', style='rounded,filled', fillcolor=gray)
G.node('admin','Administrateur', shape='box', style='rounded,filled', fillcolor=gray)
with G.subgraph(name='cluster_system') as c:
    c.attr(label='Système Agentic Vectorial Graph RAG', color=line, fontname=font, fontsize='12', style='rounded')
    for key,label in [
        ('ask','Poser une question'),('answer','Consulter la réponse sourcée'),
        ('chunks','Visualiser chunks et pages'),('graph','Explorer graphe et Cypher'),
        ('metrics','Comparer chunkings et embeddings'),('policy','Consulter Q-table et rewards'),
        ('build','Reconstruire les artefacts'),('aura','Vérifier / synchroniser Aura')]:
        c.node(key,label, shape='ellipse', style='filled', fillcolor=blue)
G.edge('user','ask', dir='none'); G.edge('user','answer', dir='none'); G.edge('user','chunks', dir='none'); G.edge('user','graph', dir='none'); G.edge('user','metrics', dir='none'); G.edge('user','policy', dir='none')
G.edge('admin','build', dir='none'); G.edge('admin','aura', dir='none')
G.edge('answer','ask', style='dashed', label='include')
G.edge('answer','chunks', style='dashed', label='extend')
G.edge('answer','graph', style='dashed', label='extend')
G.render(out/'uml_usecase', cleanup=True)

# 3 Activity
G=Digraph('activity', format='png')
G.attr(rankdir='TB', bgcolor='white', dpi='180', pad='0.25', nodesep='0.35', ranksep='0.42')
G.attr('node', fontname=font, fontsize='10', shape='box', style='rounded,filled', fillcolor=blue, color=line, penwidth='1.1', margin='0.16,0.09')
G.attr('edge', color=line, penwidth='1.0', arrowsize='0.75', fontname=font, fontsize='8')
G.node('start','', shape='circle', width='0.25', fillcolor=line)
G.node('q','Saisir et envoyer la question')
G.node('features','Extraire les features\net construire l’état')
G.node('policy','Lire la Q-table\net choisir l’action')
G.node('route','Route ?', shape='diamond', fillcolor=yellow)
G.node('vector','FAISS / BM25')
G.node('graph','Cypher / sous-graphe')
G.node('hybrid','Vector + Graph')
G.node('abstain','Je ne sais pas')
G.node('verify','Vérifier la couverture\ndu contexte')
G.node('enough','Contexte suffisant ?', shape='diamond', fillcolor=yellow)
G.node('answer','Construire la réponse extractive')
G.node('no','Retourner « Je ne sais pas »')
G.node('reward','Calculer le reward\net enregistrer le résultat')
G.node('end','', shape='doublecircle', width='0.25', fillcolor=line)
G.edge('start','q'); G.edge('q','features'); G.edge('features','policy'); G.edge('policy','route')
G.edge('route','vector',label='vector'); G.edge('route','graph',label='graph'); G.edge('route','hybrid',label='hybrid'); G.edge('route','abstain',label='abstain')
G.edge('vector','verify'); G.edge('graph','verify'); G.edge('hybrid','verify'); G.edge('verify','enough')
G.edge('enough','answer',label='oui'); G.edge('enough','no',label='non'); G.edge('answer','reward'); G.edge('no','reward'); G.edge('abstain','reward'); G.edge('reward','end')
G.render(out/'uml_activity', cleanup=True)

# 4 Sequence with matplotlib
actors=['Utilisateur','React','FastAPI','LangGraph','FAISS','Neo4j Aura']
xs=range(len(actors))
fig,ax=plt.subplots(figsize=(13,6.8))
ax.set_xlim(-0.5,len(actors)-0.5); ax.set_ylim(10.5,-0.7); ax.axis('off')
for x,name in zip(xs,actors):
    ax.add_patch(plt.Rectangle((x-0.38,-0.35),0.76,0.55,fill=False,linewidth=1.2))
    ax.text(x,-0.075,name,ha='center',va='center',fontsize=10)
    ax.plot([x,x],[0.25,10],linestyle='--',linewidth=0.9)
msgs=[
    (0,1,1,'Question'),(1,2,2,'POST /query'),(2,3,3,'État initial'),
    (3,4,4,'Embedding + top-k'),(4,3,5,'Chunks, pages, scores'),
    (3,5,6,'Cypher paramétré'),(5,3,7,'Nœuds et relations'),
    (3,2,8,'Contexte fusionné'),(2,1,9,'JSON complet'),(1,0,10,'Réponse + preuves')]
for a,b,y,label in msgs:
    ax.annotate('',xy=(b,y),xytext=(a,y),arrowprops=dict(arrowstyle='->',lw=1.1))
    ax.text((a+b)/2,y-0.15,label,ha='center',va='bottom',fontsize=9)
ax.set_title('Séquence d’une requête hybride',fontsize=14,pad=20)
fig.tight_layout(); fig.savefig(out/'uml_sequence.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# 5 Components
G=Digraph('components', format='png')
G.attr(rankdir='TB', bgcolor='white', dpi='180', pad='0.25', nodesep='0.5', ranksep='0.6')
G.attr('node', fontname=font, fontsize='10', shape='component', style='filled', fillcolor=blue, color=line, penwidth='1.1')
G.attr('edge', color=line, penwidth='1.0', arrowsize='0.75', fontname=font, fontsize='8')
G.node('react','Frontend React\nVite + visualisations')
G.node('fast','FastAPI\nPydantic + CORS')
G.node('lang','LangGraph\nQ-Learning')
G.node('vector','Vector services\nMiniLM + FAISS + BM25')
G.node('graph','Graph services\nNetworkX + Neo4j')
G.node('art','Artefacts JSON\nIndex + Q-table', shape='cylinder', fillcolor=gray)
G.node('aura','Neo4j Aura\n66 nœuds / 171 relations', shape='cylinder', fillcolor=gray)
G.edge('react','fast',label='REST'); G.edge('fast','lang'); G.edge('lang','vector'); G.edge('lang','graph'); G.edge('art','vector'); G.edge('aura','graph')
G.render(out/'uml_components', cleanup=True)

# 6 Deployment
G=Digraph('deployment', format='png')
G.attr(rankdir='LR', bgcolor='white', dpi='180', pad='0.3', nodesep='0.55', ranksep='0.7')
G.attr('node', fontname=font, fontsize='10', shape='box', style='rounded,filled', fillcolor=blue, color=line, penwidth='1.1')
G.attr('edge', color=line, penwidth='1.0', arrowsize='0.75', fontname=font, fontsize='8')
with G.subgraph(name='cluster_pc') as c:
    c.attr(label='PC Windows 10 - environnement local', color=line, style='rounded', fontname=font)
    c.node('browser','Navigateur / React\n127.0.0.1:5173')
    c.node('api','FastAPI / Uvicorn\n127.0.0.1:8000')
    c.node('cache','FAISS, BM25, JSON\nQ-table', shape='cylinder', fillcolor=gray)
with G.subgraph(name='cluster_cloud') as c:
    c.attr(label='Cloud Neo4j Aura', color=line, style='rounded', fontname=font)
    c.node('db','Base neo4j', shape='cylinder', fillcolor=gray)
    c.node('query','Query / Cypher / Graph view')
G.edge('browser','api',label='HTTP'); G.edge('cache','api'); G.edge('api','db',label='TLS / Bolt'); G.edge('db','query')
G.render(out/'uml_deployment', cleanup=True)

# 7 Class diagram
G=Digraph('classes', format='png')
G.attr(rankdir='LR', bgcolor='white', dpi='180', pad='0.25', nodesep='0.45', ranksep='0.7')
G.attr('node', fontname=font, fontsize='9', shape='record', style='filled', fillcolor=blue, color=line, penwidth='1.1')
G.attr('edge', color=line, penwidth='1.0', arrowsize='0.75', fontname=font, fontsize='8')
G.node('chunk','{Chunk|chunk_id: str\ltext: str\lpage_pdf: int\lpage_these: int\lsection: str\llanguage: str\l}')
G.node('entity','{Entity|name: str\laliases: list\lentity_type: str\lcommunity_id: int\lcentralities: dict\l}')
G.node('relation','{Relation|source: Entity\ltarget: Entity\ltype: str\lchunk_id: str\lconfidence: float\lexcerpt: str\l}')
G.node('request','{QueryRequest|question: str\lexploration: bool\l}')
G.node('state','{AgentState|query_type: enum\lhas_entity: bool\lcoverage: enum\laction: enum\lreward: float\l}')
G.node('response','{QueryResponse|answer: str\lroute: str\lconfidence: float\lchunks: list\lsubgraph: dict\ldecision_path: list\l}')
G.edge('chunk','relation',label='provenance'); G.edge('entity','relation'); G.edge('request','state'); G.edge('state','response'); G.edge('chunk','response'); G.edge('relation','response')
G.render(out/'uml_classes', cleanup=True)

print('done')
