import gradio as gr
import pandas as pd
import kuzu
import os
from groq import Groq
from dotenv import load_dotenv
from pyvis.network import Network
import networkx as nx
import uuid

def build_graph_from_df(df):
    G = nx.Graph()

    if df.empty:
        return None

    # Heuristic: detect columns with entity-like data
    cols = df.columns.tolist()

    for _, row in df.iterrows():
        values = [str(v) for v in row.values if pd.notna(v)]

        # Connect all values in the row (co-occurrence graph)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                G.add_node(values[i])
                G.add_node(values[j])
                G.add_edge(values[i], values[j])

    return G


def generate_pyvis_html(G):
    if G is None or len(G.nodes) == 0:
        return "<h3>No graph data available</h3>"

    net = Network(
    height="400px",
    width="100%",
    bgcolor="#ffffff",   # ✅ light background
    font_color="#0f172a" # dark text
)
    net.from_nx(G)

    # Styling for demo impact
    net.repulsion(node_distance=120, central_gravity=0.3)
    
    filename = f"/tmp/graph_{uuid.uuid4().hex}.html"
    net.save_graph(filename)

    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    return html


load_dotenv()
# -------------------------
# Setup
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

db = kuzu.Database("data/icij_graph_db")
conn = kuzu.Connection(db)

# -------------------------
# Schema Context
# -------------------------
schema_context = """
Graph Schema (Kùzu):

Node Tables:

Entity(
    node_id INT64,
    name STRING,
    jurisdiction STRING,
    country_codes STRING  -- comma-separated string (NOT a list)
)

Officer(
    node_id INT64,
    name STRING,
    country_codes STRING  -- comma-separated string
)

Address(
    node_id INT64,
    address STRING,
    country_codes STRING
)

Intermediary(
    node_id INT64,
    name STRING
)

Other(
    node_id INT64,
    name STRING
)

Relationships:

(Officer)-[:OFFICER_OF]->(Entity)
(Entity)-[:REGISTERED_ADDRESS]->(Address)
(Intermediary)-[:INTERMEDIARY_OF]->(Entity)

IMPORTANT RULES:
- country_codes is STRING, not list
- Do NOT use size(), UNWIND, or array functions on it
- Use string comparison (=, <>, LIKE) only
- Always LIMIT 20
"""

# -------------------------
# Helpers
# -------------------------
def call_llm(prompt, temp=0):
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    return res.choices[0].message.content.strip()

def clean_query(query):
    query = query.strip()

    # remove markdown fences
    query = query.replace("```cypher", "")
    query = query.replace("```", "")

    return query.strip()

# -------------------------
# Self-Healing Query Executor
# -------------------------
def execute_with_repair(query):
    for _ in range(3):
        try:
            df = conn.execute(query).get_as_df()
            return query, df
        except Exception as e:
            fix_prompt = f"""
Fix this Kùzu query.

Error:
{str(e)}

Query:
{query}

Schema:
{schema_context}

Return only corrected query.
Rules:
    - Use only given schema
    - No hallucination
    - Limit 20 results
    Do NOT include:
    - markdown
    - ```cypher
    - explanations
    - comments
"""
            query = clean_query(call_llm(fix_prompt))

    return query, pd.DataFrame()
# -------------------------
# Final Investigator with Graph
# -------------------------
def investigate_simple(question):
        try:
            prompt = f"""
    You are an expert in graph databases and AML investigation.

    Convert the question into a Kùzu Cypher query.

    Rules:
    - Use only given schema
    - No hallucination
    - Limit 20 results
    Do NOT include:
    - markdown
    - ```cypher
    - explanations
    - comments

    Schema:
    {schema_context}

    Question:
    {question}

    Return ONLY the query
    """
            query = clean_query(call_llm(prompt))
            query, df = execute_with_repair(query)

            explanation = call_llm(f"""
    You are an AML investigator.

    Question:
    {question}
    {df.head(10).to_string()}
    Explain:
    - What pattern is observed
    - Why it is suspicious
    - AML risk level (Low/Medium/High)
    """, 0.3)

            # 🆕 Graph
            G = build_graph_from_df(df)
            graph_html = generate_pyvis_html(G)

            return query, df, explanation, graph_html

        except Exception as e:
            return "Error", pd.DataFrame(), str(e), "<h3>Error</h3>"

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="AML GenAI Investigator") as demo:

    # -------- Custom Styling --------
    gr.HTML("""
<style>

/* 🌗 Auto Theme Support */
@media (prefers-color-scheme: dark) {

    body, .gradio-container {
        background: #0f172a !important;
        color: #e2e8f0 !important;
    }

    .title {
        color: #f1f5f9 !important;
    }

    .subtitle {
        color: #94a3b8 !important;
    }

    .card {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    .card h3 {
        color: #38bdf8 !important;
    }

    .card li {
        color: #cbd5f5 !important;
    }

}

/* ☀️ Light Theme (default) */
@media (prefers-color-scheme: light) {

    body, .gradio-container {
        background: #f8fafc !important;
        color: #0f172a !important;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #334155;
    }

}

/* 🧱 Shared Styling */
.gradio-container {
    max-width: 1250px;
    margin: auto;
    padding: 20px;
}

.card {
    border-radius: 14px;
    padding: 18px;
    height: 100%;
    transition: all 0.2s ease;
}

.card:hover {
    transform: translateY(-5px);
}

</style>
""")

    # -------- Title --------
    gr.HTML("""
    <div style="text-align: center; font-family: sans-serif; padding: 10px;">
        <h1 style="color: #2D3E50; margin-bottom: 5px;">🕵️ Anti Money Laundering GenAI Investigator</h1>
        <h3 style="color: #5D6D7E; margin-top: 0;">Offshore Leaks Database</h3>
        
        <p style="font-size: 1.1em; max-width: 800px; margin: 20px auto; line-height: 1.5;">
            Uncover the entities behind <b>810,000+</b> offshore companies, foundations, and trusts. 
            Leverage AI-powered graph intelligence to generate Cypher queries and extract deep AML insights from ICIJ data.
        </p>
        
        <div style="font-size: 0.85em; color: #85929E; border-top: 1px solid #EAECEE; padding-top: 10px;">
            Citation: International Consortium of Investigative Journalists. <i>Offshore Leaks Database</i>. 
            Retrieved April 14, 2026, from <a href="https://offshoreleaks.icij.org/" target="_blank" style="color: #3498DB;">offshoreleaks.icij.org</a>
        </div>
    </div>
""")

    # -------- 4 Column Layout --------
    with gr.Row():

        # Context + Data
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <h3>🔍 Context & Data</h3>
                <ul>
                    <li>ICIJ Offshore Leaks dataset (Panama, Pandora Papers)</li>
                    <li>Millions of records across jurisdictions</li>
                    <li>Entities, Officers, Intermediaries, Addresses</li>
                    <li>Highly connected financial ownership network</li>
                    <li>Requires graph-based investigation</li>
                </ul>
            </div>
            """)

        # Problem
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <h3>🚨 Problem</h3>
                <ul>
                    <li>Hidden beneficial ownership</li>
                    <li>Cross-border financial structures</li>
                    <li>Shell companies & proxy directors</li>
                    <li>Layered transactions (multi-hop)</li>
                    <li>Hard to detect using SQL/manual analysis</li>
                </ul>
            </div>
            """)

        # Solution
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <h3>💡 Solution</h3>
                <ul>
                    <li>Graph DB (Kùzu) for relationship modeling</li>
                    <li>GenAI → Natural language to Cypher</li>
                    <li>Agentic investigation workflow</li>
                    <li>Auto query → results → AML insights</li>
                    <li>Explainable AI risk analysis</li>
                </ul>
            </div>
            """)

        # Detection Methods
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <h3>🧩 Detection Methods & Sample Queries</h3>
                <p style="font-size: 0.9em; color: #666;">Click a method to see what you can ask:</p>
                <ul style="line-height: 1.8;">
                    <li><b>Ownership Mismatch:</b> <i>"Which officers are linked to companies in a different country than their own?"</i></li>
                    <li><b>Intermediary Hotspots:</b> <i>"Which intermediaries manage the most companies?"</i></li>
                    <li><b>Address Clusters:</b> <i>"“Which addresses are used by many companies?”"</i></li>
                    <li><b>Layered Ownership:</b> <i>"Which officers are connected through multiple layers of companies?"</i></li>
                    <li><b>Suspicious Triangular Relationship:</b> <i>"Show officer–company–intermediary connections"</i></li>
                </ul>
            </div>
            """)
    # -------- Input Section --------
    gr.Markdown("### ⚡ Run Investigation")

    question = gr.Textbox(
        label="",
        placeholder="e.g. Find Indian officers controlling foreign entities"
    )

    run_btn = gr.Button("🔎 Investigate")

    # -------- Outputs --------
    query_output = gr.Textbox(label="Generated Query")
    table_output = gr.Dataframe(label="Results")
    report_output = gr.Textbox(label="AML Insight")
    #graph_output = gr.HTML(label="Network Graph")

    def run_pipeline(q):
        return investigate_simple(q)

    run_btn.click(
        run_pipeline,
        inputs=[question],
        outputs=[query_output, table_output, report_output]
    )

demo.launch()