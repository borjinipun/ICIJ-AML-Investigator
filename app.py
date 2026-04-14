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

    net = Network(height="500px", width="100%", bgcolor="#111", font_color="white")

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
        model="llama-3.3-70b-versatile",
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

    gr.Markdown("# 🕵️ AML GenAI Investigator")
    gr.Markdown("AI-powered financial investigation with graph intelligence")

    question = gr.Textbox(
        label="Investigation Query",
        placeholder="Find offshore entities with shared officers"
    )

    run_btn = gr.Button("Investigate")

    query_output = gr.Textbox(label="Generated Query")
    table_output = gr.Dataframe(label="Results")
    report_output = gr.Textbox(label="Report / Insight")

    graph_output = gr.HTML(label="Network Graph")

    def run_pipeline(q):
        return investigate_simple(q)  # ✅ FIXED

    run_btn.click(
        run_pipeline,
        inputs=[question],
        outputs=[query_output, table_output, report_output, graph_output]
    )

demo.launch()