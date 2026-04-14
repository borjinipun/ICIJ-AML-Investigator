import gradio as gr
import pandas as pd
import kuzu
import os
from groq import Groq
from dotenv import load_dotenv

from pyvis.network import Network
import networkx as nx
import uuid

# -------------------------
# GRAPH BUILDER
# -------------------------
def build_graph_from_df(df):
    G = nx.Graph()

    if df is None or df.empty:
        return None

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Add nodes
        for col, val in row_dict.items():
            if pd.notna(val):
                node_id = str(val)
                G.add_node(node_id, label=node_id, group=col)

        # Add edges (column relationships)
        keys = list(row_dict.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                v1 = row_dict[keys[i]]
                v2 = row_dict[keys[j]]

                if pd.notna(v1) and pd.notna(v2):
                    G.add_edge(str(v1), str(v2), label=f"{keys[i]} → {keys[j]}")

    return G


def generate_pyvis_html(G):
    if G is None or len(G.nodes) == 0:
        return "<h3>No graph data available</h3>"

    # Prevent UI crash
    if len(G.nodes) > 300:
        return "<h3>Graph too large to render</h3>"

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0f172a",
        font_color="white",
        notebook=False
    )

    net.from_nx(G)

    net.repulsion(
        node_distance=180,
        central_gravity=0.2,
        spring_length=150,
        spring_strength=0.05
    )

    net.set_options("""
    var options = {
      nodes: {
        shape: "dot",
        size: 12,
        font: { size: 14 }
      },
      edges: {
        color: "#aaa",
        smooth: true
      },
      physics: {
        stabilization: true
      }
    }
    """)

    filename = f"/tmp/graph_{uuid.uuid4().hex}.html"
    net.save_graph(filename)

    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    return html


# -------------------------
# ENV + SETUP
# -------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

db = kuzu.Database("data/icij_graph_db")
conn = kuzu.Connection(db)

# -------------------------
# SCHEMA CONTEXT
# -------------------------
schema_context = """
Graph Schema (Kùzu):

Node Tables:

Entity(
    node_id INT64,
    name STRING,
    jurisdiction STRING,
    country_codes STRING
)

Officer(
    node_id INT64,
    name STRING,
    country_codes STRING
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
- country_codes is STRING
- No array functions
- Use LIKE or =
- Always LIMIT 20
"""

# -------------------------
# LLM HELPERS
# -------------------------
def call_llm(prompt, temp=0):
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    return res.choices[0].message.content.strip()


def clean_query(query):
    return query.replace("```cypher", "").replace("```", "").strip()


# -------------------------
# PLANNER
# -------------------------
def plan_investigation(question):
    prompt = f"""
    You are a financial investigator.
Break into investigation steps.

Question:
{question}

Return steps like:
1. Identify entities
2. Find relsationships
3. Detect supicious patterns
4. Summarize

Be concise.
"""
    return call_llm(prompt)


# -------------------------
# QUERY GENERATOR
# -------------------------
def generate_step_query(step, context=""):
    prompt = f"""
You are an expert in Kùzu Cypher.

Schema:
{schema_context}

Context from previous steps:
{context}

Task:
{step}

Rules:
- Only valid Cypher
- No markdown
- LIMIT 20

Return only query.
"""
    return clean_query(call_llm(prompt))


# -------------------------
# SELF-HEALING EXECUTOR
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
# AGENT LOOP
# -------------------------
def run_agent(question):
    plan = plan_investigation(question)
    steps = [s for s in plan.split("\n") if s.strip()]

    context = ""
    all_results = []

    for i, step in enumerate(steps):
        query = generate_step_query(step, context)
        query, df = execute_with_repair(query)

        if not df.empty:
            context += f"\nStep {i+1}:\n{df.head(5).to_string()}\n"
            all_results.append(df)

    return plan, context, all_results


# -------------------------
# RISK ANALYSIS
# -------------------------
def analyze_risk(context):
    prompt = f"""
You are an AML expert.

Based on the investigation:

{context}

Identify:
- suspicious patterns
- possible money laundering indicators
- risk level (Low/Medium/High)

Be specific.
"""
    return call_llm(prompt, 0.3)


# -------------------------
# REPORT GENERATOR
# -------------------------
def generate_report(question, context, risk):
    prompt = f"""
Generate a professional AML investigation report.

Question:
{question}

Findings:
{context}

Risk Analysis:
{risk}

Include:
- Summary
- Key findings
- Risk assessment
- Recommended actions
"""
    return call_llm(prompt, 0.3)


# -------------------------
# FINAL AGENT
# -------------------------
def investigate_agent(question):
    try:
        plan, context, results = run_agent(question)
        risk = analyze_risk(context)
        report = generate_report(question, context, risk)

        final_df = pd.concat(results) if results else pd.DataFrame()

        # GRAPH GENERATION
        G = build_graph_from_df(final_df)
        graph_html = generate_pyvis_html(G)

        return plan, final_df, report, graph_html

    except Exception as e:
        return "Error", pd.DataFrame(), str(e), "<h3>Error generating graph</h3>"


# -------------------------
# GRADIO UI
# -------------------------
with gr.Blocks(title="AML GenAI Investigator") as demo:

    gr.Markdown("# 🕵️ AML GenAI Investigator (Agent Mode)")
    gr.Markdown("Multi-step AI investigation with risk intelligence + graph visualization.")

    question = gr.Textbox(
        label="Investigation Query",
        placeholder="Find offshore entities with shared officers"
    )

    run_btn = gr.Button("Investigate")

    plan_output = gr.Textbox(label="Investigation Plan")
    table_output = gr.Dataframe(label="Results")
    report_output = gr.Textbox(label="Final Report / Insight")
    graph_output = gr.HTML(label="Relationship Graph")

    def run_pipeline(q):
        return investigate_agent(q)

    run_btn.click(
        run_pipeline,
        inputs=[question],
        outputs=[plan_output, table_output, report_output, graph_output]
    )

demo.launch()