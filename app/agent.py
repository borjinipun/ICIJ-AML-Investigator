from app.llm import call_llm
from app.db import run_query
from app.schema import schema_context

def clean_query(query):
    query = query.strip()

    # remove markdown fences
    query = query.replace("```cypher", "")
    query = query.replace("```", "")

    return query.strip()

def generate_query(question):
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

Return ONLY the query.
"""
    return clean_query(call_llm(prompt))


def investigate(question):
    query = generate_query(question)

    result = run_query(query)

    if isinstance(result, str):
        return {"error": result, "query": query}

    explain_prompt = f"""
You are an AML investigator.

Question:
{question}

Data:
{result.head(10).to_string()}

Explain:
- What pattern is observed
- Why it is suspicious
- AML risk level (Low/Medium/High)
"""

    explanation = call_llm(explain_prompt, temperature=0.3)

    return {
        "query": query,
        "results": result.to_dict(orient="records"),
        "analysis": explanation
    }