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