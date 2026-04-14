import kuzu
import os

db_path = os.getenv("DB_PATH", "data/icij_graph_db")

db = kuzu.Database(db_path)
conn = kuzu.Connection(db)

def run_query(query):
    try:
        return conn.execute(query).get_as_df()
    except Exception as e:
        return str(e)