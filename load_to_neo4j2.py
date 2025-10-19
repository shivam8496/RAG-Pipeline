# load_to_neo4j_no_apoc.py
import json
from neo4j import GraphDatabase
from tqdm import tqdm
import config
from collections import defaultdict

DATA_FILE = "./vietnam_travel_dataset.json"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def create_constraints(tx):
    # Constraint on the generic :Entity label
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    
    # Add specific constraints for known types (this is good practice)
    # Note: These are optional but help speed up the MERGE later
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Attraction) REQUIRE a.id IS UNIQUE")
    print("Constraints created.")

def batch_upsert_nodes(tx, nodes_by_type):
    """
    Upserts nodes in batches, one batch per node type.
    We use an f-string for the label, which is safe because we control the input.
    """
    for node_type, batch in nodes_by_type.items():
        # Dynamically insert the label (e.g., :City)
        # We also add the :Entity label to all nodes
        cypher_upsert = f"""
        UNWIND $batch as node_props
        MERGE (n:Entity:{node_type} {{id: node_props.id}})
        SET n += node_props
        """
        tx.run(cypher_upsert, batch=batch)

def batch_create_relationships(tx, rels_by_type):
    """
    Creates relationships in batches, one batch per relationship type.
    """
    for rel_type, batch in rels_by_type.items():
        # Dynamically insert the relationship type (e.g., :CONNECTED_TO)
        cypher_create = f"""
        UNWIND $batch as rel
        MATCH (a:Entity {{id: rel.source_id}})
        MATCH (b:Entity {{id: rel.target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        tx.run(cypher_create, batch=batch)

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    print(f"Loaded {len(nodes)} nodes from JSON.")

    # --- Pre-process data in Python ---
    
    # 1. Prepare nodes for batching
    # We group nodes by type (e.g., "City")
    # and clean their properties
    nodes_by_type = defaultdict(list)
    for node in nodes:
        node_type = node.get("type", "Unknown")
        # Clean properties: remove 'connections' and 'type'
        props = {k:v for k,v in node.items() if k not in ("connections", "type")}
        nodes_by_type[node_type].append(props)

    # 2. Prepare relationships for batching
    # We group relationships by their type (e.g., "CONNECTED_TO")
    rels_by_type = defaultdict(list)
    for node in nodes:
        source_id = node['id']
        for rel in node.get("connections", []):
            rel_type = rel.get("relation", "RELATED_TO")
            target_id = rel.get("target")
            if target_id:
                rels_by_type[rel_type].append({
                    "source_id": source_id,
                    "target_id": target_id
                })
    
    print(f"Found {sum(len(v) for v in rels_by_type.values())} relationships to create.")

    # --- Run Batched Transactions ---
    with driver.session() as session:
        # 1. Single transaction for constraints
        session.execute_write(create_constraints)
        
        # 2. One transaction for all node batches
        print("Creating/updating all nodes...")
        session.execute_write(batch_upsert_nodes, nodes_by_type)
        print("Node batches complete.")
        
        # 3. One transaction for all relationship batches
        print("Creating/updating all relationships...")
        session.execute_write(batch_create_relationships, rels_by_type)
        print("Relationship batches complete.")

    print("Done loading into Neo4j.")

if __name__ == "__main__":
    main()