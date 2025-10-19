# hybrid_chat.py
from datetime import datetime
import re
import json
import config
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import traceback
from functools import lru_cache 
# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
try:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Connect to Pinecone index
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating managed index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
        )

    index = pc.Index(INDEX_NAME)

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
except Exception as e:
    print(f"FATAL: Could not initialize clients. Error: {e}")
    traceback.print_exc()
    exit(1)

# -----------------------------
# Helper functions
# -----------------------------
@lru_cache(maxsize=128)  # Caching embedded calls 
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string."""
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        return resp.data[0].embedding
    except Exception as e:
        print(f"Error in embed_text: {e}")
        return []

@lru_cache(maxsize=32) #For caching already embedded word 
def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    try:
        print("Top k elements =============> ",TOP_K)
        vec = embed_text(query_text)
        res = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        print(f"DEBUG: Pinecone top {top_k} results:")
        print(len(res["matches"]))
        return res["matches"]
    except Exception as e:
        print(f"Error in pinecone_query: {e}")
        traceback.print_exc()
        return []




# Assuming 'driver' is your Neo4j driver instance, e.g.,
# from neo4j import GraphDatabase
# driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

@lru_cache(maxsize=32) # <-- Make sure to re-apply the cache decorator
def fetch_graph_context(node_ids: tuple, neighborhood_depth=1):
    """
    Fetch neighboring nodes from Neo4j efficiently, up to a dynamic depth,
    using a single batched query. (Now fixed to address server warnings).
    """
    facts = []
    
    try:
        depth = int(neighborhood_depth)
        if depth < 1:
            depth = 1
    except (ValueError, TypeError):
        depth = 1

    # Dynamically create the path-matching part of the query
    path_match_query = f"MATCH p = (n)-[*1..{depth}]-(m:Entity)"
    
    # --- THIS PART IS FIXED ---
    # We now generate 'type' from the labels list, filtering out 'Entity'
    subquery_return = (
        "RETURN length(p) AS path_len, "
        "       'RELATED_AT_DEPTH_' + toString(length(p)) AS rel, "
        "       labels(m) AS labels, m.id AS id, m.name AS name, "
        "       [l IN labels(m) WHERE l <> 'Entity'][0] AS type, "  
        "       m.description AS description"
    )
    # ---------------------------

    # Assemble the final query
    q = (
        "UNWIND $node_ids AS nid "
        "MATCH (n:Entity {id: nid}) "
        "CALL (n) { "  
        # "    WITH n " <-- FIX 1 (part b): This line is now removed.
        f"   {path_match_query} "
        "    WHERE n <> m "
        f"   {subquery_return} "
        "    ORDER BY path_len ASC "
        "    LIMIT 10 "
        "} "
        "RETURN nid AS source, rel, labels, id, name, type, description"
    )
    
    try:
        with driver.session() as session:
            recs = session.run(q, node_ids=list(node_ids))
            
            for r in recs:
                facts.append({
                    "source": r["source"],
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    # This now works because our query generates the 'type' field
                    "target_type": r["type"], 
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
                
        print(f"DEBUG: Graph facts fetched: {len(facts)}")
        return facts
    except Exception as e:
        print(f"Error in fetch_graph_context: {e}")
        traceback.print_exc()
        return []




def build_prompt(user_query, previous_out, extra_output, pinecone_matches, graph_facts):
    """
    Build a detailed and explicit chat prompt combining vector DB matches and graph facts
    for a travel assistant.
    """
    if extra_output== None : extra_output= "No output"
    # 1. SYSTEM PROMPT: More elaborated instructions for the AI
    system = (
        "You are an expert travel assistant. Your mission is to provide concise, "
        "actionable, and fact-based travel advice. "
        "You MUST strictly follow these rules:\n\n"
        "1.  **Use Provided Context Only:** Base your entire answer *only* on the "
        "    <semantic_matches> and <graph_facts> provided. Do not use external knowledge.\n\n"
        "2.  **Explain Your Sources:** Use the context sections to answer:\n"
        "    - `<semantic_matches>` (Vector DB): Use these to identify the main places, "
        "      activities, or concepts the user is asking about.\n"
        "    - `<graph_facts>` (Graph DB): Use these to find *relationships*, "
        "      like what's nearby, what's inside a city, or what activities a place offers. "
        "      This is how you build an itinerary.\n\n"
        "3.  **Cite Your Sources:** This is critical. When you mention any place or attraction "
        "    from the context, you MUST cite its ID in brackets, like `[id: 123]` or `[id: p4]`. \n\n"
        "4.  **Be Actionable:** If the user asks for suggestions, provide 2-3 concrete "
        "    itinerary steps or tips. Do not just list facts.\n\n"
        "5.  **Be Concise & Direct:** Keep your answer to the point. Start the answer directly "
        "    without conversational fluff (e.g., no 'Sure, I can help with that!')."
    )

    # 2. CONTEXT FORMATTING (Same as your original, this is good)
    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", None)
        # Using f-strings for cleaner formatting
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score:.4f}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    # 3. HISTORY FORMATTING
    history_section = ""
    if previous_out:
        history_str = "\n".join(previous_out)
        history_section = f"<chat_history>\n{history_str}\n{extra_output}\n</chat_history>\n\n"

    # 4. USER PROMPT: Uses clear XML tags to separate context
    user_prompt_content = (
        f"Please answer the following query based *only* on the provided context.\n\n"
        f"<user_query>\n{user_query}\n</user_query>\n\n"
        f"{history_section}"
        f"<semantic_matches>\n"
        "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n"
        f"</semantic_matches>\n\n"
        f"<graph_facts>\n"
        "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n"
        f"</graph_facts>\n\n"
        "Answer:"
    )

    # 5. FINAL PROMPT ASSEMBLY
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt_content},
    ]
    
    return prompt


def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=prompt_messages,
            max_tokens=600,
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error in call_chat: {e}")
        traceback.print_exc()
        return "I'm sorry, I encountered an error while generating a response."


def memory_prompt(conversation_memory , user_input_query  , gpt_outputs ):
    """
    Build stores the history of whole session of the conversation ,
    and decides if the need to fetch the database or previous output would do the work
    """
    
    if gpt_outputs : system = {"role":"system","content":gpt_outputs} ; conversation_memory.append(system)
    if user_input_query: user = {"role":"user","content":user_input_query} ;conversation_memory.append(user)
    
    
    if user_input_query : 

        final_prompt = f"""
                        You are a query analysis assistant. Analyze the `user_query` and `chat_history`.
                        Your response MUST be ONLY the following JSON object, based on these rules:

                        1.  `content`:
                            * If 100% answerable from history (e.g., "thanks", "what was that?"), set to the *full answer*.
                            * If references history but needs new data (e.g., "what *else* is in Hanoi?"), set to a *brief context summary*.
                            * Otherwise, MUST be `null`.

                        2.  `continue`:
                            * Set to `true` if new data is needed (i.e., query is not 100% answerable by history).
                            * Set to `false` *only if* `content` contains the full and final answer.

                        3.  `top_k`:
                            * Extract integer if user asks for a number (e.g., "top 3", "give me 5").
                            * Otherwise, `null`.

                        4.  `depth`:
                            * An integer from 1-5. Default to `1` if unsure.
                            * `1`: Default. Standard queries ("What's in...", "Tell me about...").
                            * `2`: Implied 2-hop ("What's near the *attractions in*...").
                            * `3`: "Deep dive", "all related things".
                            * `4-5`: Rare, "exhaustive", "every possible connection".

                        ---
                        **User Query:**
                        {user_input_query}
                        ---

                        **Example 1 (New Query):**
                        *User Query: "What are the top 3 sights in Hanoi?"*
                        ```json
                        {{
                        "content": null,
                        "continue": true,
                        "top_k": 3,
                        "depth": 1
                        }}
                        """
        conversation_memory.append({"role":"user","content":final_prompt})
    
    return conversation_memory

def parse_to_json(gpt_memory_prompt_res):
    try:
        pattern = r"```json(.*?)```"
        ans =  re.findall(pattern, gpt_memory_prompt_res, re.DOTALL)
        ans = ans[0].strip().removeprefix("```json").removesuffix("```").strip()
        ans = ans.replace("'", '"')
        ans = json.loads(ans)
        if ans.get('top_k') == None or ans.get('top_k') > 10 :ans['top_k']= TOP_K
        return ans
    except Exception as e:
        traceback.print_exc()        
        print(f"Exception At prase_to_json :{e} ,  ")
        return {'content':gpt_memory_prompt_res, 'continue':True ,'top_k':TOP_K,"depth":1}



# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    output = "You are an expert travel assistant. Your mission is to provide concise, actionable, and fact-based travel advice."
    conversation_memory = []
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower().strip() in ("exit","quit"):
            break
        prompt = memory_prompt(conversation_memory, query,output)
        # print("Prompt added to Conversation list ........  ")
        ans = parse_to_json(call_chat(prompt))
        if(ans["continue"]):
            print("Hit dataset ............. ")
            start_time = datetime.now()
            matches = pinecone_query(query, top_k=TOP_K)            # print("pinecone completed ..................", matches)
            match_ids = tuple(m["id"] for m in matches)
            graph_facts = fetch_graph_context(match_ids,ans.get("depth",1))
            print(f"Graphs Facts ==>{len(graph_facts)}, in time {datetime.now()-start_time}")
            extra_output = ans.get("content")
            if(output.__eq__("You are an expert travel assistant. Your mission is to provide concise, actionable, and fact-based travel advice.")): output = None
            prompt = build_prompt(query,output,extra_output, matches, graph_facts)
            answer = call_chat(prompt)
            output = answer
            print("\n=== Assistant Answer ===\n")
            print(answer)
            print("\n=== End ===\n")
            if(len(conversation_memory)>2): conversation_memory.pop()
        else:
            print("Dataset skipped ............. ")
            
            print("\n=== Assistant Answer ===\n")
            print(ans["content"])
            memory_prompt(conversation_memory,None,ans["content"])
            # print("Added to memory prompt(other condition) ..........")
            print("\n=== End ===\n")
            if(len(conversation_memory)>2): conversation_memory.pop()

if __name__ == "__main__":
    interactive_chat()


