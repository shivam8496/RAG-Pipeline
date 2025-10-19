# hybrid_chat.py 
from datetime import datetime
import json
import config
from async_lru import alru_cache
from typing import List, Dict 
from openai import AsyncOpenAI  
from pinecone import Pinecone, ServerlessSpec
from neo4j import AsyncGraphDatabase  
import traceback
import asyncio  

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
    
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)  
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating managed index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
        )
    index = pc.Index(INDEX_NAME)

    
    driver = AsyncGraphDatabase.driver(  
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    print("Async clients initialized.")
    

except Exception as e:
    print(f"FATAL: Could not initialize clients. Error: {e}")
    traceback.print_exc()
    exit(1)

# -----------------------------
# Helper functions
# -----------------------------


@alru_cache(maxsize=128)
async def embed_text(text: str) -> List[float]:  
    """Get embedding for a text string."""
    try:
        # Use await and the async client
        resp = await client.embeddings.create(model=EMBED_MODEL, input=[text]) 
        return resp.data[0].embedding
    except Exception as e:
        print(f"Error in embed_text: {e}")
        return []

@alru_cache(maxsize=32)
async def pinecone_query(query_text: str, top_k=TOP_K):  
    """Query Pinecone index using embedding."""
    try:
        print("Top k elements =============> ", top_k) 
        vec = await embed_text(query_text)  
        
        # Pinecone client is SYNC, so i ran it in a threadpool
        # to avoid blocking the asyncio event loop.
        res = await asyncio.to_thread(  
            index.query,
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

@alru_cache(maxsize=32)
async def fetch_graph_context(node_ids: tuple, neighborhood_depth=1):  
    """
    Fetch neighboring nodes from Neo4j efficiently (async).
    """
    facts = []
    
    try:
        depth = int(neighborhood_depth)
        if depth < 1:
            depth = 1
    except (ValueError, TypeError):
        depth = 1

    path_match_query = f"MATCH p = (n)-[*1..{depth}]-(m:Entity)"
    subquery_return = (
        "RETURN length(p) AS path_len, "
        "       'RELATED_AT_DEPTH_' + toString(length(p)) AS rel, "
        "       labels(m) AS labels, m.id AS id, m.name AS name, "
        "       [l IN labels(m) WHERE l <> 'Entity'][0] AS type, "
        "       m.description AS description"
    )
    q = (
        "UNWIND $node_ids AS nid "
        "MATCH (n:Entity {id: nid}) "
        "CALL (n) { "
        f"   {path_match_query} "
        "    WHERE n <> m "
        f"   {subquery_return} "
        "    ORDER BY path_len ASC "
        "    LIMIT 10 "
        "} "
        "RETURN nid AS source, rel, labels, id, name, type, description"
    )
    
    try:
        
        async with driver.session() as session:  
            recs = await session.run(q, node_ids=node_ids)  
            
            async for r in recs:  
                facts.append({
                    "source": r["source"],
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
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



def build_rag_prompt(user_query: str, chat_history: List[Dict], extra_output: str, pinecone_matches, graph_facts):
    """
    Build a detailed and explicit chat prompt combining vector DB matches and graph facts
    for a travel assistant.
    """
    if extra_output is None: 
        extra_output = "No output"
        
    system = (
        "You are an expert travel assistant 'Trax'. Provide concise, actionable, fact-based advice using ONLY the provided context. Cite sources like `[id: 123]`."
        
        #  ADD CHAIN-OF-THOUGHT INSTRUCTIONS 
        "Before generating the final answer, perform a step-by-step analysis within <thinking> tags:"
        "1. **Identify User Intent:** What is the user's main goal or question?"
        "2. **Extract Key Entities:** List the main places/concepts from <semantic_matches> relevant to the intent."
        "3. **Analyze Relationships:** Use <graph_facts> to find connections (nearby, part of, etc.) between the key entities."
        "4. **Synthesize Plan:** Based ONLY on steps 1-3 and the provided context, outline the core points of the answer, including citations."
        
        "After your thinking process, provide the final answer directly to the user, following all rules below. DO NOT include the <thinking> tags in your final output."
        #  END CHAIN-OF-THOUGHT INSTRUCTIONS 
        
        "\n\n**Rules:**\n"
        "1. **Use Provided Context Only:** Base your answer *only* on <semantic_matches> and <graph_facts>.\n"
        "2. **Cite Sources:** Crucial. Cite IDs like `[id: 123]` or `[id: p4]` for any mentioned place.\n"
        "3. **Be Actionable:** Provide 2-3 concrete itinerary steps or tips if asked for suggestions.\n"
        "4. **Be Concise & Direct:** Start answers directly, no conversational fluff.\n"
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m.get("metadata", {})
        score = m.get("score", 0.0)
        snippet = f"- id: {m.get('id', 'N/A')}, name: {meta.get('name','N/A')}, type: {meta.get('type','N/A')}, score: {score:.4f}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)
    vec_context_str = "\n".join(vec_context) or "No semantic matches found."

    graph_context = [
        f"- ({f.get('source')}) -[{f.get('rel')}]-> ({f.get('target_id')}) {f.get('target_name')}: {f.get('target_desc')}"
        for f in graph_facts
    ]
    graph_context_str = "\n".join(graph_context) or "No graph facts found."

    history_section = ""
    if chat_history:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        history_section = f"<chat_history>\n{history_str}\n</chat_history>\n\n"
        
    if extra_output:
        history_section += f"<gatekeeper_notes>\n{extra_output}\n</gatekeeper_notes>\n\n"

    user_prompt_content = (
        f"Please answer the following query based *only* on the provided context.\n\n"
        f"<user_query>\n{user_query}\n</user_query>\n\n"
        f"{history_section}"
        f"<semantic_matches>\n{vec_context_str}\n</semantic_matches>\n\n"
        f"<graph_facts>\n{graph_context_str}\n</graph_facts>\n\n"
        "Answer:"
    )

    prompt = [
        {"role": "system", "content": system},
        *chat_history,
        {"role": "user", "content": user_prompt_content},
    ]
    
    return prompt


async def call_chat(prompt_messages):  
    """Call OpenAI ChatCompletion."""
    try:
        resp = await client.chat.completions.create(  
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



def create_gatekeeper_prompt(conversation_memory: List[Dict], user_input_query: str) -> List[Dict]:
    """Creates the prompt for the gatekeeper LLM."""
    history_str = json.dumps(conversation_memory[-4:], indent=2)
    print("History ===>>", history_str)
    final_prompt = f"""
    You are a query analysis assistant. Analyze the `user_query` and `chat_history`.
    Your response MUST be ONLY the following JSON object, based on these rules:

    1.  `content`:
        * If 100% answerable from history, set to the *full answer*.
        * If references history but needs new data, set to a *brief context summary*.
        * Otherwise, MUST be `null`.

    2.  `continue`:
        * Set to `true` if new data is needed.
        * Set to `false` *only if* `content` contains the full and final answer.

    3.  `top_k`:
        * Extract integer if user asks for a number (e.g., "top 3", "give me 5").
        * Otherwise, `null`.

    4.  `depth`:
        * An integer from 1-5. Default to `1`.
        * `1`: Standard queries.
        * `2-3`: Deeper queries ("hidden gems", "deep dive").
        * `4-5`: Rare, exhaustive queries.
    ---
    **Chat History:**
    {history_str}
    **User Query:**
    {user_input_query}
    ---
    """
    return [{"role": "system", "content": final_prompt}]


async def call_gatekeeper_llm(prompt_messages: List[Dict]) -> Dict: 
    """Calls the gatekeeper LLM in JSON mode for a robust, parsable response."""
    try:
        resp = await client.chat.completions.create(  
            model=CHAT_MODEL,
            messages=prompt_messages,
            max_tokens=300,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        ans = json.loads(resp.choices[0].message.content)
        
        
        if ans.get('top_k') is None or not isinstance(ans.get('top_k'), int):
            ans['top_k'] = TOP_K
        if ans.get('depth') is None or not isinstance(ans.get('depth'), int):
            ans['depth'] = 1
        if 'content' not in ans:
            ans['content'] = None
        if 'continue' not in ans:
            ans['continue'] = True
        return ans
    except Exception as e:
        traceback.print_exc()        
        print(f"Exception At call_gatekeeper_llm: {e} , falling back to default.")
        return {'content': None, 'continue': True, 'top_k': TOP_K, "depth": 1}

async def summarize_history_chunk(history_chunk: List[Dict]) -> str:
    """
    Uses an LLM call to summarize a list of conversation messages.
    """
    if not history_chunk:
        return "No history to summarize."

    print(f"DEBUG: Summarizing {len(history_chunk)} middle messages...")
    
    
    transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_chunk])

    summary_prompt_messages = [
        {"role": "system", "content": (
            "You are a summarization assistant. "
            "Concisely summarize the key topics, entities, and user preferences mentioned "
            "in the following conversation excerpt. Focus on information that might be "
            "relevant for future context in a travel chat. Output only the summary."
        )},
        {"role": "user", "content": f"<conversation_excerpt>\n{transcript}\n</conversation_excerpt>\n\nSummary:"}
    ]

    try:
        summary = await call_chat(summary_prompt_messages)
        print("DEBUG: Middle history summary created.")
        return summary
    except Exception as e:
        print(f"Error summarizing history chunk: {e}")
        traceback.print_exc()
        return "[Summary unavailable due to an error]"

# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat():  
    print("Hybrid travel assistant. Type 'exit' to quit.")
    conversation_memory: List[Dict] = []

    while True:
        # Use asyncio.to_thread to run the blocking input()
        query = await asyncio.to_thread(input, "\nEnter your travel question: ")  
        query = query.strip()
        
        if not query or query.lower().strip() in ("exit","quit"):
            break

        gatekeeper_prompt = create_gatekeeper_prompt(conversation_memory, query)
        ans = await call_gatekeeper_llm(gatekeeper_prompt)  
        print(f"DEBUG: Gatekeeper decision: {ans}")
        if ans["continue"]:
            print("Hit dataset (in parallel) ............. ")
            start_time = datetime.now()
            
            # --- PARALLEL EXECUTION ---
            # Create tasks for both I/O-bound operations
            pinecone_task = asyncio.create_task(
                pinecone_query(query, top_k=ans.get('top_k', TOP_K))
            )
            

            start_time = datetime.now()
            print("DEBUG: Calling gatekeeper and embedding in parallel...")
            
            gatekeeper_prompt = create_gatekeeper_prompt(conversation_memory, query)
            
            # Create tasks for the two independent operations
            task_gatekeeper = asyncio.create_task(call_gatekeeper_llm(gatekeeper_prompt))
            task_embed = asyncio.create_task(embed_text(query))
            
            # Wait for both to complete
            ans, query_vector = await asyncio.gather(task_gatekeeper, task_embed) 
            
            print(f"DEBUG: Parallel step complete. Time: {datetime.now() - start_time}")
            print(f"DEBUG: Gatekeeper decision: {ans}")

            if ans["continue"]:
                print("Hit dataset (sequentially) ............. ")
                start_time_db = datetime.now()
                
                #  using the *pre-fetched* vector to query Pinecone
                try:
                    res = await asyncio.to_thread(  # Run sync pinecone in thread
                        index.query,
                        vector=query_vector,
                        top_k=ans.get('top_k', TOP_K),
                        include_metadata=True,
                        include_values=False
                    )
                    matches = res["matches"]
                except Exception as e:
                    print(f"Error in parallel pinecone_query: {e}")
                    matches = []
                
                match_ids = tuple(m["id"] for m in matches)
                
                #  fetching graph facts
                graph_facts = await fetch_graph_context(match_ids, ans.get("depth", 1))
                
                print(f"Graphs Facts ==>{len(graph_facts)}, DB time {datetime.now()-start_time_db}")
                extra_output = ans.get("content")
                
                if len(conversation_memory) > 25: 
                    head = conversation_memory[:7] 
                    tail = conversation_memory[-7:] 
                    middle = conversation_memory[7:-7] 

                    # Calling the summarizer function asynchronously
                    summary_str = await summarize_history_chunk(middle)

                    # Creating a system message containing the summary
                    summary_message = {
                        "role": "system",
                        "content": f"<summary_of_omitted_conversation>\n{summary_str}\n</summary_of_omitted_conversation>"
                    }

                    # Construct the final context history for the RAG prompt
                    context_history = head + [summary_message] + tail
                    
                    print(f"DEBUG: Using summarized history (10 head + summary + 10 tail), total {len(context_history)} items.")
                
                else:
                    # Use the full history if it's short
                    context_history = conversation_memory
                    print(f"DEBUG: Using full history ({len(context_history)} messages).")
                # --- END UPDATED HISTORY MANAGEMENT ---

                
                prompt = build_rag_prompt(
                    query,
                    context_history, 
                    extra_output,
                    matches,
                    graph_facts
                )
                answer = await call_chat(prompt)

            else:
                print("Dataset skipped ............. ")
                answer = ans["content"]
            
        else:
            # This is the "continue: false" case from the *first* gatekeeper
            print("Dataset skipped ............. ")
            answer = ans["content"]
            
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")
        
        conversation_memory.append({"role": "user", "content": query})
        conversation_memory.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    print("Starting async chat...")
    try:
        asyncio.run(interactive_chat())  
    except KeyboardInterrupt:
        print("\nExiting chat.")