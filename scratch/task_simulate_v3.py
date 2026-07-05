from app_core.bridge import Bridge
import tiktoken

def simulate():
    b = Bridge()
    text = "@Exp Can you help me what fuzzy match with discription help to map the Received A-c"
    cleaned_text, mentioned_rag = b._resolve_rag_mention(text)
    actual_n_ctx = getattr(b, "actual_n_ctx", 4096)
    
    results = b.rag_manager.retrieve(mentioned_rag, cleaned_text, k=5)
    chunks = [item[0] if isinstance(item, tuple) else str(item) for item in results]
    raw_total_chunk_chars = sum(len(c) for c in chunks)
    
    rag_budget = max(2500, int(actual_n_ctx * 3.5 * 0.28))
    per_chunk_cap = max(450, rag_budget // max(1, len(chunks)))
    
    bounded_chunks = []
    used = 0
    for ch in chunks:
        piece = ch[:per_chunk_cap]
        if used + len(piece) > rag_budget:
            remain = rag_budget - used
            if remain > 10:
                bounded_chunks.append(piece[:remain])
            break
        bounded_chunks.append(piece)
        used += len(piece) + 5
    
    rag_context = "Reference context:\n" + "\n---\n".join(bounded_chunks)
    final_rag_chars = len(rag_context)
    
    system_prompt = b.get_system_prompt()
    conversation_msgs = b._get_conversation_messages()
    extra_context = ""
    if rag_context:
        extra_context = "\n\n" + rag_context
        
    full_prompt = b._build_chat_prompt(system_prompt, conversation_msgs, extra_context)
    
    # Precise tokens
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens_count = len(enc.encode(full_prompt))
    except:
        tokens_count = len(full_prompt) // 4
        
    available_tokens = 4096 - tokens_count

    print(f"raw total chunk chars: {raw_total_chunk_chars}")
    print(f"rag_budget: {rag_budget}")
    print(f"per_chunk_cap: {per_chunk_cap}")
    print(f"bounded total chars: {used}")
    print(f"final rag_context chars: {final_rag_chars}")
    print(f"full prompt chars: {len(full_prompt)}")
    print(f"tokens estimate: {tokens_count}")
    print(f"available tokens @4096: {available_tokens}")

simulate()
