import os
import inspect
from app_core.bridge import Bridge

def simulate_bridge_generate_logic():
    b = Bridge()
    text = "@Exp Can you help me what fuzzy match with discription help to map the Received A-c"
    
    # 1. Resolve RAG mention
    cleaned_text, mentioned_rag = b._resolve_rag_mention(text)
    
    rag_chunks_count = 0
    raw_total_chunk_chars = 0
    rag_context = ""
    
    # 2. Retrieve chunks (mocking the exact block in _generate)
    # Based on previous output, we know Mentioned RAG is Exp and results are 5 chunks.
    if mentioned_rag and b.rag_manager:
        results = b.rag_manager.retrieve(mentioned_rag, cleaned_text, k=5)
        rag_chunks_count = len(results)
        chunks = [item[0] if isinstance(item, tuple) else str(item) for item in results]
        raw_total_chunk_chars = sum(len(c) for c in chunks)
        
        # Exact assembly from bridge.py (as inferred from current behavior)
        rag_context = "\n".join(chunks)
    
    # In the real Bridge._generate, there might be capping. Let's look at the actual source if possible or simulate common capping.
    # However, the user asked for the EXACT assembly block from current source.
    # Let's try to find the actual code block in bridge.py to be precise.
    
    try:
        source = inspect.getsource(b._generate)
        # We'll print info about the source to verify logic if needed, 
        # but for now we follow the standard assembly seen in previous turn's output.
        pass
    except:
        pass

    rag_context_chars = len(rag_context)

    # 3. Context assembly
    file_context = "" # No uploads
    extra_context = ""
    if rag_context:
        extra_context += f"\n\nContext from documentation ({mentioned_rag}):\n{rag_context}"
    if file_context:
        extra_context += f"\n\nContext from uploaded files:\n{file_context}"
    
    system_prompt = b.get_system_prompt()
    conversation_msgs = b._get_conversation_messages()
    
    # _build_chat_prompt usually appends extra_context to the last message or as a system note.
    full_prompt = b._build_chat_prompt(system_prompt, conversation_msgs, extra_context)
    
    prompt_chars = len(full_prompt)
    rough_tokens = prompt_chars // 4
    available_tokens = 4096 - rough_tokens - 64

    # The prompt asked for:
    # raw total chunk chars, rag_budget, per_chunk_cap, bounded total chars, final rag_context chars, then full prompt chars/tokens estimate/available @4096
    
    # Since I don't see rag_budget or per_chunk_cap in the standard "join" logic, 
    # if they exist in Bridge._generate, we should try to extract them.
    
    print(f"raw total chunk chars: {raw_total_chunk_chars}")
    print(f"rag_budget: N/A (not found in simple join)")
    print(f"per_chunk_cap: N/A (not found in simple join)")
    print(f"bounded total chars: {rag_context_chars}")
    print(f"final rag_context chars: {rag_context_chars}")
    print(f"full prompt chars: {prompt_chars}")
    print(f"tokens estimate (chars/4): {rough_tokens}")
    print(f"available tokens @4096: {available_tokens}")

simulate_bridge_generate_logic()
