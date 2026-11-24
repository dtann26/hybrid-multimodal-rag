import json
import logging
from typing import List, Dict, Any

from app.rag.generator import generate
from app.services.llm_switch_service import get_active_vllm_model_name

logger = logging.getLogger(__name__)

def plan_rag_queries(
    user_message: str,
    chat_history: List[Dict[str, str]],
    user_memories: List[Any]
) -> Dict[str, Any]:
    """
    Acts as a routing planner for Agentic RAG.
    Analyzes the user's input, chat history, and relevant memories to generate:
    - user_intent
    - sub_queries (breaking down multi-hop questions)
    - metadata_filters
    """
    
    # Format memories for the prompt
    memory_text = ""
    if user_memories:
        memory_lines = [f"- {m.memory_text} (importance: {m.importance})" for m in user_memories]
        memory_text = "\n".join(memory_lines)
    else:
        memory_text = "No specific memories retrieved."
        
    # Format chat history for context
    history_text = ""
    if chat_history:
        # Take the last 5 turns to keep it concise
        recent_history = chat_history[-5:]
        history_lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history]
        history_text = "\n".join(history_lines)
    else:
        history_text = "No recent chat history."

    system_prompt = """You are an advanced Query Planner for a Retrieval-Augmented Generation (RAG) system.
Your job is to analyze the user's latest message, consider their past chat history to resolve pronouns or implied context, and consider their relevant memories.
Based on this analysis, you must output a STRICT JSON object with the following schema:
{
  "user_intent": "A short string describing the user's core goal",
  "sub_queries": ["query 1", "query 2"],
  "metadata_filters": {} 
}

Instructions:
1. 'sub_queries': Break down complex or multi-hop questions into independent standard search queries. If the question is simple, provide a single query. Resolve pronouns (e.g., "what did he say" -> "what did John say") using the Chat History.
2. 'metadata_filters': Extract any explicit requests for specific sources (e.g., {"source_type": "youtube"} if they mention video/youtube). If none, leave empty.
3. OUTPUT ONLY JSON. Do not include markdown formatting like ```json or any other text.
"""

    user_prompt = f"""Chat History:
{history_text}

Relevant User Memories:
{memory_text}

Latest User Message:
{user_message}

Plan the queries and output JSON:
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Generate response
    try:
        active_model = get_active_vllm_model_name()
        # Force low temperature for structured output
        raw_response = generate(messages, temperature=0.1, max_new_tokens=500, model_path=active_model)
        
        # Clean up possible markdown code blocks
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
            
        clean_response = clean_response.strip()
        
        parsed = json.loads(clean_response)
        
        # Ensure required keys exist
        return {
            "user_intent": parsed.get("user_intent", "General"),
            "sub_queries": parsed.get("sub_queries", [user_message]),
            "metadata_filters": parsed.get("metadata_filters", {})
        }
    except Exception as e:
        logger.error(f"Failed to parse query planner LLM output: {e}. Raw response: {raw_response if 'raw_response' in locals() else 'None'}")
        # Fallback cleanly
        return {
            "user_intent": "General",
            "sub_queries": [user_message],
            "metadata_filters": {}
        }
