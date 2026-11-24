import math
from datetime import datetime, timezone
from typing import List

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models import UserMemory

class MemoryManager:
    """
    Manages user memories with an exponential decay algorithm for retrieval.
    """
    def __init__(self, decay_rate: float = 0.05, alpha: float = 0.5, beta: float = 0.5):
        # decay_rate: How quickly memory recency weight decays over time
        self.decay_rate = decay_rate
        # alpha: Weight for textual similarity
        self.alpha = alpha
        # beta: Weight for importance & recency combined
        self.beta = beta

    def _calculate_similarity(self, query: str, memory_text: str) -> float:
        """
        Calculates simple text overlap similarity (Overlap coefficient).
        """
        query_words = set(query.lower().split())
        memory_words = set(memory_text.lower().split())
        
        if not query_words or not memory_words:
            return 0.0
            
        intersection = query_words.intersection(memory_words)
        return len(intersection) / len(query_words)

    def add_memory(self, db: Session, user_id: str, text: str, importance: float = 5.0) -> UserMemory:
        """
        Adds a new memory for a user to the database.
        """
        now = datetime.now(timezone.utc)
        memory = UserMemory(
            user_id=user_id,
            memory_text=text,
            importance=importance,
            created_at=now,
            last_accessed_at=now,
            memory_type="core"
        )
        db.add(memory)
        db.commit()
        db.refresh(memory)
        return memory

    def get_relevant_memories(self, db: Session, user_id: str, query_text: str, top_k: int = 5) -> List[UserMemory]:
        """
        Retrieves top_k memories for a user based on similarity, importance, and recency.
        Updates the last_accessed_at timestamp for the retrieved memories.
        """
        stmt = select(UserMemory).where(UserMemory.user_id == user_id)
        memories = db.execute(stmt).scalars().all()
        
        if not memories:
            return []
            
        now = datetime.now(timezone.utc)
        scored_memories = []
        
        for mem in memories:
            # 1. Evaluate base similarity
            similarity = self._calculate_similarity(query_text, mem.memory_text)
            
            # 2. Recency weight
            # Ensure timezone awareness for datetime math
            last_acc = mem.last_accessed_at
            if last_acc.tzinfo is None:
                last_acc = last_acc.replace(tzinfo=timezone.utc)
                
            age_in_days = max(0.0, (now - last_acc).total_seconds() / 86400.0)
            recency_weight = math.exp(-self.decay_rate * age_in_days)
            
            # 3. Final Score: (alpha * similarity) + (beta * importance * recency_weight)
            score = (self.alpha * similarity) + (self.beta * mem.importance * recency_weight)
            
            scored_memories.append((score, mem))
            
        # Sort by score descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [m for score, m in scored_memories[:top_k]]
        
        # 4. When a memory is retrieved, update its last_accessed_at timestamp
        if top_memories:
            for m in top_memories:
                m.last_accessed_at = datetime.now(timezone.utc)
            db.commit()
            # refresh? not strictly necessary if just returning them, but good practice
            for m in top_memories:
                db.refresh(m)
                
        return top_memories
