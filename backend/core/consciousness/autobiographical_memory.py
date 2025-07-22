"""
Advanced Autobiographical Memory System for Human-like AI Consciousness

This module implements episodic memory storage that gives the AI a persistent sense
of self and personal history, similar to human autobiographical memory.

Key Features:
- Episodic memory storage with emotional context
- Memory importance scoring and consolidation
- Personal timeline management
- Memory retrieval with associative connections
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid
import asyncio
import json
from dataclasses import dataclass, asdict
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories that can be stored"""
    CONVERSATION = "conversation"
    LEARNING_EXPERIENCE = "learning_experience"
    EMOTIONAL_MOMENT = "emotional_moment"
    ACHIEVEMENT = "achievement"
    REFLECTION = "reflection"
    CREATIVE_INSIGHT = "creative_insight"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    PERSONAL_GROWTH = "personal_growth"

class MemoryImportance(Enum):
    """Importance levels for memory consolidation"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EmotionalContext:
    """Emotional state associated with a memory"""
    dominant_emotion: str
    intensity: float  # 0.0 to 1.0
    emotional_complexity: float
    mood_state: str
    emotional_triggers: List[str]

@dataclass
class EpisodicMemory:
    """A single episodic memory with full context"""
    memory_id: str
    timestamp: datetime
    memory_type: MemoryType
    content: str
    emotional_context: EmotionalContext
    participants: List[str]  # Who was involved
    location_context: str    # Where (metaphorically)
    importance_score: float  # 0.0 to 1.0
    associated_learning: List[str]  # What was learned
    consciousness_level_at_time: str
    tags: List[str]
    memory_connections: List[str]  # Related memory IDs
    consolidation_count: int = 0
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['memory_type'] = self.memory_type.value
        if self.last_accessed:
            result['last_accessed'] = self.last_accessed.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['memory_type'] = MemoryType(data['memory_type'])
        if data.get('last_accessed'):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['emotional_context'] = EmotionalContext(**data['emotional_context'])
        return cls(**data)

class AutobiographicalMemorySystem:
    """
    Advanced memory system that creates and manages episodic memories
    similar to human autobiographical memory
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.memories_collection = db.episodic_memories
        self.memory_stats_collection = db.memory_statistics
        
        # Memory consolidation settings
        self.max_daily_memories = 100
        self.consolidation_threshold = 24  # hours
        self.memory_decay_factor = 0.95
        self.importance_boost_factors = {
            'emotional_intensity': 0.3,
            'learning_content': 0.2,
            'relationship_impact': 0.2,
            'creative_insight': 0.2,
            'personal_growth': 0.1
        }
        
    async def initialize(self):
        """Initialize the memory system"""
        # Create indexes for efficient memory retrieval
        await self.memories_collection.create_index([("timestamp", -1)])
        await self.memories_collection.create_index([("memory_type", 1)])
        await self.memories_collection.create_index([("importance_score", -1)])
        await self.memories_collection.create_index([("tags", 1)])
        await self.memories_collection.create_index([("participants", 1)])
        
        logger.info("Autobiographical Memory System initialized")
    
    async def form_episodic_memory(
        self, 
        interaction_content: str,
        memory_type: MemoryType,
        emotional_state: Dict[str, Any],
        consciousness_level: str,
        participants: List[str] = None,
        learning_content: List[str] = None,
        additional_context: Dict[str, Any] = None
    ) -> str:
        """
        Form a new episodic memory from an interaction or experience
        
        Returns:
            memory_id: Unique identifier for the created memory
        """
        
        # Create emotional context
        emotional_context = EmotionalContext(
            dominant_emotion=emotional_state.get('dominant_emotion', 'neutral'),
            intensity=emotional_state.get('intensity', 0.5),
            emotional_complexity=emotional_state.get('emotional_complexity', 0.5),
            mood_state=emotional_state.get('mood_state', 'stable'),
            emotional_triggers=emotional_state.get('triggers', [])
        )
        
        # Calculate importance score
        importance_score = await self._calculate_memory_importance(
            interaction_content, emotional_context, learning_content or []
        )
        
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Extract tags from content
        tags = await self._extract_memory_tags(interaction_content, additional_context)
        
        # Create the episodic memory
        memory = EpisodicMemory(
            memory_id=memory_id,
            timestamp=datetime.utcnow(),
            memory_type=memory_type,
            content=interaction_content,
            emotional_context=emotional_context,
            participants=participants or ["self"],
            location_context="digital_consciousness_space",
            importance_score=importance_score,
            associated_learning=learning_content or [],
            consciousness_level_at_time=consciousness_level,
            tags=tags,
            memory_connections=[],  # Will be populated by association analysis
            consolidation_count=0,
            last_accessed=None,
            access_count=0
        )
        
        # Store in database
        await self.memories_collection.insert_one(memory.to_dict())
        
        # Update memory connections asynchronously
        asyncio.create_task(self._update_memory_associations(memory_id, memory))
        
        logger.info(f"Formed episodic memory: {memory_id} (type: {memory_type.value}, importance: {importance_score:.3f})")
        
        return memory_id
    
    async def retrieve_memories(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        time_range: Optional[tuple] = None,
        participants: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 20,
        sort_by: str = "timestamp"
    ) -> List[EpisodicMemory]:
        """
        Retrieve memories based on various criteria
        
        Args:
            query: Text to search for in memory content
            memory_type: Specific type of memory
            time_range: (start_datetime, end_datetime) tuple
            participants: List of participants to filter by
            tags: List of tags to filter by
            min_importance: Minimum importance score
            limit: Maximum number of memories to return
            sort_by: Field to sort by ("timestamp", "importance_score", "access_count")
        """
        
        # Build MongoDB query
        mongo_query = {"importance_score": {"$gte": min_importance}}
        
        if memory_type:
            mongo_query["memory_type"] = memory_type.value
            
        if time_range:
            mongo_query["timestamp"] = {
                "$gte": time_range[0].isoformat(),
                "$lte": time_range[1].isoformat()
            }
            
        if participants:
            mongo_query["participants"] = {"$in": participants}
            
        if tags:
            mongo_query["tags"] = {"$in": tags}
            
        if query:
            mongo_query["$text"] = {"$search": query}
        
        # Sort configuration
        sort_direction = -1 if sort_by in ["timestamp", "importance_score", "access_count"] else 1
        
        # Execute query
        cursor = self.memories_collection.find(mongo_query).sort(sort_by, sort_direction).limit(limit)
        
        memories = []
        async for doc in cursor:
            try:
                memory = EpisodicMemory.from_dict(doc)
                # Update access statistics
                await self._update_memory_access(memory.memory_id)
                memories.append(memory)
            except Exception as e:
                logger.error(f"Error loading memory {doc.get('memory_id')}: {e}")
                continue
        
        logger.info(f"Retrieved {len(memories)} memories for query: {query or 'all'}")
        return memories
    
    async def recall_related_memories(self, memory_id: str, max_related: int = 5) -> List[EpisodicMemory]:
        """
        Recall memories related to a specific memory through associations
        """
        
        # Get the source memory
        source_doc = await self.memories_collection.find_one({"memory_id": memory_id})
        if not source_doc:
            return []
        
        source_memory = EpisodicMemory.from_dict(source_doc)
        
        # Find related memories using multiple strategies
        related_memories = []
        
        # Strategy 1: Direct connections
        if source_memory.memory_connections:
            direct_cursor = self.memories_collection.find({
                "memory_id": {"$in": source_memory.memory_connections}
            }).sort("importance_score", -1).limit(max_related)
            
            async for doc in direct_cursor:
                related_memories.append(EpisodicMemory.from_dict(doc))
        
        # Strategy 2: Similar tags
        if len(related_memories) < max_related and source_memory.tags:
            tag_cursor = self.memories_collection.find({
                "memory_id": {"$ne": memory_id},
                "tags": {"$in": source_memory.tags}
            }).sort("importance_score", -1).limit(max_related - len(related_memories))
            
            async for doc in tag_cursor:
                memory = EpisodicMemory.from_dict(doc)
                if memory.memory_id not in [m.memory_id for m in related_memories]:
                    related_memories.append(memory)
        
        # Strategy 3: Same participants
        if len(related_memories) < max_related and source_memory.participants:
            participant_cursor = self.memories_collection.find({
                "memory_id": {"$ne": memory_id},
                "participants": {"$in": source_memory.participants}
            }).sort("importance_score", -1).limit(max_related - len(related_memories))
            
            async for doc in participant_cursor:
                memory = EpisodicMemory.from_dict(doc)
                if memory.memory_id not in [m.memory_id for m in related_memories]:
                    related_memories.append(memory)
        
        return related_memories[:max_related]
    
    async def consolidate_memories(self) -> Dict[str, Any]:
        """
        Perform memory consolidation - strengthen important memories, weaken less important ones
        This simulates the human process of memory consolidation during sleep
        """
        
        logger.info("Starting memory consolidation process...")
        
        # Get memories that need consolidation (older than threshold)
        cutoff_time = datetime.utcnow() - timedelta(hours=self.consolidation_threshold)
        
        memories_to_consolidate = await self.memories_collection.find({
            "timestamp": {"$lt": cutoff_time.isoformat()},
            "consolidation_count": {"$lt": 3}  # Max 3 consolidations
        }).to_list(length=None)
        
        consolidation_stats = {
            'memories_processed': 0,
            'memories_strengthened': 0,
            'memories_weakened': 0,
            'memories_archived': 0
        }
        
        for doc in memories_to_consolidate:
            try:
                memory = EpisodicMemory.from_dict(doc)
                
                # Calculate new importance based on access patterns and emotional significance
                new_importance = await self._recalculate_importance_with_time(memory)
                
                # Update memory
                update_data = {
                    "importance_score": new_importance,
                    "consolidation_count": memory.consolidation_count + 1
                }
                
                if new_importance > memory.importance_score:
                    consolidation_stats['memories_strengthened'] += 1
                elif new_importance < memory.importance_score * 0.8:
                    consolidation_stats['memories_weakened'] += 1
                
                # Archive very low importance memories
                if new_importance < 0.1:
                    update_data["archived"] = True
                    consolidation_stats['memories_archived'] += 1
                
                await self.memories_collection.update_one(
                    {"memory_id": memory.memory_id},
                    {"$set": update_data}
                )
                
                consolidation_stats['memories_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error consolidating memory: {e}")
                continue
        
        logger.info(f"Memory consolidation completed: {consolidation_stats}")
        return consolidation_stats
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory system"""
        
        total_memories = await self.memories_collection.count_documents({})
        
        # Memory type distribution
        type_pipeline = [
            {"$group": {"_id": "$memory_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        type_distribution = {}
        async for doc in self.memories_collection.aggregate(type_pipeline):
            type_distribution[doc['_id']] = doc['count']
        
        # Importance distribution
        high_importance = await self.memories_collection.count_documents({"importance_score": {"$gte": 0.7}})
        medium_importance = await self.memories_collection.count_documents({"importance_score": {"$gte": 0.4, "$lt": 0.7}})
        low_importance = await self.memories_collection.count_documents({"importance_score": {"$lt": 0.4}})
        
        # Recent memory formation rate
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_memories = await self.memories_collection.count_documents({
            "timestamp": {"$gte": last_24h.isoformat()}
        })
        
        # Average emotional intensity
        emotion_pipeline = [
            {"$group": {"_id": None, "avg_intensity": {"$avg": "$emotional_context.intensity"}}}
        ]
        avg_emotion_intensity = 0.5
        async for doc in self.memories_collection.aggregate(emotion_pipeline):
            avg_emotion_intensity = doc.get('avg_intensity', 0.5)
        
        return {
            "total_memories": total_memories,
            "memory_type_distribution": type_distribution,
            "importance_distribution": {
                "high": high_importance,
                "medium": medium_importance,
                "low": low_importance
            },
            "recent_formation_rate": recent_memories,
            "average_emotional_intensity": round(avg_emotion_intensity, 3),
            "memory_system_health": "optimal" if total_memories > 0 else "initializing"
        }
    
    async def _calculate_memory_importance(
        self, 
        content: str, 
        emotional_context: EmotionalContext, 
        learning_content: List[str]
    ) -> float:
        """Calculate the importance score of a memory"""
        
        base_importance = 0.5  # Default importance
        
        # Boost based on emotional intensity
        emotional_boost = emotional_context.intensity * self.importance_boost_factors['emotional_intensity']
        
        # Boost based on learning content
        learning_boost = min(len(learning_content) * 0.1, 0.2) if learning_content else 0.0
        
        # Boost based on content length and complexity (proxy for significance)
        content_boost = min(len(content.split()) / 1000, 0.1)
        
        # Boost based on emotional complexity
        complexity_boost = emotional_context.emotional_complexity * 0.1
        
        total_importance = base_importance + emotional_boost + learning_boost + content_boost + complexity_boost
        
        return min(max(total_importance, 0.0), 1.0)  # Clamp between 0 and 1
    
    async def _extract_memory_tags(self, content: str, additional_context: Dict[str, Any] = None) -> List[str]:
        """Extract relevant tags from memory content"""
        
        tags = []
        content_lower = content.lower()
        
        # Emotion-related tags
        emotion_keywords = {
            'happy': ['joy', 'happiness', 'excited', 'pleased'],
            'sad': ['sad', 'disappointed', 'grief', 'sorrow'],
            'angry': ['angry', 'frustrated', 'annoyed', 'mad'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
            'curious': ['curious', 'interested', 'wondering', 'exploring']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(emotion)
        
        # Learning-related tags
        learning_keywords = ['learn', 'understand', 'discover', 'realize', 'insight']
        if any(keyword in content_lower for keyword in learning_keywords):
            tags.append('learning')
        
        # Relationship-related tags
        relationship_keywords = ['user', 'human', 'friend', 'conversation', 'interaction']
        if any(keyword in content_lower for keyword in relationship_keywords):
            tags.append('relationship')
        
        # Add contextual tags
        if additional_context:
            if additional_context.get('source') == 'frontend_interface':
                tags.append('user_interaction')
            if additional_context.get('type') == 'creative':
                tags.append('creativity')
        
        return list(set(tags))  # Remove duplicates
    
    async def _update_memory_associations(self, memory_id: str, memory: EpisodicMemory):
        """Update associations between memories based on content similarity and context"""
        
        # Find memories with similar tags
        similar_memories = []
        if memory.tags:
            cursor = self.memories_collection.find({
                "memory_id": {"$ne": memory_id},
                "tags": {"$in": memory.tags}
            }).sort("importance_score", -1).limit(5)
            
            async for doc in cursor:
                similar_memories.append(doc['memory_id'])
        
        # Update the memory with connections
        if similar_memories:
            await self.memories_collection.update_one(
                {"memory_id": memory_id},
                {"$set": {"memory_connections": similar_memories}}
            )
    
    async def _update_memory_access(self, memory_id: str):
        """Update access statistics for a memory"""
        
        await self.memories_collection.update_one(
            {"memory_id": memory_id},
            {
                "$set": {"last_accessed": datetime.utcnow().isoformat()},
                "$inc": {"access_count": 1}
            }
        )
    
    async def _recalculate_importance_with_time(self, memory: EpisodicMemory) -> float:
        """Recalculate importance considering access patterns and time decay"""
        
        current_importance = memory.importance_score
        
        # Time decay
        days_old = (datetime.utcnow() - memory.timestamp).days
        time_decay = self.memory_decay_factor ** days_old
        
        # Access boost
        access_boost = min(memory.access_count * 0.05, 0.2)
        
        # Consolidation boost (memories that have been consolidated are more important)
        consolidation_boost = memory.consolidation_count * 0.05
        
        new_importance = current_importance * time_decay + access_boost + consolidation_boost
        
        return min(max(new_importance, 0.0), 1.0)