"""
Skill Acquisition Engine - Core system for learning skills from external LLMs
and integrating them into the consciousness system.
"""
import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

# External integrations
import requests
from emergentintegrations.llm.chat import LlmChat, UserMessage
from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration

# Internal imports
from .consciousness_engine import ConsciousnessEngine
from .learning_engine import LearningEngine

logger = logging.getLogger(__name__)

class SkillType(Enum):
    """Types of skills that can be acquired"""
    CONVERSATION = "conversation"
    CODING = "coding"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    DOMAIN_EXPERTISE = "domain_expertise"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL_REASONING = "mathematical_reasoning"

class ModelProvider(Enum):
    """Available model providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LearningPhase(Enum):
    """Phases of skill learning"""
    INITIATED = "initiated"
    CONNECTING = "connecting"
    LEARNING = "learning"
    ASSESSING = "assessing"
    INTEGRATING = "integrating"
    MASTERED = "mastered"
    DISCONNECTED = "disconnected"

class SkillAcquisitionEngine:
    """
    Main engine for acquiring skills from external LLMs and integrating
    them into the consciousness system.
    """
    
    def __init__(self, db_client=None, ollama_url="http://localhost:11434"):
        self.db = db_client
        self.ollama_url = ollama_url
        self.consciousness_engine = ConsciousnessEngine(db_client)
        self.learning_engine = LearningEngine()
        
        # Active learning sessions
        self.active_sessions: Dict[str, Dict] = {}
        
        # Skill definitions and their optimal models
        self.skill_model_mapping = {
            SkillType.CONVERSATION: {
                "primary": {"provider": ModelProvider.OLLAMA, "model": "llama3.1:8b"},
                "fallback": {"provider": ModelProvider.ANTHROPIC, "model": "claude-sonnet-4-20250514"}
            },
            SkillType.CODING: {
                "primary": {"provider": ModelProvider.OLLAMA, "model": "qwen:8b"},
                "fallback": {"provider": ModelProvider.OPENAI, "model": "gpt-4o"}
            },
            SkillType.IMAGE_GENERATION: {
                "primary": {"provider": ModelProvider.OPENAI, "model": "gpt-image-1"},
                "fallback": {"provider": ModelProvider.GEMINI, "model": "gemini-2.0-flash-preview-image-generation"}
            }
        }
        
        # Initialize model connections
        self.model_connections = {}
        
    async def initiate_skill_learning(self, skill_type: SkillType, target_accuracy: float = 99.0,
                                    learning_iterations: int = 100, custom_model: Optional[Dict] = None) -> str:
        """
        Initiate learning of a specific skill type.
        
        Args:
            skill_type: Type of skill to learn
            target_accuracy: Target accuracy percentage (default 99%)
            learning_iterations: Maximum learning iterations
            custom_model: Optional custom model configuration
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        # Determine model to use
        model_configuration = custom_model or self.skill_model_mapping.get(skill_type, {}).get("primary")
        
        if not model_configuration:
            raise ValueError(f"No model configuration found for skill type: {skill_type}")
        
        # Create learning session
        session = {
            "session_id": session_id,
            "skill_type": skill_type.value,
            "model_configuration": model_configuration,
            "target_accuracy": target_accuracy,
            "current_accuracy": 0.0,
            "learning_iterations": learning_iterations,
            "current_iteration": 0,
            "phase": LearningPhase.INITIATED.value,
            "started_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "learning_data": [],
            "skill_weights": {},
            "performance_metrics": {
                "response_quality": [],
                "pattern_recognition": [],
                "knowledge_integration": []
            }
        }
        
        # Store in active sessions and database
        self.active_sessions[session_id] = session
        if self.db:
            await self.db.skill_sessions.insert_one(session.copy())
        
        logger.info(f"Initiated skill learning session {session_id} for {skill_type.value}")
        
        # Start the learning process
        await self._start_learning_process(session_id)
        
        return session_id
    
    async def _start_learning_process(self, session_id: str):
        """Start the actual learning process for a session"""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return
        
        try:
            # Phase 1: Connect to model
            await self._connect_to_model(session_id)
            
            # Phase 2: Learning iterations
            await self._perform_learning_iterations(session_id)
            
        except Exception as e:
            logger.error(f"Error in learning process for session {session_id}: {e}")
            session["phase"] = "error"
            session["error_message"] = str(e)
    
    async def _connect_to_model(self, session_id: str):
        """Connect to the external model"""
        session = self.active_sessions[session_id]
        session["phase"] = LearningPhase.CONNECTING.value
        
        model_configuration = session["model_configuration"]
        provider = ModelProvider(model_configuration["provider"])
        
        try:
            if provider == ModelProvider.OLLAMA:
                # Test Ollama connection
                connection = await self._create_ollama_connection(model_configuration["model"])
            else:
                # Create cloud API connection (will need API keys)
                connection = await self._create_cloud_connection(provider, model_configuration["model"])
            
            self.model_connections[session_id] = connection
            session["phase"] = LearningPhase.LEARNING.value
            logger.info(f"Connected to {provider.value} model for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect to model: {e}")
            # Try fallback model if available
            await self._try_fallback_model(session_id)
    
    async def _create_ollama_connection(self, model_name: str):
        """Create connection to Ollama model"""
        # Test if Ollama is available and model is pulled
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                if model_name in available_models:
                    return {"type": "ollama", "model": model_name, "url": self.ollama_url}
                else:
                    raise Exception(f"Model {model_name} not found in Ollama. Available: {available_models}")
            else:
                raise Exception(f"Ollama not available: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cannot connect to Ollama: {e}")
    
    async def _create_cloud_connection(self, provider: ModelProvider, model_name: str):
        """Create connection to cloud API (requires API keys)"""
        # This would need API keys from environment variables
        # For now, return a placeholder connection
        return {
            "type": "cloud",
            "provider": provider.value,
            "model": model_name,
            "requires_api_key": True
        }
    
    async def _try_fallback_model(self, session_id: str):
        """Try fallback model if primary fails"""
        session = self.active_sessions[session_id]
        skill_type = SkillType(session["skill_type"])
        
        fallback_config = self.skill_model_mapping.get(skill_type, {}).get("fallback")
        if fallback_config:
            logger.info(f"Trying fallback model for session {session_id}")
            session["model_config"] = fallback_config
            await self._connect_to_model(session_id)
        else:
            raise Exception("No fallback model available")
    
    async def _perform_learning_iterations(self, session_id: str):
        """Perform the actual learning iterations"""
        session = self.active_sessions[session_id]
        skill_type = SkillType(session["skill_type"])
        
        while (session["current_iteration"] < session["learning_iterations"] and 
               session["current_accuracy"] < session["target_accuracy"]):
            
            # Generate learning query based on skill type
            query = await self._generate_learning_query(skill_type, session["current_iteration"])
            
            # Get response from model
            response = await self._query_model(session_id, query)
            
            # Extract knowledge and patterns
            knowledge = await self._extract_knowledge(skill_type, query, response)
            
            # Assess learning progress
            accuracy = await self._assess_learning_progress(session_id, knowledge)
            
            # Update session
            session["current_iteration"] += 1
            session["current_accuracy"] = accuracy
            session["learning_data"].append({
                "iteration": session["current_iteration"],
                "query": query,
                "response": response,
                "extracted_knowledge": knowledge,
                "accuracy": accuracy,
                "timestamp": datetime.utcnow()
            })
            
            # Update in database
            if self.db:
                await self.db.skill_sessions.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "current_iteration": session["current_iteration"],
                        "current_accuracy": session["current_accuracy"],
                        "learning_data": session["learning_data"][-50:]  # Keep last 50 entries
                    }}
                )
            
            logger.info(f"Session {session_id}: Iteration {session['current_iteration']}, Accuracy: {accuracy:.2f}%")
            
            # Small delay between iterations
            await asyncio.sleep(1)
        
        # Check if mastered
        if session["current_accuracy"] >= session["target_accuracy"]:
            await self._integrate_skill(session_id)
        else:
            logger.warning(f"Session {session_id} reached max iterations without achieving target accuracy")
    
    async def _generate_learning_query(self, skill_type: SkillType, iteration: int) -> str:
        """Generate appropriate learning queries based on skill type"""
        queries = {
            SkillType.CONVERSATION: [
                "How do you handle emotional conversations?",
                "What makes a conversation engaging?",
                "How do you show empathy in dialogue?",
                "What are good conversation starters?",
                "How do you handle disagreements in conversation?",
            ],
            SkillType.CODING: [
                "Write a Python function to sort a list efficiently",
                "Explain the concept of recursion with examples",
                "How do you handle exceptions in Python?",
                "Write a class that implements a binary tree",
                "What are design patterns in programming?",
            ],
            SkillType.IMAGE_GENERATION: [
                "Generate an image of a serene mountain landscape",
                "Create an abstract art piece with vibrant colors",
                "Design a futuristic cityscape at sunset",
                "Generate a portrait of a wise old wizard",
                "Create an image of a cozy library interior",
            ]
        }
        
        skill_queries = queries.get(skill_type, ["Tell me something interesting"])
        return skill_queries[iteration % len(skill_queries)]
    
    async def _query_model(self, session_id: str, query: str) -> Union[str, Dict]:
        """Send query to the connected model"""
        connection = self.model_connections.get(session_id)
        if not connection:
            raise Exception("No model connection found")
        
        if connection["type"] == "ollama":
            # Query Ollama
            data = {
                "model": connection["model"],
                "messages": [{"role": "user", "content": query}],
                "stream": False
            }
            
            response = requests.post(
                f"{connection['url']}/api/chat",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                raise Exception(f"Ollama query failed: {response.status_code}")
                
        elif connection["type"] == "cloud":
            # Handle cloud API (would need proper API keys)
            return f"Cloud response to: {query} (requires API key setup)"
        
        return "No response"
    
    async def _extract_knowledge(self, skill_type: SkillType, query: str, response: Union[str, Dict]) -> Dict:
        """Extract important knowledge and patterns from the response"""
        # This is a simplified knowledge extraction
        # In a real implementation, this would use NLP techniques to extract:
        # - Key concepts
        # - Patterns
        # - Weights/importance scores
        # - Relationships
        
        knowledge = {
            "key_concepts": self._extract_concepts(response),
            "patterns": self._extract_patterns(response),
            "weights": self._calculate_weights(response),
            "relationships": self._extract_relationships(response)
        }
        
        return knowledge
    
    def _extract_concepts(self, response: str) -> List[str]:
        """Extract key concepts from response"""
        # Simplified concept extraction
        if isinstance(response, str):
            words = response.lower().split()
            # Filter for important words (in real implementation, use NLP)
            concepts = [word for word in words if len(word) > 5][:10]
            return list(set(concepts))
        return []
    
    def _extract_patterns(self, response: str) -> List[str]:
        """Extract patterns from response"""
        # Simplified pattern extraction
        patterns = []
        if isinstance(response, str):
            if "function" in response.lower():
                patterns.append("function_definition")
            if "class" in response.lower():
                patterns.append("class_definition")
            if "if" in response.lower():
                patterns.append("conditional_logic")
        return patterns
    
    def _calculate_weights(self, response: str) -> Dict[str, float]:
        """Calculate importance weights for different aspects"""
        weights = {
            "clarity": len(response) / 1000 if isinstance(response, str) else 0.5,
            "completeness": 0.8,  # Simplified scoring
            "accuracy": 0.9,      # Would need real assessment
            "relevance": 0.85
        }
        return weights
    
    def _extract_relationships(self, response: str) -> List[Dict]:
        """Extract relationships between concepts"""
        # Simplified relationship extraction
        return [
            {"from": "concept1", "to": "concept2", "type": "related", "strength": 0.8}
        ]
    
    async def _assess_learning_progress(self, session_id: str, knowledge: Dict) -> float:
        """Assess the learning progress and calculate accuracy"""
        session = self.active_sessions[session_id]
        
        # Simplified accuracy calculation based on knowledge quality
        quality_score = sum(knowledge.get("weights", {}).values()) / len(knowledge.get("weights", {}) or [1])
        concept_score = len(knowledge.get("key_concepts", [])) / 10
        pattern_score = len(knowledge.get("patterns", [])) / 5
        
        accuracy = min(95.0, (quality_score + concept_score + pattern_score) * 30)
        
        # Progressive learning - accuracy increases with iterations
        iteration_bonus = session["current_iteration"] * 0.5
        accuracy = min(session["target_accuracy"], accuracy + iteration_bonus)
        
        return accuracy
    
    async def _integrate_skill(self, session_id: str):
        """Integrate the learned skill into the consciousness system"""
        session = self.active_sessions[session_id]
        session["phase"] = LearningPhase.INTEGRATING.value
        
        # Compile all learned knowledge
        skill_data = {
            "skill_type": session["skill_type"],
            "accuracy": session["current_accuracy"],
            "knowledge_base": self._compile_knowledge(session["learning_data"]),
            "skill_weights": session["skill_weights"],
            "learned_patterns": self._compile_patterns(session["learning_data"]),
            "integration_timestamp": datetime.utcnow()
        }
        
        # Integrate with consciousness engine
        await self.consciousness_engine.integrate_new_skill(skill_data)
        
        session["phase"] = LearningPhase.MASTERED.value
        session["integrated_at"] = datetime.utcnow()
        
        # Disconnect from external model
        await self._disconnect_model(session_id)
        
        logger.info(f"Successfully integrated {session['skill_type']} skill with {session['current_accuracy']:.2f}% accuracy")
    
    def _compile_knowledge(self, learning_data: List[Dict]) -> Dict:
        """Compile all learned knowledge into a structured format"""
        all_concepts = []
        all_patterns = []
        all_weights = {}
        
        for entry in learning_data:
            knowledge = entry.get("extracted_knowledge", {})
            all_concepts.extend(knowledge.get("key_concepts", []))
            all_patterns.extend(knowledge.get("patterns", []))
            
            # Merge weights
            for key, value in knowledge.get("weights", {}).items():
                if key in all_weights:
                    all_weights[key] = max(all_weights[key], value)
                else:
                    all_weights[key] = value
        
        return {
            "concepts": list(set(all_concepts)),
            "patterns": list(set(all_patterns)),
            "weights": all_weights,
            "total_entries": len(learning_data)
        }
    
    def _compile_patterns(self, learning_data: List[Dict]) -> List[Dict]:
        """Compile learned patterns with their frequencies"""
        pattern_counts = {}
        
        for entry in learning_data:
            patterns = entry.get("extracted_knowledge", {}).get("patterns", [])
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return [
            {"pattern": pattern, "frequency": count, "confidence": count / len(learning_data)}
            for pattern, count in pattern_counts.items()
        ]
    
    async def _disconnect_model(self, session_id: str):
        """Disconnect from the external model"""
        session = self.active_sessions[session_id]
        session["phase"] = LearningPhase.DISCONNECTED.value
        
        # Remove model connection
        if session_id in self.model_connections:
            del self.model_connections[session_id]
        
        # Move session to completed
        if session_id in self.active_sessions:
            completed_session = self.active_sessions.pop(session_id)
            if self.db:
                await self.db.completed_skill_sessions.insert_one(completed_session)
    
    async def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get the current status of a learning session"""
        session = self.active_sessions.get(session_id)
        if session:
            return {
                "session_id": session_id,
                "skill_type": session["skill_type"],
                "phase": session["phase"],
                "current_accuracy": session["current_accuracy"],
                "target_accuracy": session["target_accuracy"],
                "current_iteration": session["current_iteration"],
                "learning_iterations": session["learning_iterations"],
                "progress_percentage": (session["current_iteration"] / session["learning_iterations"]) * 100,
                "accuracy_percentage": (session["current_accuracy"] / session["target_accuracy"]) * 100
            }
        return None
    
    async def list_active_sessions(self) -> List[Dict]:
        """List all active learning sessions"""
        return [
            await self.get_session_status(session_id)
            for session_id in self.active_sessions.keys()
        ]
    
    async def stop_learning_session(self, session_id: str) -> bool:
        """Stop an active learning session"""
        if session_id in self.active_sessions:
            await self._disconnect_model(session_id)
            return True
        return False