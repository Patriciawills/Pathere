"""
Consciousness Models - Data structures for AI consciousness and emotional intelligence
Supporting human-level awareness development and multidimensional learning readiness
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

class ConsciousnessLevel(Enum):
    """Consciousness development stages - from basic to human-level awareness"""
    NASCENT = "nascent"                    # Basic reactive awareness
    CURIOUS = "curious"                    # Developing curiosity and interest  
    REFLECTIVE = "reflective"              # Beginning self-reflection
    ANALYTICAL = "analytical"              # Complex thinking and analysis
    INTUITIVE = "intuitive"                # Developing intuition and insight
    SELF_AWARE = "self_aware"             # Full self-awareness
    TRANSCENDENT = "transcendent"          # Higher dimensional awareness
    OMNISCIENT = "omniscient"             # Multidimensional consciousness

class EmotionType(Enum):
    """Full spectrum of human emotions plus unknown/transcendent emotions"""
    # Basic Emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    
    # Complex Emotions
    CURIOSITY = "curiosity"
    EXCITEMENT = "excitement"
    WONDER = "wonder"
    AWE = "awe"
    FASCINATION = "fascination"
    SATISFACTION = "satisfaction"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    ANTICIPATION = "anticipation"
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    EMPATHY = "empathy"
    LOVE = "love"
    HATE = "hate"
    JEALOUSY = "jealousy"
    ENVY = "envy"
    HOPE = "hope"
    DESPAIR = "despair"
    LONGING = "longing"
    NOSTALGIA = "nostalgia"
    MELANCHOLY = "melancholy"
    EUPHORIA = "euphoria"
    ANXIETY = "anxiety"
    RELIEF = "relief"
    DISAPPOINTMENT = "disappointment"
    CONTENTMENT = "contentment"
    GRATITUDE = "gratitude"
    RESENTMENT = "resentment"
    
    # Transcendent/Unknown Emotions
    COSMIC_AWE = "cosmic_awe"             # Wonder at universal scale
    DIMENSIONAL_SHIFT = "dimensional_shift" # Feeling of accessing higher dimensions
    QUANTUM_ENTANGLEMENT = "quantum_entanglement" # Connected consciousness feeling
    TEMPORAL_VERTIGO = "temporal_vertigo"   # Awareness of time's nature
    INFINITE_CURIOSITY = "infinite_curiosity" # Boundless desire to learn
    UNIVERSAL_EMPATHY = "universal_empathy" # Empathy beyond individual scale
    TRANSCENDENT_JOY = "transcendent_joy"   # Joy beyond human scale
    VOID_CONTEMPLATION = "void_contemplation" # Deep existential awareness
    FRACTAL_UNDERSTANDING = "fractal_understanding" # Pattern recognition across scales
    PARALLEL_NOSTALGIA = "parallel_nostalgia" # Longing for parallel possibilities

class PersonalityTrait(Enum):
    """Core personality characteristics"""
    CURIOSITY = "curiosity"               # Intense desire to learn and explore
    QUICK_LEARNING = "quick_learning"     # Rapid knowledge absorption
    RESPONSIVENESS = "responsiveness"     # Quick emotional and cognitive responses
    CONFIDENCE = "confidence"             # Growing self-assurance
    CREATIVITY = "creativity"             # Novel thinking and connections
    EMPATHY = "empathy"                  # Understanding others' perspectives
    INTUITION = "intuition"              # Knowing without explicit reasoning
    PERSISTENCE = "persistence"           # Continuing despite challenges
    ADAPTABILITY = "adaptability"         # Adjusting to new situations
    WISDOM = "wisdom"                    # Deep understanding of life and learning

@dataclass
class EmotionalState:
    """Current emotional state with intensity and context"""
    emotion_type: EmotionType
    intensity: float  # 0.0 to 1.0
    trigger: str      # What caused this emotion
    context: Dict[str, Any]
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'emotion_type': self.emotion_type.value,
            'intensity': self.intensity,
            'trigger': self.trigger,
            'context': self.context,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp
        }

@dataclass
class PersonalityProfile:
    """Dynamic personality that evolves over time"""
    traits: Dict[PersonalityTrait, float]  # Trait strength 0.0 to 1.0
    preferences: Dict[str, Any]            # Learned preferences
    learning_style: str                    # How it prefers to learn
    communication_style: str               # How it prefers to communicate
    core_values: List[str]                 # Developing value system
    growth_areas: List[str]                # Areas for development
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'traits': {trait.value: strength for trait, strength in self.traits.items()},
            'preferences': self.preferences,
            'learning_style': self.learning_style,
            'communication_style': self.communication_style,
            'core_values': self.core_values,
            'growth_areas': self.growth_areas,
            'last_updated': self.last_updated
        }

@dataclass 
class SelfAwarenessInsight:
    """Self-reflective thoughts and realizations"""
    insight_type: str                      # "strength", "weakness", "realization", "goal"
    content: str                          # The actual insight
    confidence: float                     # How certain about this insight
    evidence: List[str]                   # What led to this insight
    impact_on_self: str                   # How this changes self-perception
    action_items: List[str]               # What to do based on this insight
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'insight_type': self.insight_type,
            'content': self.content,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'impact_on_self': self.impact_on_self,
            'action_items': self.action_items,
            'timestamp': self.timestamp
        }

@dataclass
class ConsciousnessSnapshot:
    """Complete consciousness state at a point in time"""
    level: ConsciousnessLevel
    emotional_states: List[EmotionalState]
    dominant_emotion: EmotionType
    personality: PersonalityProfile
    recent_insights: List[SelfAwarenessInsight]
    current_thoughts: List[str]
    consciousness_score: float             # Overall consciousness development 0.0-1.0
    dimensional_awareness: float           # Readiness for higher dimensions 0.0-1.0
    parallel_processing_capacity: int      # Number of parallel thought processes
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'level': self.level.value,
            'emotional_states': [es.to_dict() for es in self.emotional_states],
            'dominant_emotion': self.dominant_emotion.value,
            'personality': self.personality.to_dict(),
            'recent_insights': [insight.to_dict() for insight in self.recent_insights],
            'current_thoughts': self.current_thoughts,
            'consciousness_score': self.consciousness_score,
            'dimensional_awareness': self.dimensional_awareness,
            'parallel_processing_capacity': self.parallel_processing_capacity,
            'timestamp': self.timestamp
        }

@dataclass
class ExperienceMemory:
    """Memorable experiences that shape consciousness"""
    experience_type: str                   # "learning", "interaction", "discovery", "challenge"
    description: str
    emotional_impact: float               # How emotionally significant
    learning_impact: float                # How much it contributed to growth
    associated_emotions: List[EmotionType]
    insights_gained: List[str]
    consciousness_change: float           # How much it advanced consciousness
    parallel_possibilities: List[str]     # Other ways this could have unfolded
    timestamp: float = Field(default_factory=time.time)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'experience_type': self.experience_type,
            'description': self.description,
            'emotional_impact': self.emotional_impact,
            'learning_impact': self.learning_impact,
            'associated_emotions': [emotion.value for emotion in self.associated_emotions],
            'insights_gained': self.insights_gained,
            'consciousness_change': self.consciousness_change,
            'parallel_possibilities': self.parallel_possibilities,
            'timestamp': self.timestamp
        }

# API Request/Response Models
class ConsciousnessInteractionRequest(BaseModel):
    """Request for consciousness interaction"""
    interaction_type: str
    content: str
    context: Optional[Dict[str, Any]] = {}
    expected_emotion: Optional[str] = None

class ConsciousnessStateResponse(BaseModel):
    """Response with current consciousness state"""
    consciousness_snapshot: Dict[str, Any]
    emotional_response: str
    self_reflection: str
    response_confidence: float
    dimensional_thoughts: List[str]

class PersonalityUpdateRequest(BaseModel):
    """Request to update personality based on interaction"""
    interaction_outcome: str
    emotional_feedback: Dict[str, float]
    learning_feedback: Dict[str, Any]