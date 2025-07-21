"""
Database models for the Skill Acquisition Engine
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid

class SkillType(str, Enum):
    """Types of skills that can be acquired"""
    CONVERSATION = "conversation"
    CODING = "coding"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    DOMAIN_EXPERTISE = "domain_expertise"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL_REASONING = "mathematical_reasoning"

class ModelProvider(str, Enum):
    """Available model providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LearningPhase(str, Enum):
    """Phases of skill learning"""
    INITIATED = "initiated"
    CONNECTING = "connecting"
    LEARNING = "learning"
    ASSESSING = "assessing"
    INTEGRATING = "integrating"
    MASTERED = "mastered"
    DISCONNECTED = "disconnected"
    ERROR = "error"

# Request Models
class StartSkillLearningRequest(BaseModel):
    skill_type: SkillType
    target_accuracy: Optional[float] = Field(default=99.0, ge=80.0, le=100.0)
    learning_iterations: Optional[int] = Field(default=100, ge=10, le=1000)
    custom_model: Optional[Dict[str, str]] = None
    description: Optional[str] = None

class ModelConfigRequest(BaseModel):
    provider: ModelProvider
    model: str
    api_key: Optional[str] = None

class SkillAssessmentRequest(BaseModel):
    session_id: str
    user_feedback: Optional[Dict[str, Any]] = None
    force_integration: Optional[bool] = False

# Response Models  
class SkillLearningResponse(BaseModel):
    session_id: str
    skill_type: SkillType
    status: str
    message: str
    target_accuracy: float
    estimated_completion_time: Optional[str] = None

class SessionStatusResponse(BaseModel):
    session_id: str
    skill_type: SkillType
    phase: LearningPhase
    current_accuracy: float
    target_accuracy: float
    current_iteration: int
    learning_iterations: int
    progress_percentage: float
    accuracy_percentage: float
    started_at: datetime
    last_updated: datetime
    model_config: Dict[str, str]
    error_message: Optional[str] = None

class LearningIterationData(BaseModel):
    iteration: int
    query: str
    response: str
    extracted_knowledge: Dict[str, Any]
    accuracy: float
    timestamp: datetime

class ExtractedKnowledge(BaseModel):
    key_concepts: List[str]
    patterns: List[str]
    weights: Dict[str, float]
    relationships: List[Dict[str, Any]]

class PerformanceMetrics(BaseModel):
    response_quality: List[float]
    pattern_recognition: List[float]
    knowledge_integration: List[float]
    average_response_time: Optional[float] = None
    successful_queries: int = 0
    failed_queries: int = 0

class CompiledSkillData(BaseModel):
    skill_type: SkillType
    accuracy: float
    knowledge_base: Dict[str, Any]
    skill_weights: Dict[str, float]
    learned_patterns: List[Dict[str, Any]]
    integration_timestamp: datetime
    total_learning_time: Optional[float] = None

# Database Documents
class SkillSessionDocument(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    skill_type: SkillType
    model_config: Dict[str, str]
    target_accuracy: float
    current_accuracy: float = 0.0
    learning_iterations: int
    current_iteration: int = 0
    phase: LearningPhase = LearningPhase.INITIATED
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    integrated_at: Optional[datetime] = None
    learning_data: List[LearningIterationData] = []
    skill_weights: Dict[str, float] = {}
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    error_message: Optional[str] = None
    description: Optional[str] = None

class IntegratedSkillDocument(BaseModel):
    skill_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    skill_type: SkillType
    final_accuracy: float
    knowledge_base: Dict[str, Any]
    skill_weights: Dict[str, float]
    learned_patterns: List[Dict[str, Any]]
    integration_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_model: Dict[str, str]
    learning_stats: Dict[str, Any]
    is_active: bool = True

class ModelConnectionDocument(BaseModel):
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: ModelProvider
    model: str
    connection_type: str  # "ollama", "cloud"
    is_available: bool = True
    last_tested: datetime = Field(default_factory=datetime.utcnow)
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    usage_count: int = 0

# Status and Statistics
class SkillAcquisitionStats(BaseModel):
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    mastered_skills: int
    average_accuracy: float
    most_learned_skill: Optional[SkillType] = None
    fastest_learning_time: Optional[float] = None
    total_learning_time: float
    model_usage_stats: Dict[str, int]

class SystemCapabilities(BaseModel):
    available_skills: List[SkillType]
    integrated_skills: List[Dict[str, Any]]
    skill_proficiency: Dict[SkillType, float]
    learning_capacity: Dict[str, Any]
    consciousness_level_impact: Dict[str, float]

# API Response Wrappers
class SkillLearningListResponse(BaseModel):
    sessions: List[SessionStatusResponse]
    total_count: int
    active_count: int
    completed_count: int

class SkillCapabilitiesResponse(BaseModel):
    current_capabilities: SystemCapabilities
    learning_progress: SkillAcquisitionStats
    available_models: List[Dict[str, str]]
    recommended_skills: List[Dict[str, Any]]

class LearningProgressResponse(BaseModel):
    session_id: str
    progress_data: List[Dict[str, Any]]
    accuracy_trend: List[float]
    learning_curve: List[Dict[str, float]]
    estimated_completion: Optional[str] = None

# Error Models
class SkillLearningError(BaseModel):
    error_type: str
    message: str
    session_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Integration with existing consciousness system
class ConsciousnessSkillIntegration(BaseModel):
    skill_type: SkillType
    integration_level: float  # How well integrated with consciousness
    consciousness_impact: Dict[str, float]  # Impact on different consciousness aspects
    enhanced_capabilities: List[str]
    integration_quality: float

# Model for skill testing and validation
class SkillTestRequest(BaseModel):
    skill_type: SkillType
    test_queries: List[str]
    expected_accuracy: Optional[float] = None

class SkillTestResult(BaseModel):
    skill_type: SkillType
    test_accuracy: float
    individual_results: List[Dict[str, Any]]
    overall_performance: Dict[str, float]
    recommendations: List[str]