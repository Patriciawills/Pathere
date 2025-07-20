"""
Pydantic models for language data structures
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class LanguageType(str, Enum):
    ENGLISH = "english"
    HINDI = "hindi"
    SANSKRIT = "sanskrit"

class DataType(str, Enum):
    DICTIONARY = "dictionary"
    GRAMMAR = "grammar"
    TEXT = "text"
    WORD = "word"
    RULE = "rule"

class PartOfSpeech(str, Enum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    INTERJECTION = "interjection"
    UNKNOWN = "unknown"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# Base models
class BaseLanguageModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    language: LanguageType = LanguageType.ENGLISH
    
class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

# Dictionary Models
class DictionaryEntry(BaseLanguageModel):
    word: str = Field(..., min_length=1, max_length=100)
    definition: str = Field(..., min_length=5, max_length=1000)
    phonetic: Optional[str] = Field(None, max_length=100)
    part_of_speech: PartOfSpeech = PartOfSpeech.UNKNOWN
    examples: List[str] = Field(default_factory=list, max_items=10)
    synonyms: List[str] = Field(default_factory=list, max_items=20)
    antonyms: List[str] = Field(default_factory=list, max_items=20)
    etymology: Optional[str] = Field(None, max_length=500)
    frequency: float = Field(default=0.0, ge=0.0, le=1.0)
    learning_stage: str = Field(default="new")
    
    @validator('word')
    def word_must_be_clean(cls, v):
        if not v or not v.strip():
            raise ValueError('Word cannot be empty')
        return v.strip().lower()
    
    @validator('examples')
    def examples_must_be_clean(cls, v):
        return [example.strip() for example in v if example.strip()]
    
    @validator('synonyms', 'antonyms')
    def word_lists_must_be_clean(cls, v):
        return [word.strip().lower() for word in v if word.strip()]

class DictionaryEntryCreate(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)
    definition: str = Field(..., min_length=5, max_length=1000)
    phonetic: Optional[str] = None
    part_of_speech: Optional[PartOfSpeech] = PartOfSpeech.UNKNOWN
    examples: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    antonyms: List[str] = Field(default_factory=list)
    etymology: Optional[str] = None

class DictionaryEntryUpdate(BaseModel):
    definition: Optional[str] = Field(None, min_length=5, max_length=1000)
    phonetic: Optional[str] = None
    part_of_speech: Optional[PartOfSpeech] = None
    examples: Optional[List[str]] = None
    synonyms: Optional[List[str]] = None
    antonyms: Optional[List[str]] = None
    etymology: Optional[str] = None

# Grammar Models
class GrammarRule(BaseLanguageModel):
    rule_id: str = Field(..., min_length=1)
    rule_name: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    category: str = Field(default="general", max_length=50)
    examples: List[str] = Field(default_factory=list, max_items=15)
    exceptions: List[str] = Field(default_factory=list, max_items=10)
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    pattern: Optional[str] = Field(None, max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    
    @validator('examples', 'exceptions')
    def clean_example_lists(cls, v):
        return [example.strip() for example in v if example.strip()]

class GrammarRuleCreate(BaseModel):
    rule_name: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    category: Optional[str] = Field(default="general")
    examples: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    difficulty_level: Optional[DifficultyLevel] = DifficultyLevel.INTERMEDIATE

class GrammarRuleUpdate(BaseModel):
    description: Optional[str] = Field(None, min_length=10, max_length=2000)
    category: Optional[str] = None
    examples: Optional[List[str]] = None
    exceptions: Optional[List[str]] = None
    difficulty_level: Optional[DifficultyLevel] = None

# Query Models
class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, max_length=500)
    language: LanguageType = LanguageType.ENGLISH
    query_type: str = Field(..., regex=r'^(meaning|grammar|usage)$')
    
    @validator('query_text')
    def query_text_must_be_clean(cls, v):
        return v.strip()

class QueryResponse(BaseModel):
    query_id: str
    query_text: str
    language: LanguageType
    query_type: str
    result: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Feedback Models
class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    correction: str = Field(..., min_length=1, max_length=1000)
    feedback_type: str = Field(default="error", regex=r'^(error|improvement|suggestion)$')
    
    @validator('correction')
    def correction_must_be_clean(cls, v):
        return v.strip()

class FeedbackResponse(BaseModel):
    feedback_id: str
    query_id: str
    correction: str
    feedback_type: str
    processed: bool = False
    improvements: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# PDF Processing Models
class PDFUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    file_size: Optional[int] = None
    upload_time: datetime = Field(default_factory=datetime.utcnow)

class PDFProcessRequest(BaseModel):
    pdf_file_id: str = Field(..., min_length=1)
    page_number: Optional[int] = Field(None, ge=0)
    processing_type: str = Field(..., regex=r'^(dictionary|grammar|text)$')

class PDFProcessResponse(BaseModel):
    processing_id: str
    pdf_file_id: str
    processing_type: str
    status: str
    page_number: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Data Management Models
class DataAddRequest(BaseModel):
    data_type: str = Field(..., regex=r'^(word|rule|phrase|text)$')
    language: LanguageType = LanguageType.ENGLISH
    content: Dict[str, Any] = Field(...)
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Content cannot be empty')
        return v

class DataAddResponse(BaseModel):
    data_id: str
    data_type: str
    language: LanguageType
    status: str
    learned: bool = False
    graph_connections: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Statistics Models
class DatabaseStats(BaseModel):
    pdf_files: int = Field(default=0, ge=0)
    language_data: int = Field(default=0, ge=0)
    queries: int = Field(default=0, ge=0)
    feedback: int = Field(default=0, ge=0)

class LearningEngineStats(BaseModel):
    memory_usage: str = "0 MB"
    rules_count: int = Field(default=0, ge=0)
    vocabulary_size: int = Field(default=0, ge=0)
    successful_queries: int = Field(default=0, ge=0)
    failed_queries: int = Field(default=0, ge=0)
    total_words: int = Field(default=0, ge=0)
    total_rules: int = Field(default=0, ge=0)

class KnowledgeGraphStats(BaseModel):
    nodes_count: int = Field(default=0, ge=0)
    edges_count: int = Field(default=0, ge=0)
    languages: List[str] = Field(default_factory=list)
    node_types: List[str] = Field(default_factory=list)
    average_connections: float = Field(default=0.0, ge=0.0)

class SystemStats(BaseModel):
    memory_usage: str = "0 MB"
    active_languages: List[str] = Field(default_factory=lambda: ["english"])
    version: str = "1.0.0"

class StatsResponse(BaseModel):
    database: DatabaseStats
    learning_engine: LearningEngineStats
    knowledge_graph: KnowledgeGraphStats
    system: SystemStats
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Learning Progress Models
class VocabularyProgress(BaseModel):
    total_words: int = Field(default=0, ge=0)
    mastered_words: int = Field(default=0, ge=0)
    learning_words: int = Field(default=0, ge=0)
    new_words: int = Field(default=0, ge=0)

class LanguageProgress(BaseModel):
    language: LanguageType
    vocabulary: VocabularyProgress
    grammar_rules: int = Field(default=0, ge=0)
    last_activity: Optional[datetime] = None

class LearningSession(TimestampedModel):
    session_id: str = Field(..., min_length=1)
    language: LanguageType
    session_type: str = Field(..., regex=r'^(vocabulary|grammar|mixed)$')
    items_learned: int = Field(default=0, ge=0)
    items_reviewed: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_minutes: float = Field(default=0.0, ge=0.0)

# Text Processing Models
class TextChunk(BaseModel):
    text: str = Field(..., min_length=10)
    language: LanguageType = LanguageType.ENGLISH
    word_count: int = Field(default=0, ge=0)
    sentence_count: int = Field(default=0, ge=0)
    difficulty_level: Optional[DifficultyLevel] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessingResult(BaseModel):
    success: bool
    items_processed: int = Field(default=0, ge=0)
    items_failed: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Error Models
class ErrorResponse(BaseModel):
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationError(BaseModel):
    field: str
    message: str
    value: Optional[Any] = None