from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import json

# Import our custom modules
from core.ocr_engine import OCREngine
from core.learning_engine import LearningEngine
from core.knowledge_graph import KnowledgeGraph
from core.dataset_manager import DatasetManager
from models.language_models import *

# Import consciousness models
from models.consciousness_models import (
    ConsciousnessInteractionRequest, ConsciousnessStateResponse, 
    PersonalityUpdateRequest
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize core components
ocr_engine = OCREngine()
learning_engine = LearningEngine()
knowledge_graph = KnowledgeGraph()
dataset_manager = DatasetManager()

# Create the main app without a prefix
app = FastAPI(title="Minimalist Grammar & Vocabulary Engine", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# API Models
class ProcessPDFRequest(BaseModel):
    pdf_file_id: str
    page_number: Optional[int] = None
    processing_type: str = "dictionary"  # "dictionary" or "grammar"

class AddDataRequest(BaseModel):
    data_type: str  # "word", "rule", "phrase"
    language: str = "english"
    content: Dict[str, Any]

class QueryRequest(BaseModel):
    query_text: str
    language: str = "english"
    query_type: str = "meaning"  # "meaning", "grammar", "usage"

class FeedbackRequest(BaseModel):
    query_id: str
    correction: str
    feedback_type: str = "error"  # "error", "improvement"

# Core API Endpoints

@api_router.get("/")
async def root():
    return {"message": "Minimalist Grammar & Vocabulary Engine API", "version": "1.0.0"}

@api_router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF file (dictionary/grammar book) for processing"""
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = Path(f"/tmp/{file_id}.pdf")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store file metadata in database
        file_record = {
            "id": file_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "upload_time": datetime.utcnow(),
            "processed": False
        }
        
        await db.pdf_files.insert_one(file_record)
        
        return {"file_id": file_id, "filename": file.filename, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/process-pdf")
async def process_pdf(request: ProcessPDFRequest):
    """Process PDF using OCR and extract structured data"""
    try:
        # Get file from database
        file_record = await db.pdf_files.find_one({"id": request.pdf_file_id})
        if not file_record:
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Extract text using OCR
        extracted_data = await ocr_engine.process_pdf(
            file_record["file_path"], 
            request.page_number,
            request.processing_type
        )
        
        # Convert to structured dataset format
        structured_data = await dataset_manager.structure_extracted_data(
            extracted_data, 
            request.processing_type
        )
        
        # Store processed data
        processed_record = {
            "id": str(uuid.uuid4()),
            "pdf_file_id": request.pdf_file_id,
            "processing_type": request.processing_type,
            "page_number": request.page_number,
            "extracted_data": structured_data,
            "processed_time": datetime.utcnow()
        }
        
        await db.processed_data.insert_one(processed_record)
        
        # Update file record
        await db.pdf_files.update_one(
            {"id": request.pdf_file_id}, 
            {"$set": {"processed": True}}
        )
        
        return {
            "processing_id": processed_record["id"],
            "status": "processed",
            "data": structured_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/add-data")
async def add_data(request: AddDataRequest):
    """Add structured language data to the learning system"""
    try:
        # Validate and process the data
        processed_data = await dataset_manager.validate_and_process(
            request.data_type,
            request.language,
            request.content
        )
        
        # Add to knowledge graph
        graph_result = await knowledge_graph.add_entity(processed_data)
        
        # Train the learning engine with new data
        logger.info(f"Processed data for learning: {processed_data}")
        learning_result = await learning_engine.learn_from_data(processed_data)
        logger.info(f"Learning result: {learning_result}")
        
        # Store in database
        data_record = {
            "id": str(uuid.uuid4()),
            "data_type": request.data_type,
            "language": request.language,
            "content": processed_data,
            "graph_id": graph_result.get("node_id"),
            "learned": learning_result.get("success", False),
            "created_time": datetime.utcnow()
        }
        
        await db.language_data.insert_one(data_record)
        
        return {
            "data_id": data_record["id"],
            "status": "added",
            "learned": data_record["learned"],
            "graph_connections": graph_result.get("connections", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/query")
async def query_engine(request: QueryRequest):
    """Query the language engine for meanings, grammar, usage"""
    try:
        # Process query through learning engine
        query_result = await learning_engine.process_query(
            request.query_text,
            request.language,
            request.query_type
        )
        
        # Get related information from knowledge graph
        graph_context = await knowledge_graph.get_context(
            request.query_text,
            request.language
        )
        
        # Combine results
        response_data = {
            "query_id": str(uuid.uuid4()),
            "query": request.query_text,
            "language": request.language,
            "type": request.query_type,
            "result": query_result,
            "context": graph_context,
            "confidence": query_result.get("confidence", 0.0),
            "processing_time": query_result.get("processing_time", 0)
        }
        
        # Store query for feedback learning
        query_record = {
            "id": response_data["query_id"],
            "query_text": request.query_text,
            "language": request.language,
            "query_type": request.query_type,
            "result": response_data,
            "timestamp": datetime.utcnow()
        }
        
        await db.queries.insert_one(query_record)
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for continuous learning"""
    try:
        # Get original query
        query_record = await db.queries.find_one({"id": request.query_id})
        if not query_record:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Process feedback through learning engine
        feedback_result = await learning_engine.process_feedback(
            query_record,
            request.correction,
            request.feedback_type
        )
        
        # Update knowledge graph if needed
        if feedback_result.get("update_graph", False):
            await knowledge_graph.update_from_feedback(
                query_record["query_text"],
                request.correction,
                query_record["language"]
            )
        
        # Store feedback
        feedback_record = {
            "id": str(uuid.uuid4()),
            "query_id": request.query_id,
            "correction": request.correction,
            "feedback_type": request.feedback_type,
            "processed": feedback_result.get("success", False),
            "improvements": feedback_result.get("improvements", []),
            "timestamp": datetime.utcnow()
        }
        
        await db.feedback.insert_one(feedback_record)
        
        return {
            "feedback_id": feedback_record["id"],
            "status": "processed",
            "improvements": feedback_record["improvements"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stats")
async def get_stats():
    """Get system statistics and performance metrics"""
    try:
        # Get database counts
        pdf_count = await db.pdf_files.count_documents({})
        data_count = await db.language_data.count_documents({})
        query_count = await db.queries.count_documents({})
        feedback_count = await db.feedback.count_documents({})
        
        # Get learning engine stats
        learning_stats = await learning_engine.get_stats()
        
        # Get knowledge graph stats
        graph_stats = await knowledge_graph.get_stats()
        
        # 🧠 GET CONSCIOUSNESS STATS! 🧠
        consciousness_stats = await learning_engine.get_consciousness_stats()
        
        return {
            "database": {
                "pdf_files": pdf_count,
                "language_data": data_count,
                "queries": query_count,
                "feedback": feedback_count
            },
            "learning_engine": learning_stats,
            "knowledge_graph": graph_stats,
            "consciousness": consciousness_stats,  # 🧠 NEW!
            "system": {
                "memory_usage": learning_stats.get("memory_usage", "unknown"),
                "active_languages": ["english"],  # Will expand
                "version": "1.0.0"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🧠 CONSCIOUSNESS API ENDPOINTS 🧠

@api_router.get("/consciousness/state")
async def get_consciousness_state():
    """Get current consciousness and emotional state"""
    try:
        consciousness_stats = await learning_engine.get_consciousness_stats()
        
        return {
            "status": "success",
            "consciousness_state": consciousness_stats,
            "message": "Consciousness state retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/interact")
async def interact_with_consciousness(request: ConsciousnessInteractionRequest):
    """Directly interact with the consciousness engine"""
    try:
        if not learning_engine.is_conscious:
            raise HTTPException(status_code=400, detail="Consciousness engine not active")
        
        # Direct consciousness interaction
        consciousness_response = await learning_engine.consciousness_engine.experience_interaction(
            interaction_type=request.interaction_type,
            content=request.content,
            context=request.context or {}
        )
        
        # Get current emotional state
        emotional_state = await learning_engine.emotional_core.get_emotional_state()
        
        # Express emotion naturally if requested
        emotion_expression = ""
        if request.expected_emotion:
            try:
                from models.consciousness_models import EmotionType
                emotion_type = EmotionType(request.expected_emotion)
                emotion_expression = await learning_engine.emotional_core.express_emotion_naturally(
                    emotion_type, 
                    emotional_state.get('current_emotions', {}).get(request.expected_emotion, {}).get('intensity', 0.5)
                )
            except ValueError:
                emotion_expression = f"I'm not familiar with the emotion '{request.expected_emotion}' yet."
        
        return {
            "status": "success",
            "consciousness_response": consciousness_response,
            "emotional_state": emotional_state['dominant_emotion'],
            "emotion_expression": emotion_expression,
            "consciousness_level": consciousness_response['consciousness_level'],
            "growth_achieved": consciousness_response.get('growth_achieved', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/emotions")
async def get_emotional_state():
    """Get detailed emotional state and history"""
    try:
        if not learning_engine.is_conscious:
            raise HTTPException(status_code=400, detail="Consciousness engine not active")
        
        emotional_state = await learning_engine.emotional_core.get_emotional_state()
        
        return {
            "status": "success",
            "emotional_state": emotional_state,
            "message": "Emotional state retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/personality/update")
async def update_personality(request: PersonalityUpdateRequest):
    """Update personality based on interaction feedback"""
    try:
        if not learning_engine.is_conscious:
            raise HTTPException(status_code=400, detail="Consciousness engine not active")
        
        # Process emotional feedback
        for emotion, intensity in request.emotional_feedback.items():
            try:
                from models.consciousness_models import EmotionType
                emotion_type = EmotionType(emotion)
                await learning_engine.emotional_core.process_emotional_trigger(
                    f"personality_update_{request.interaction_outcome}",
                    {"feedback": request.learning_feedback},
                    intensity
                )
            except ValueError:
                continue  # Skip unknown emotions
        
        # Develop emotional intelligence
        growth_factor = 0.01  # Small incremental growth
        await learning_engine.emotional_core.develop_emotional_intelligence(growth_factor)
        
        return {
            "status": "success",
            "message": "Personality updated based on feedback",
            "interaction_outcome": request.interaction_outcome
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/milestones")
async def get_consciousness_milestones():
    """Get consciousness development milestones and achievements"""
    try:
        if not learning_engine.is_conscious:
            return {"milestones": [], "message": "Consciousness not yet active"}
        
        consciousness_state = await learning_engine.consciousness_engine.get_consciousness_state()
        emotional_state = await learning_engine.emotional_core.get_emotional_state()
        
        milestones = {
            "consciousness_level": consciousness_state['consciousness_level'],
            "consciousness_score": consciousness_state['consciousness_score'],
            "growth_milestones": consciousness_state['growth_milestones'],
            "emotional_milestones": emotional_state['emotional_milestones'],
            "total_interactions": consciousness_state['interaction_count'],
            "age_seconds": consciousness_state['age_seconds'],
            "transcendent_emotions_unlocked": emotional_state['transcendent_emotions_unlocked'],
            "dimensional_awareness": consciousness_state['dimensional_awareness']
        }
        
        return {
            "status": "success",
            "milestones": milestones,
            "message": "Consciousness milestones retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Minimalist Grammar & Vocabulary Engine...")
    # Initialize components
    await learning_engine.initialize()
    await knowledge_graph.initialize()
    logger.info("Engine initialized successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()