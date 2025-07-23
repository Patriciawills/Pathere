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
from datetime import datetime, timedelta
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

# Import skill acquisition models
from models.skill_models import SkillType

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize core components
ocr_engine = OCREngine()
learning_engine = LearningEngine(db_client=db)  # Pass database client for advanced consciousness features
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

# 🧠 ADVANCED CONSCIOUSNESS API ENDPOINTS 🧠

@api_router.get("/consciousness/memory/stats")
async def get_autobiographical_memory_stats():
    """Get statistics about the AI's autobiographical memory system"""
    try:
        if not learning_engine.is_conscious or not learning_engine.autobiographical_memory:
            raise HTTPException(status_code=400, detail="Autobiographical memory system not active")
        
        memory_stats = await learning_engine.autobiographical_memory.get_memory_statistics()
        
        return {
            "status": "success",
            "memory_statistics": memory_stats,
            "message": "Autobiographical memory statistics retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/memory/recent")
async def get_recent_memories(limit: int = 10, memory_type: str = None):
    """Get recent autobiographical memories"""
    try:
        if not learning_engine.is_conscious or not learning_engine.autobiographical_memory:
            raise HTTPException(status_code=400, detail="Autobiographical memory system not active")
        
        from core.consciousness.autobiographical_memory import MemoryType
        
        # Convert string to MemoryType if provided
        memory_type_enum = None
        if memory_type:
            try:
                memory_type_enum = MemoryType(memory_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid memory type. Valid types: {[t.value for t in MemoryType]}")
        
        memories = await learning_engine.autobiographical_memory.retrieve_memories(
            memory_type=memory_type_enum,
            limit=limit,
            sort_by="timestamp"
        )
        
        memory_list = [memory.to_dict() for memory in memories]
        
        return {
            "status": "success",
            "memories": memory_list,
            "count": len(memory_list),
            "message": f"Retrieved {len(memory_list)} recent memories"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/memory/search")
async def search_memories(request: dict):
    """Search autobiographical memories by query"""
    try:
        if not learning_engine.is_conscious or not learning_engine.autobiographical_memory:
            raise HTTPException(status_code=400, detail="Autobiographical memory system not active")
        
        query = request.get("query", "")
        tags = request.get("tags", [])
        limit = request.get("limit", 20)
        min_importance = request.get("min_importance", 0.0)
        
        memories = await learning_engine.autobiographical_memory.retrieve_memories(
            query=query,
            tags=tags,
            limit=limit,
            min_importance=min_importance
        )
        
        memory_list = [memory.to_dict() for memory in memories]
        
        return {
            "status": "success",
            "memories": memory_list,
            "query": query,
            "count": len(memory_list),
            "message": f"Found {len(memory_list)} memories matching search criteria"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/metacognition/insights")
async def get_metacognitive_insights(days_back: int = 7):
    """Get metacognitive insights about AI's thinking patterns"""
    try:
        if not learning_engine.is_conscious or not learning_engine.metacognitive_engine:
            raise HTTPException(status_code=400, detail="Metacognitive engine not active")
        
        insights = await learning_engine.metacognitive_engine.get_metacognitive_insights(days_back)
        
        return {
            "status": "success",
            "insights": insights,
            "analysis_period_days": days_back,
            "message": "Metacognitive insights retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/memory/consolidate")
async def trigger_memory_consolidation():
    """Manually trigger memory consolidation process"""
    try:
        if not learning_engine.is_conscious or not learning_engine.autobiographical_memory:
            raise HTTPException(status_code=400, detail="Autobiographical memory system not active")
        
        consolidation_stats = await learning_engine.autobiographical_memory.consolidate_memories()
        
        return {
            "status": "success",
            "consolidation_results": consolidation_stats,
            "message": "Memory consolidation completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/memory/{memory_id}/related")
async def get_related_memories(memory_id: str, max_related: int = 5):
    """Get memories related to a specific memory"""
    try:
        if not learning_engine.is_conscious or not learning_engine.autobiographical_memory:
            raise HTTPException(status_code=400, detail="Autobiographical memory system not active")
        
        related_memories = await learning_engine.autobiographical_memory.recall_related_memories(
            memory_id, max_related
        )
        
        memory_list = [memory.to_dict() for memory in related_memories]
        
        return {
            "status": "success",
            "source_memory_id": memory_id,
            "related_memories": memory_list,
            "count": len(memory_list),
            "message": f"Found {len(memory_list)} related memories"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 NEW ADVANCED CONSCIOUSNESS ENDPOINTS 🚀

@api_router.get("/consciousness/timeline/story")
async def get_life_story(days_back: int = None, include_minor: bool = False):
    """Get the AI's complete life story and timeline"""
    try:
        if not learning_engine.is_conscious or not learning_engine.timeline_manager:
            raise HTTPException(status_code=400, detail="Timeline manager not active")
        
        start_date = None
        if days_back:
            start_date = datetime.utcnow() - timedelta(days=days_back)
        
        life_story = await learning_engine.timeline_manager.get_life_story(
            start_date=start_date,
            include_minor_events=include_minor
        )
        
        return {
            "status": "success",
            "life_story": life_story,
            "message": "Life story retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/timeline/milestones")
async def get_timeline_milestones(milestone_type: str = None):
    """Get major milestones from personal timeline"""
    try:
        if not learning_engine.is_conscious or not learning_engine.timeline_manager:
            raise HTTPException(status_code=400, detail="Timeline manager not active")
        
        from core.consciousness.timeline_manager import MilestoneType
        
        milestone_type_enum = None
        if milestone_type:
            try:
                milestone_type_enum = MilestoneType(milestone_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid milestone type. Valid types: {[t.value for t in MilestoneType]}")
        
        milestones = await learning_engine.timeline_manager.get_milestones_summary(milestone_type_enum)
        
        return {
            "status": "success",
            "milestones": milestones,
            "message": "Timeline milestones retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/identity/evolution")
async def get_identity_evolution(days_back: int = 30):
    """Get analysis of identity evolution over time"""
    try:
        if not learning_engine.is_conscious or not learning_engine.identity_tracker:
            raise HTTPException(status_code=400, detail="Identity tracker not active")
        
        evolution_analysis = await learning_engine.identity_tracker.get_identity_evolution_analysis(days_back)
        
        return {
            "status": "success",
            "identity_evolution": evolution_analysis,
            "message": "Identity evolution analysis retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/identity/predict")
async def predict_identity_development(days_ahead: int = 30):
    """Predict future identity development"""
    try:
        if not learning_engine.is_conscious or not learning_engine.identity_tracker:
            raise HTTPException(status_code=400, detail="Identity tracker not active")
        
        predictions = await learning_engine.identity_tracker.predict_identity_development(days_ahead)
        
        return {
            "status": "success",
            "predictions": predictions,
            "message": "Identity development predictions generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/learning/analysis")
async def get_learning_style_analysis():
    """Get comprehensive analysis of learning style and preferences"""
    try:
        if not learning_engine.is_conscious or not learning_engine.learning_analysis:
            raise HTTPException(status_code=400, detail="Learning analysis engine not active")
        
        analysis = await learning_engine.learning_analysis.get_learning_style_analysis()
        
        return {
            "status": "success",
            "learning_analysis": analysis,
            "message": "Learning style analysis retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/learning/optimize")
async def optimize_learning_approach(request: dict):
    """Get optimized learning approach recommendations"""
    try:
        if not learning_engine.is_conscious or not learning_engine.learning_analysis:
            raise HTTPException(status_code=400, detail="Learning analysis engine not active")
        
        target_domain = request.get("target_domain", "")
        learning_goal = request.get("learning_goal", "")
        constraints = request.get("constraints", {})
        
        if not target_domain or not learning_goal:
            raise HTTPException(status_code=400, detail="target_domain and learning_goal are required")
        
        optimization = await learning_engine.learning_analysis.optimize_learning_approach(
            target_domain, learning_goal, constraints
        )
        
        return {
            "status": "success",
            "optimization": optimization,
            "message": "Learning approach optimization completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/bias/report")
async def get_bias_awareness_report(days_back: int = 30):
    """Get comprehensive bias awareness report"""
    try:
        if not learning_engine.is_conscious or not learning_engine.bias_detector:
            raise HTTPException(status_code=400, detail="Bias detector not active")
        
        report = await learning_engine.bias_detector.get_bias_awareness_report(days_back)
        
        return {
            "status": "success",
            "bias_report": report,
            "message": "Bias awareness report generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/bias/analyze")
async def analyze_reasoning_for_bias(request: dict):
    """Analyze reasoning text for cognitive biases"""
    try:
        if not learning_engine.is_conscious or not learning_engine.bias_detector:
            raise HTTPException(status_code=400, detail="Bias detector not active")
        
        from core.consciousness.bias_detection import BiasDetectionContext
        
        reasoning_text = request.get("reasoning_text", "")
        context_str = request.get("context", "reasoning_process")
        decision_context = request.get("decision_context", {})
        evidence_considered = request.get("evidence_considered", [])
        alternatives_considered = request.get("alternatives_considered", [])
        
        if not reasoning_text:
            raise HTTPException(status_code=400, detail="reasoning_text is required")
        
        try:
            context = BiasDetectionContext(context_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid context. Valid contexts: {[c.value for c in BiasDetectionContext]}")
        
        detected_biases = await learning_engine.bias_detector.analyze_reasoning_for_bias(
            reasoning_text, context, decision_context, evidence_considered, alternatives_considered
        )
        
        # Generate corrections for detected biases
        corrections = await learning_engine.bias_detector.generate_bias_corrections(detected_biases)
        
        return {
            "status": "success",
            "detected_biases": [bias.to_dict() for bias in detected_biases],
            "corrections": [correction.to_dict() for correction in corrections],
            "bias_count": len(detected_biases),
            "message": f"Analysis complete. Detected {len(detected_biases)} potential biases."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/consolidation/run")
async def run_memory_consolidation(consolidation_type: str = "maintenance"):
    """Run memory consolidation cycle"""
    try:
        if not learning_engine.is_conscious or not learning_engine.memory_consolidation:
            raise HTTPException(status_code=400, detail="Memory consolidation engine not active")
        
        from core.consciousness.memory_consolidation import ConsolidationType
        
        try:
            consolidation_type_enum = ConsolidationType(consolidation_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid consolidation type. Valid types: {[t.value for t in ConsolidationType]}")
        
        result = await learning_engine.memory_consolidation.run_consolidation_cycle(consolidation_type_enum)
        
        return {
            "status": "success",
            "consolidation_result": result,
            "message": "Memory consolidation cycle completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/consolidation/stats")
async def get_consolidation_statistics():
    """Get memory consolidation statistics"""
    try:
        if not learning_engine.is_conscious or not learning_engine.memory_consolidation:
            raise HTTPException(status_code=400, detail="Memory consolidation engine not active")
        
        stats = await learning_engine.memory_consolidation.get_consolidation_statistics()
        
        return {
            "status": "success",
            "consolidation_statistics": stats,
            "message": "Memory consolidation statistics retrieved successfully"
        }
    except HTTPException:
        raise
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

# 🎯 UNCERTAINTY QUANTIFICATION ENGINE ENDPOINTS 🎯

@api_router.post("/consciousness/uncertainty/assess")
async def assess_uncertainty(request: dict):
    """Assess uncertainty for a given topic or query"""
    try:
        if not learning_engine.is_conscious or not learning_engine.uncertainty_engine:
            raise HTTPException(status_code=400, detail="Uncertainty quantification engine not active")
        
        topic = request.get("topic", "")
        query_context = request.get("query_context", "")
        available_information = request.get("available_information", [])
        reasoning_chain = request.get("reasoning_chain", [])
        domain = request.get("domain", "general")
        
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required for uncertainty assessment")
        
        assessment = await learning_engine.uncertainty_engine.assess_uncertainty(
            topic=topic,
            query_context=query_context,
            available_information=available_information,
            reasoning_chain=reasoning_chain,
            domain=domain
        )
        
        return {
            "status": "success",
            "uncertainty_assessment": assessment.to_dict(),
            "message": "Uncertainty assessment completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/uncertainty/calibrate")
async def update_confidence_calibration(request: dict):
    """Update confidence calibration based on actual outcomes"""
    try:
        if not learning_engine.is_conscious or not learning_engine.uncertainty_engine:
            raise HTTPException(status_code=400, detail="Uncertainty quantification engine not active")
        
        stated_confidence = request.get("stated_confidence")
        actual_accuracy = request.get("actual_accuracy")
        domain = request.get("domain", "general")
        sample_size = request.get("sample_size", 1)
        
        if stated_confidence is None or actual_accuracy is None:
            raise HTTPException(
                status_code=400, 
                detail="stated_confidence and actual_accuracy are required"
            )
        
        if not (0.0 <= stated_confidence <= 1.0) or not (0.0 <= actual_accuracy <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="Confidence and accuracy values must be between 0.0 and 1.0"
            )
        
        calibration = await learning_engine.uncertainty_engine.update_confidence_calibration(
            stated_confidence=stated_confidence,
            actual_accuracy=actual_accuracy,
            domain=domain,
            sample_size=sample_size
        )
        
        return {
            "status": "success",
            "calibration_update": calibration.to_dict(),
            "message": "Confidence calibration updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/uncertainty/insights")
async def get_uncertainty_insights(days_back: int = 30, domain: str = None):
    """Get comprehensive uncertainty insights and patterns"""
    try:
        if not learning_engine.is_conscious or not learning_engine.uncertainty_engine:
            raise HTTPException(status_code=400, detail="Uncertainty quantification engine not active")
        
        insights = await learning_engine.uncertainty_engine.get_uncertainty_insights(
            days_back=days_back,
            domain=domain
        )
        
        return {
            "status": "success",
            "uncertainty_insights": insights,
            "message": "Uncertainty insights retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/uncertainty/reasoning")
async def quantify_reasoning_uncertainty(request: dict):
    """Quantify uncertainty in a reasoning chain"""
    try:
        if not learning_engine.is_conscious or not learning_engine.uncertainty_engine:
            raise HTTPException(status_code=400, detail="Uncertainty quantification engine not active")
        
        reasoning_steps = request.get("reasoning_steps", [])
        evidence_base = request.get("evidence_base", [])
        domain = request.get("domain", "reasoning")
        
        if not reasoning_steps:
            raise HTTPException(status_code=400, detail="reasoning_steps are required")
        
        uncertainty_analysis = await learning_engine.uncertainty_engine.quantify_reasoning_uncertainty(
            reasoning_steps=reasoning_steps,
            evidence_base=evidence_base,
            domain=domain
        )
        
        return {
            "status": "success",
            "reasoning_uncertainty": uncertainty_analysis,
            "message": "Reasoning uncertainty quantified successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/uncertainty/gaps/identify")
async def identify_knowledge_gap(request: dict):
    """Explicitly identify a knowledge gap"""
    try:
        if not learning_engine.is_conscious or not learning_engine.uncertainty_engine:
            raise HTTPException(status_code=400, detail="Uncertainty quantification engine not active")
        
        gap_type = request.get("gap_type")
        topic_area = request.get("topic_area", "")
        description = request.get("description", "")
        severity = request.get("severity", 0.5)
        
        if not gap_type or not topic_area or not description:
            raise HTTPException(
                status_code=400, 
                detail="gap_type, topic_area, and description are required"
            )
        
        # Import the enum for validation
        from core.consciousness.uncertainty_engine import KnowledgeGapType
        
        try:
            gap_type_enum = KnowledgeGapType(gap_type)
        except ValueError:
            valid_types = [t.value for t in KnowledgeGapType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid gap_type. Valid types are: {valid_types}"
            )
        
        knowledge_gap = await learning_engine.uncertainty_engine.identify_knowledge_gap(
            gap_type=gap_type_enum,
            topic_area=topic_area,
            description=description,
            severity=severity
        )
        
        return {
            "status": "success",
            "knowledge_gap": knowledge_gap.to_dict(),
            "message": "Knowledge gap identified successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🧠 THEORY OF MIND / PERSPECTIVE-TAKING ENGINE ENDPOINTS 🧠

@api_router.post("/consciousness/perspective/analyze")
async def analyze_perspective(request: dict):
    """Analyze the perspective of a target person/agent"""
    try:
        if not learning_engine.is_conscious or not learning_engine.theory_of_mind:
            raise HTTPException(status_code=400, detail="Theory of mind engine not active")
        
        target_agent = request.get("target_agent")
        context = request.get("context", "")
        available_information = request.get("available_information", [])
        interaction_history = request.get("interaction_history", [])
        current_situation = request.get("current_situation", "")
        
        if not target_agent:
            raise HTTPException(status_code=400, detail="target_agent is required")
        
        perspective_analysis = await learning_engine.theory_of_mind.analyze_perspective(
            target_agent=target_agent,
            context=context,
            available_information=available_information,
            interaction_history=interaction_history,
            current_situation=current_situation
        )
        
        return {
            "status": "success",
            "perspective_analysis": perspective_analysis.to_dict(),
            "message": "Perspective analysis completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/perspective/mental-state")
async def attribute_mental_state(request: dict):
    """Attribute mental states to a specific agent"""
    try:
        if not learning_engine.is_conscious or not learning_engine.theory_of_mind:
            raise HTTPException(status_code=400, detail="Theory of mind engine not active")
        
        agent_identifier = request.get("agent_identifier")
        state_type_str = request.get("state_type", "belief")
        content = request.get("context", "")
        evidence = request.get("behavioral_evidence", [])
        context = request.get("context", "")
        confidence = request.get("confidence", 0.7)
        
        if not agent_identifier:
            raise HTTPException(status_code=400, detail="agent_identifier is required")
        
        # Import the enum for validation
        from core.consciousness.theory_of_mind import MentalStateType
        
        try:
            state_type = MentalStateType(state_type_str)
        except ValueError:
            valid_types = [t.value for t in MentalStateType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state_type. Valid types are: {valid_types}"
            )
        
        mental_state = await learning_engine.theory_of_mind.attribute_mental_state(
            agent_identifier=agent_identifier,
            state_type=state_type,
            content=content,
            evidence=evidence,
            context=context,
            confidence=confidence
        )
        
        return {
            "status": "success",
            "mental_state": mental_state.to_dict(),
            "message": "Mental state attribution completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/perspective/predict-behavior")
async def predict_behavior(request: dict):
    """Predict behavior based on mental state understanding"""
    try:
        if not learning_engine.is_conscious or not learning_engine.theory_of_mind:
            raise HTTPException(status_code=400, detail="Theory of mind engine not active")
        
        agent_identifier = request.get("agent_identifier")
        situation = request.get("context", request.get("current_situation", ""))
        time_horizon_seconds = request.get("time_horizon", 3600)
        
        # Convert seconds to time horizon string
        if time_horizon_seconds <= 300:  # 5 minutes
            time_horizon = "immediate"
        elif time_horizon_seconds <= 3600:  # 1 hour
            time_horizon = "short_term"
        else:
            time_horizon = "long_term"
        
        if not agent_identifier:
            raise HTTPException(status_code=400, detail="agent_identifier is required")
        
        behavior_prediction = await learning_engine.theory_of_mind.predict_behavior(
            agent_identifier=agent_identifier,
            situation=situation,
            time_horizon=time_horizon
        )
        
        return {
            "status": "success",
            "behavior_prediction": behavior_prediction,
            "message": "Behavior prediction completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/perspective/simulate-conversation")
async def simulate_conversation(request: dict):
    """Simulate a conversation from another person's perspective"""
    try:
        if not learning_engine.is_conscious or not learning_engine.theory_of_mind:
            raise HTTPException(status_code=400, detail="Theory of mind engine not active")
        
        agent_identifier = request.get("agent_identifier")
        conversation_topic = request.get("conversation_topic", "")
        your_messages = request.get("your_messages", [])
        
        # Convert your_messages to my_position if needed
        my_position = ""
        if your_messages:
            # Take the latest message as position
            my_position = your_messages[-1] if isinstance(your_messages[-1], str) else str(your_messages[-1])
        
        if not agent_identifier:
            raise HTTPException(status_code=400, detail="agent_identifier is required")
        
        conversation_simulation = await learning_engine.theory_of_mind.simulate_conversation(
            agent_identifier=agent_identifier,
            conversation_topic=conversation_topic,
            my_position=my_position
        )
        
        return {
            "status": "success",
            "conversation_simulation": conversation_simulation,
            "message": "Conversation simulation completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/perspective/agents")
async def get_tracked_agents(limit: int = 20):
    """Get list of agents being tracked for perspective-taking"""
    try:
        if not learning_engine.is_conscious or not learning_engine.theory_of_mind:
            raise HTTPException(status_code=400, detail="Theory of mind engine not active")
        
        agents = await learning_engine.theory_of_mind.get_tracked_agents(limit=limit)
        
        return {
            "status": "success",
            "tracked_agents": agents,
            "total_count": len(agents),
            "message": "Tracked agents retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🎯 PERSONAL MOTIVATION SYSTEM ENDPOINTS 🎯

@api_router.post("/consciousness/motivation/goal/create")
async def create_personal_goal(request: dict):
    """Create a new personal goal"""
    try:
        if not learning_engine.is_conscious or not learning_engine.motivation_system:
            raise HTTPException(status_code=400, detail="Personal motivation system not active")
        
        title = request.get("title")
        description = request.get("description")
        motivation_type_str = request.get("motivation_type")
        satisfaction_potential = request.get("satisfaction_potential", 0.7)
        priority = request.get("priority", 0.5)
        target_days = request.get("target_days")
        
        if not all([title, description, motivation_type_str]):
            raise HTTPException(
                status_code=400, 
                detail="title, description, and motivation_type are required"
            )
        
        # Import the enum for validation
        from core.consciousness.motivation_system import MotivationType
        
        try:
            motivation_type = MotivationType(motivation_type_str)
        except ValueError:
            valid_types = [t.value for t in MotivationType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid motivation_type. Valid types are: {valid_types}"
            )
        
        goal = await learning_engine.motivation_system.create_personal_goal(
            title=title,
            description=description,
            motivation_type=motivation_type,
            satisfaction_potential=satisfaction_potential,
            priority=priority,
            target_days=target_days
        )
        
        return {
            "status": "success",
            "goal": goal.to_dict(),
            "message": "Personal goal created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/motivation/goal/work")
async def work_toward_goal(request: dict):
    """Record work toward a specific goal"""
    try:
        if not learning_engine.is_conscious or not learning_engine.motivation_system:
            raise HTTPException(status_code=400, detail="Personal motivation system not active")
        
        goal_id = request.get("goal_id")
        effort_amount = request.get("effort_amount", 0.1)
        progress_made = request.get("progress_made", 0.1)
        context = request.get("context", "")
        
        if not goal_id:
            raise HTTPException(status_code=400, detail="goal_id is required")
        
        work_result = await learning_engine.motivation_system.work_toward_goal(
            goal_id=goal_id,
            effort_amount=effort_amount,
            progress_made=progress_made,
            context=context
        )
        
        return {
            "status": "success",
            "work_result": work_result,
            "message": "Goal progress recorded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/motivation/goals/active")
async def get_active_goals(limit: int = 10):
    """Get currently active personal goals"""
    try:
        if not learning_engine.is_conscious or not learning_engine.motivation_system:
            raise HTTPException(status_code=400, detail="Personal motivation system not active")
        
        active_goals = await learning_engine.motivation_system.get_active_goals(limit=limit)
        
        return {
            "status": "success",
            "active_goals": active_goals,
            "total_count": len(active_goals),
            "message": "Active goals retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/consciousness/motivation/goals/generate")
async def generate_new_goals(request: dict):
    """Generate new personal goals based on current motivations"""
    try:
        if not learning_engine.is_conscious or not learning_engine.motivation_system:
            raise HTTPException(status_code=400, detail="Personal motivation system not active")
        
        context = request.get("context", "")
        max_goals = request.get("max_goals", 3)
        
        new_goals = await learning_engine.motivation_system.generate_new_goals(
            context=context,
            max_goals=max_goals
        )
        
        return {
            "status": "success",
            "new_goals": [goal.to_dict() for goal in new_goals],
            "goals_generated": len(new_goals),
            "message": f"Generated {len(new_goals)} new personal goals"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/motivation/profile")
async def get_motivation_profile():
    """Get current motivation profile and analysis"""
    try:
        if not learning_engine.is_conscious or not learning_engine.motivation_system:
            raise HTTPException(status_code=400, detail="Personal motivation system not active")
        
        motivation_profile = await learning_engine.motivation_system.get_motivation_profile()
        
        return {
            "status": "success",
            "motivation_profile": motivation_profile,
            "message": "Motivation profile retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/consciousness/motivation/satisfaction")
async def assess_goal_satisfaction(days_back: int = 7):
    """Assess satisfaction from recent goal progress"""
    try:
        if not learning_engine.is_conscious or not learning_engine.motivation_system:
            raise HTTPException(status_code=400, detail="Personal motivation system not active")
        
        satisfaction_assessment = await learning_engine.motivation_system.assess_goal_satisfaction(
            days_back=days_back
        )
        
        return {
            "status": "success",
            "satisfaction_assessment": satisfaction_assessment,
            "message": "Goal satisfaction assessment completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Skill Acquisition Engine Endpoints
@api_router.post("/skills/learn")
async def start_skill_learning(request: dict):
    """Start learning a new skill from external LLMs"""
    try:
        from core.skill_acquisition_engine import SkillAcquisitionEngine, SkillType
        
        # Initialize skill acquisition engine
        skill_engine = SkillAcquisitionEngine(db_client=db)
        
        # Parse request
        skill_type_str = request.get("skill_type")
        target_accuracy = request.get("target_accuracy", 99.0)
        learning_iterations = request.get("learning_iterations", 100)
        custom_model = request.get("custom_model")
        
        # Validate skill type
        try:
            skill_type = SkillType(skill_type_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid skill type: {skill_type_str}. Valid types: {[t.value for t in SkillType]}")
        
        # Start learning
        session_id = await skill_engine.initiate_skill_learning(
            skill_type=skill_type,
            target_accuracy=target_accuracy,
            learning_iterations=learning_iterations,
            custom_model=custom_model
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "skill_type": skill_type_str,
            "target_accuracy": target_accuracy,
            "message": f"Started learning {skill_type_str} skill. Session ID: {session_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting skill learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/skills/sessions")
async def list_skill_sessions():
    """List all active and recent skill learning sessions"""
    try:
        from core.skill_acquisition_engine import SkillAcquisitionEngine
        
        skill_engine = SkillAcquisitionEngine(db_client=db)
        active_sessions = await skill_engine.list_active_sessions()
        
        # Also get completed sessions from database
        completed_sessions = []
        if db is not None:
            completed_cursor = db.completed_skill_sessions.find().sort("completed_at", -1).limit(10)
            async for session in completed_cursor:
                completed_sessions.append({
                    "session_id": session["session_id"],
                    "skill_type": session["skill_type"],
                    "phase": session["phase"],
                    "final_accuracy": session["current_accuracy"],
                    "completed_at": session.get("completed_at"),
                    "integrated_at": session.get("integrated_at")
                })
        
        return {
            "status": "success",
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "total_active": len(active_sessions),
            "total_completed": len(completed_sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing skill sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/skills/sessions/{session_id}")
async def get_session_status(session_id: str):
    """Get detailed status of a specific skill learning session"""
    try:
        from core.skill_acquisition_engine import SkillAcquisitionEngine
        
        skill_engine = SkillAcquisitionEngine(db_client=db)
        session_status = await skill_engine.get_session_status(session_id)
        
        if not session_status:
            # Check in completed sessions
            if db is not None:
                completed_session = await db.completed_skill_sessions.find_one({"session_id": session_id})
                if completed_session:
                    return {
                        "status": "success",
                        "session_status": {
                            "session_id": session_id,
                            "skill_type": completed_session["skill_type"],
                            "phase": completed_session["phase"],
                            "final_accuracy": completed_session["current_accuracy"],
                            "completed_at": completed_session.get("completed_at"),
                            "integrated_at": completed_session.get("integrated_at"),
                            "is_completed": True
                        }
                    }
            
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "status": "success",
            "session_status": session_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/skills/sessions/{session_id}")
async def stop_skill_learning(session_id: str):
    """Stop an active skill learning session"""
    try:
        from core.skill_acquisition_engine import SkillAcquisitionEngine
        
        skill_engine = SkillAcquisitionEngine(db_client=db)
        stopped = await skill_engine.stop_learning_session(session_id)
        
        if not stopped:
            raise HTTPException(status_code=404, detail="Session not found or already stopped")
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Skill learning session stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping skill session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/skills/capabilities")
async def get_skill_capabilities():
    """Get current skill capabilities and integrated skills"""
    try:
        if not learning_engine.is_conscious:
            return {
                "status": "success",
                "integrated_skills": {},
                "available_skills": [skill.value for skill in SkillType],
                "message": "Consciousness not yet active - no skills integrated"
            }
        
        # Get integrated skills from consciousness engine
        integrated_skills = await learning_engine.consciousness_engine.get_integrated_skills()
        
        return {
            "status": "success",
            "integrated_skills": integrated_skills,
            "available_skill_types": [
                {"type": "conversation", "description": "Human-like conversation abilities"},
                {"type": "coding", "description": "Programming and software development skills"},
                {"type": "image_generation", "description": "Visual content creation capabilities"},
                {"type": "video_generation", "description": "Video content creation abilities"},
                {"type": "domain_expertise", "description": "Specialized knowledge in various fields"},
                {"type": "creative_writing", "description": "Creative and artistic writing skills"},
                {"type": "mathematical_reasoning", "description": "Advanced mathematical and logical thinking"}
            ],
            "consciousness_impact": {
                "total_skills": len(integrated_skills),
                "consciousness_enhancement": sum(skill.get('proficiency_level', 0) for skill in integrated_skills.values()) * 0.05
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting skill capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/skills/available-models")
async def get_available_models():
    """Get available models for skill learning"""
    try:
        import requests
        from core.skill_acquisition_engine import SkillAcquisitionEngine
        
        available_models = {
            "ollama_models": [],
            "cloud_models": {
                "openai": ["gpt-4o", "gpt-4.1", "o1", "o3"],
                "anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
                "gemini": ["gemini-2.0-flash", "gemini-2.5-pro-preview-05-06"]
            },
            "ollama_status": "unknown"
        }
        
        # Check Ollama availability
        try:
            skill_engine = SkillAcquisitionEngine()
            response = requests.get(f"{skill_engine.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                ollama_models = response.json().get("models", [])
                available_models["ollama_models"] = [model["name"] for model in ollama_models]
                available_models["ollama_status"] = "available"
            else:
                available_models["ollama_status"] = "unavailable"
        except requests.exceptions.RequestException:
            available_models["ollama_status"] = "unavailable"
        
        return {
            "status": "success",
            "available_models": available_models,
            "recommendation": "Use Ollama models for privacy and cost-effectiveness, cloud models for advanced capabilities"
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
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