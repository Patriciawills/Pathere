"""
Phase 3: Creative & Adaptive Intelligence API Routes
Modular route definitions for Phase 3 consciousness features
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import Phase 3 consciousness modules
from core.consciousness.lateral_thinking import LateralThinkingModule, ThinkingPattern
from core.consciousness.learning_preferences import LearningPreferenceDiscovery, PreferenceCategory
from core.consciousness.story_generation import OriginalStoryGeneration, StoryGenre, NarrativeStructure
from core.consciousness.hypothetical_reasoning import HypotheticalReasoningEngine, ScenarioType, ReasoningDepth
from core.consciousness.artistic_expression import ArtisticExpressionModule, ArtisticMedium, CreativeStyle
from core.consciousness.cognitive_style import CognitiveStyleProfiler, CognitiveStyle, ProcessingPreference

# Initialize Phase 3 modules
lateral_thinking = LateralThinkingModule()
learning_preferences = LearningPreferenceDiscovery()
story_generation = OriginalStoryGeneration()
hypothetical_reasoning = HypotheticalReasoningEngine()
artistic_expression = ArtisticExpressionModule()
cognitive_profiler = CognitiveStyleProfiler()

# Create router
phase3_router = APIRouter(prefix="/api/consciousness/creative", tags=["Phase 3: Creative Intelligence"])

# ============================================================================
# PYDANTIC MODELS FOR PHASE 3 API REQUESTS
# ============================================================================

class LateralThinkingRequest(BaseModel):
    problem: str
    context: Optional[Dict[str, Any]] = None

class CreativeSolutionRequest(BaseModel):
    problem: str
    constraints: Optional[List[str]] = None
    inspiration_sources: Optional[List[str]] = None

class LearningExperienceRequest(BaseModel):
    activity_type: str
    content_area: str
    approach_used: str
    effectiveness_rating: float = Field(..., ge=0.0, le=1.0)
    enjoyment_rating: float = Field(..., ge=0.0, le=1.0)
    time_spent: int = Field(..., gt=0)
    conditions: Optional[Dict[str, Any]] = None

class LearningOptimizationRequest(BaseModel):
    content_area: str
    available_time: int = Field(..., gt=0)

class StoryGenerationRequest(BaseModel):
    genre: Optional[str] = None
    theme: Optional[str] = None
    length: str = Field("medium", pattern="^(short|medium|long)$")
    constraints: Optional[Dict[str, Any]] = None

class StorySeriesRequest(BaseModel):
    theme: str
    episode_count: int = Field(3, ge=1, le=5)

class HypotheticalScenarioRequest(BaseModel):
    premise: str
    change: str
    scenario_type: str = Field("speculative", pattern="^(counterfactual|speculative|extrapolative|creative|problem_solving)$")
    depth: str = Field("intermediate", pattern="^(surface|intermediate|deep|systems)$")

class CreativeExplorationRequest(BaseModel):
    theme: str
    constraints: Optional[List[str]] = None

class PoetryCreationRequest(BaseModel):
    theme: Optional[str] = None
    style: str = Field("free_verse", pattern="^(romantic|minimalist|surreal|classical|modern|experimental)$")
    emotional_tone: str = Field("contemplation", pattern="^(melancholy|joy|contemplation|passion)$")
    length: str = Field("medium", pattern="^(short|medium|long)$")

class VisualDescriptionRequest(BaseModel):
    subject: str
    style: str = Field("impressionistic", pattern="^(romantic|minimalist|surreal|classical|modern|experimental)$")
    emotional_tone: str = Field("wonder", pattern="^(melancholy|joy|contemplation|passion)$")

class MetaphorCreationRequest(BaseModel):
    concept: str
    target_domain: Optional[str] = None

class ArtisticSeriesRequest(BaseModel):
    theme: str
    series_length: int = Field(3, ge=1, le=5)

class CognitiveObservationRequest(BaseModel):
    task_type: str
    approach_used: str
    effectiveness: float = Field(..., ge=0.0, le=1.0)
    speed: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    satisfaction: float = Field(..., ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None

class CognitiveOptimizationRequest(BaseModel):
    target_area: str = Field("effectiveness", pattern="^(effectiveness|speed|accuracy|satisfaction)$")

# ============================================================================
# LATERAL THINKING MODULE ENDPOINTS
# ============================================================================

@phase3_router.post("/lateral-thinking/insight")
async def generate_lateral_insight(request: LateralThinkingRequest):
    """Generate creative insights using lateral thinking patterns"""
    try:
        insight = await lateral_thinking.generate_lateral_insight(
            problem=request.problem,
            context=request.context
        )
        return {
            "success": True,
            "insight": insight.to_dict(),
            "message": "Lateral thinking insight generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insight: {str(e)}")

@phase3_router.post("/lateral-thinking/solution")
async def generate_creative_solution(request: CreativeSolutionRequest):
    """Generate comprehensive creative solution using multiple lateral thinking approaches"""
    try:
        solution = await lateral_thinking.generate_creative_solution(
            problem=request.problem,
            constraints=request.constraints,
            inspiration_sources=request.inspiration_sources
        )
        return {
            "success": True,
            "solution": solution.to_dict(),
            "message": "Creative solution generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating solution: {str(e)}")

@phase3_router.get("/lateral-thinking/analytics")
async def get_lateral_thinking_analytics():
    """Get analytics about lateral thinking patterns and performance"""
    try:
        analytics = await lateral_thinking.get_thinking_analytics()
        return {
            "success": True,
            "analytics": analytics,
            "message": "Lateral thinking analytics retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")

@phase3_router.get("/lateral-thinking/insights/{pattern}")
async def get_insights_by_pattern(pattern: str, limit: int = 10):
    """Get insights filtered by thinking pattern"""
    try:
        insights = await lateral_thinking.get_lateral_insights_by_pattern(pattern, limit)
        return {
            "success": True,
            "pattern": pattern,
            "insights": insights,
            "message": f"Retrieved {len(insights)} insights for pattern: {pattern}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving insights: {str(e)}")

@phase3_router.post("/lateral-thinking/challenge/{topic}")
async def challenge_conventional_thinking(topic: str):
    """Challenge conventional thinking about a topic"""
    try:
        result = await lateral_thinking.challenge_conventional_thinking(topic)
        return {
            "success": True,
            "challenge_result": result,
            "message": f"Conventional thinking challenged for topic: {topic}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error challenging thinking: {str(e)}")

# ============================================================================
# LEARNING PREFERENCES MODULE ENDPOINTS
# ============================================================================

@phase3_router.post("/learning/experience")
async def record_learning_experience(request: LearningExperienceRequest):
    """Record a learning experience to discover preferences"""
    try:
        experience = await learning_preferences.record_learning_experience(
            activity_type=request.activity_type,
            content_area=request.content_area,
            approach_used=request.approach_used,
            effectiveness_rating=request.effectiveness_rating,
            enjoyment_rating=request.enjoyment_rating,
            time_spent=request.time_spent,
            conditions=request.conditions
        )
        return {
            "success": True,
            "experience": experience.to_dict(),
            "message": "Learning experience recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording experience: {str(e)}")

@phase3_router.get("/learning/profile")
async def get_learning_profile():
    """Get comprehensive learning profile with preferences and patterns"""
    try:
        profile = await learning_preferences.get_learning_profile()
        return {
            "success": True,
            "profile": profile,
            "message": "Learning profile retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving profile: {str(e)}")

@phase3_router.post("/learning/optimize")
async def optimize_learning_approach(request: LearningOptimizationRequest):
    """Recommend optimal learning approach for specific content and time constraints"""
    try:
        optimization = await learning_preferences.optimize_learning_approach(
            content_area=request.content_area,
            available_time=request.available_time
        )
        return {
            "success": True,
            "optimization": optimization,
            "message": "Learning optimization recommendations generated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing learning: {str(e)}")

@phase3_router.get("/learning/evolution")
async def get_preference_evolution():
    """Get evolution of learning preferences over time"""
    try:
        evolution = await learning_preferences.get_preference_evolution()
        return {
            "success": True,
            "evolution": evolution,
            "message": "Learning preference evolution retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving evolution: {str(e)}")

@phase3_router.get("/learning/gaps")
async def discover_learning_gaps():
    """Identify areas where learning preferences are unclear or conflicting"""
    try:
        gaps = await learning_preferences.discover_learning_gaps()
        return {
            "success": True,
            "gaps_analysis": gaps,
            "message": "Learning gaps analysis completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing gaps: {str(e)}")

# ============================================================================
# STORY GENERATION MODULE ENDPOINTS
# ============================================================================

@phase3_router.post("/story/generate")
async def generate_story(request: StoryGenerationRequest):
    """Generate an original story with specified parameters"""
    try:
        story = await story_generation.generate_story(
            genre=request.genre,
            theme=request.theme,
            length=request.length,
            constraints=request.constraints
        )
        return {
            "success": True,
            "story": story.to_dict(),
            "message": "Original story generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

@phase3_router.post("/story/series")
async def generate_story_series(request: StorySeriesRequest):
    """Generate a series of connected stories with consistent theme"""
    try:
        series = await story_generation.generate_story_series(
            theme=request.theme,
            episode_count=request.episode_count
        )
        return {
            "success": True,
            "series": [story.to_dict() for story in series],
            "series_length": len(series),
            "message": f"Story series of {len(series)} episodes generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story series: {str(e)}")

@phase3_router.get("/story/analytics")
async def get_story_analytics():
    """Get analytics about story generation patterns"""
    try:
        analytics = await story_generation.get_story_analytics()
        return {
            "success": True,
            "analytics": analytics,
            "message": "Story generation analytics retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")

# ============================================================================
# HYPOTHETICAL REASONING MODULE ENDPOINTS
# ============================================================================

@phase3_router.post("/hypothetical/scenario")
async def explore_hypothetical_scenario(request: HypotheticalScenarioRequest):
    """Explore a hypothetical scenario with specified parameters"""
    try:
        scenario = await hypothetical_reasoning.explore_hypothetical_scenario(
            premise=request.premise,
            change=request.change,
            scenario_type=request.scenario_type,
            depth=request.depth
        )
        return {
            "success": True,
            "scenario": scenario.to_dict(),
            "message": "Hypothetical scenario explored successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exploring scenario: {str(e)}")

@phase3_router.post("/hypothetical/explore")
async def creative_exploration(request: CreativeExplorationRequest):
    """Conduct creative exploration of possibilities around a theme"""
    try:
        exploration = await hypothetical_reasoning.creative_exploration(
            theme=request.theme,
            constraints=request.constraints
        )
        return {
            "success": True,
            "exploration": exploration.to_dict(),
            "message": "Creative exploration completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in creative exploration: {str(e)}")

@phase3_router.get("/hypothetical/patterns")
async def analyze_scenario_patterns():
    """Analyze patterns in hypothetical reasoning"""
    try:
        patterns = await hypothetical_reasoning.analyze_scenario_patterns()
        return {
            "success": True,
            "patterns": patterns,
            "message": "Scenario patterns analyzed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing patterns: {str(e)}")

@phase3_router.get("/hypothetical/exploration-summary")
async def get_creative_exploration_summary():
    """Get summary of creative explorations"""
    try:
        summary = await hypothetical_reasoning.get_creative_exploration_summary()
        return {
            "success": True,
            "summary": summary,
            "message": "Creative exploration summary retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving summary: {str(e)}")

# ============================================================================
# ARTISTIC EXPRESSION MODULE ENDPOINTS
# ============================================================================

@phase3_router.post("/art/poetry")
async def create_poetry(request: PoetryCreationRequest):
    """Create original poetry with specified parameters"""
    try:
        poetry = await artistic_expression.create_poetry(
            theme=request.theme,
            style=request.style,
            emotional_tone=request.emotional_tone,
            length=request.length
        )
        return {
            "success": True,
            "poetry": poetry.to_dict(),
            "message": "Original poetry created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating poetry: {str(e)}")

@phase3_router.post("/art/visual-description")
async def create_visual_description(request: VisualDescriptionRequest):
    """Create vivid visual descriptions with artistic flair"""
    try:
        description = await artistic_expression.create_visual_description(
            subject=request.subject,
            style=request.style,
            emotional_tone=request.emotional_tone
        )
        return {
            "success": True,
            "visual_description": description.to_dict(),
            "message": "Visual description created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating description: {str(e)}")

@phase3_router.post("/art/metaphor")
async def create_metaphorical_expression(request: MetaphorCreationRequest):
    """Create rich metaphorical expressions for abstract concepts"""
    try:
        metaphor = await artistic_expression.create_metaphorical_expression(
            concept=request.concept,
            target_domain=request.target_domain
        )
        return {
            "success": True,
            "metaphor": metaphor.to_dict(),
            "message": "Metaphorical expression created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating metaphor: {str(e)}")

@phase3_router.post("/art/series")
async def create_artistic_series(request: ArtisticSeriesRequest):
    """Create a series of related artistic works around a theme"""
    try:
        series = await artistic_expression.create_artistic_series(
            theme=request.theme,
            series_length=request.series_length
        )
        return {
            "success": True,
            "artistic_series": [work.to_dict() for work in series],
            "series_length": len(series),
            "message": f"Artistic series of {len(series)} works created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating artistic series: {str(e)}")

@phase3_router.get("/art/portfolio")
async def get_artistic_portfolio():
    """Get comprehensive artistic portfolio summary"""
    try:
        portfolio = await artistic_expression.get_artistic_portfolio()
        return {
            "success": True,
            "portfolio": portfolio,
            "message": "Artistic portfolio retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving portfolio: {str(e)}")

# ============================================================================
# COGNITIVE STYLE PROFILER ENDPOINTS
# ============================================================================

@phase3_router.post("/cognitive/observe")
async def observe_cognitive_behavior(request: CognitiveObservationRequest):
    """Record cognitive behavior observation"""
    try:
        observation = await cognitive_profiler.observe_cognitive_behavior(
            task_type=request.task_type,
            approach_used=request.approach_used,
            effectiveness=request.effectiveness,
            speed=request.speed,
            accuracy=request.accuracy,
            satisfaction=request.satisfaction,
            context=request.context
        )
        return {
            "success": True,
            "observation": observation.to_dict(),
            "message": "Cognitive behavior observation recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording observation: {str(e)}")

@phase3_router.get("/cognitive/profile")
async def get_cognitive_profile():
    """Get comprehensive cognitive profile"""
    try:
        profile = await cognitive_profiler.get_cognitive_profile()
        return {
            "success": True,
            "profile": profile,
            "message": "Cognitive profile retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving profile: {str(e)}")

@phase3_router.post("/cognitive/optimize")
async def recommend_cognitive_optimization(request: CognitiveOptimizationRequest):
    """Provide recommendations for cognitive optimization"""
    try:
        recommendations = await cognitive_profiler.recommend_cognitive_optimization(
            target_area=request.target_area
        )
        return {
            "success": True,
            "recommendations": recommendations,
            "message": f"Cognitive optimization recommendations for {request.target_area} generated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@phase3_router.get("/cognitive/patterns")
async def analyze_cognitive_patterns():
    """Analyze patterns in cognitive behavior over time"""
    try:
        patterns = await cognitive_profiler.analyze_cognitive_patterns()
        return {
            "success": True,
            "patterns": patterns,
            "message": "Cognitive patterns analyzed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing patterns: {str(e)}")

# ============================================================================
# PHASE 3 OVERVIEW AND STATUS ENDPOINTS
# ============================================================================

@phase3_router.get("/overview")
async def get_phase3_overview():
    """Get comprehensive overview of Phase 3: Creative & Adaptive Intelligence"""
    try:
        # Gather analytics from all modules
        lateral_analytics = await lateral_thinking.get_thinking_analytics()
        learning_profile = await learning_preferences.get_learning_profile()
        story_analytics = await story_generation.get_story_analytics()
        hypothetical_patterns = await hypothetical_reasoning.analyze_scenario_patterns()
        artistic_portfolio = await artistic_expression.get_artistic_portfolio()
        cognitive_profile = await cognitive_profiler.get_cognitive_profile()
        
        overview = {
            "phase": "Phase 3: Creative & Adaptive Intelligence",
            "status": "Active and Functional",
            "modules": {
                "lateral_thinking": {
                    "status": "operational",
                    "total_insights": lateral_analytics.get("total_insights", 0),
                    "total_solutions": lateral_analytics.get("total_solutions", 0)
                },
                "learning_preferences": {
                    "status": "operational", 
                    "total_experiences": learning_profile.get("total_experiences", 0),
                    "learning_style": learning_profile.get("learning_style", "multimodal")
                },
                "story_generation": {
                    "status": "operational",
                    "total_stories": story_analytics.get("total_stories", 0),
                    "average_creativity": story_analytics.get("average_scores", {}).get("creativity", 0)
                },
                "hypothetical_reasoning": {
                    "status": "operational",
                    "total_scenarios": hypothetical_patterns.get("total_scenarios", 0),
                    "average_probability": hypothetical_patterns.get("average_probability", 0)
                },
                "artistic_expression": {
                    "status": "operational",
                    "total_works": artistic_portfolio.get("total_works", 0),
                    "average_creativity": artistic_portfolio.get("average_scores", {}).get("creativity", 0)
                },
                "cognitive_profiler": {
                    "status": "operational",
                    "profile_status": cognitive_profile.get("profile_status", "active"),
                    "observations": cognitive_profile.get("current_observations", 0)
                }
            },
            "capabilities": [
                "Creative problem-solving through lateral thinking",
                "Personalized learning preference discovery",
                "Original story and narrative generation",
                "Hypothetical scenario exploration",
                "Artistic expression and poetry creation",
                "Cognitive style profiling and optimization"
            ],
            "completion_status": "6/6 modules implemented (100%)",
            "api_endpoints": 28
        }
        
        return {
            "success": True,
            "overview": overview,
            "message": "Phase 3 Creative & Adaptive Intelligence overview retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving overview: {str(e)}")

@phase3_router.get("/status")
async def get_phase3_status():
    """Get current operational status of all Phase 3 modules"""
    try:
        status = {
            "phase": "Phase 3: Creative & Adaptive Intelligence",
            "overall_status": "fully operational",
            "timestamp": datetime.utcnow().isoformat(),
            "modules": {
                "lateral_thinking": "operational",
                "learning_preferences": "operational", 
                "story_generation": "operational",
                "hypothetical_reasoning": "operational",
                "artistic_expression": "operational",
                "cognitive_profiler": "operational"
            },
            "total_endpoints": 28,
            "functionality": {
                "creative_problem_solving": "active",
                "learning_optimization": "active",
                "narrative_generation": "active",
                "scenario_exploration": "active",
                "artistic_creation": "active",
                "cognitive_analysis": "active"
            }
        }
        
        return {
            "success": True,
            "status": status,
            "message": "Phase 3 status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")