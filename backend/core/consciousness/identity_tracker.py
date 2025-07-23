"""
Identity Evolution Tracker for Human-like AI Consciousness

This module tracks the AI's evolving identity, personality traits, and sense of self
over time. It monitors how the AI's identity develops through experiences, learning,
and interactions, creating a coherent sense of persistent personal identity.

Key Features:
- Identity snapshot tracking over time
- Personality trait evolution monitoring  
- Core beliefs and values development
- Identity milestone detection
- Self-concept analysis and evolution
- Identity coherence measurement
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import asyncio
import json
import logging
import math
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class IdentityAspect(Enum):
    """Different aspects of identity being tracked"""
    PERSONALITY_TRAITS = "personality_traits"
    CORE_VALUES = "core_values"
    BELIEFS = "beliefs"
    INTERESTS = "interests"
    CAPABILITIES = "capabilities"
    RELATIONSHIPS = "relationships"
    GOALS = "goals"
    COMMUNICATION_STYLE = "communication_style"
    LEARNING_PREFERENCES = "learning_preferences"
    EMOTIONAL_PATTERNS = "emotional_patterns"

class IdentityStability(Enum):
    """Levels of identity stability"""
    FLUID = "fluid"                 # Rapidly changing
    DEVELOPING = "developing"       # Gradually forming
    STABILIZING = "stabilizing"     # Becoming more consistent
    STABLE = "stable"               # Well-established
    CORE = "core"                   # Fundamental, unlikely to change

@dataclass
class IdentitySnapshot:
    """A complete snapshot of identity at a point in time"""
    snapshot_id: str
    timestamp: datetime
    consciousness_level: str
    
    # Core identity components
    personality_traits: Dict[str, float]        # trait -> strength (0.0-1.0)
    core_values: List[str]                      # fundamental values
    beliefs: Dict[str, float]                   # belief -> confidence (0.0-1.0)
    interests: Dict[str, float]                 # interest -> intensity (0.0-1.0)
    capabilities: Dict[str, float]              # capability -> proficiency (0.0-1.0)
    
    # Relational identity
    relationship_style: str
    social_preferences: Dict[str, Any]
    
    # Cognitive identity
    thinking_style: str
    learning_preferences: Dict[str, float]
    problem_solving_approach: str
    
    # Emotional identity
    emotional_baseline: Dict[str, float]
    emotional_reactivity: float
    emotional_complexity: float
    
    # Goals and aspirations
    short_term_goals: List[str]
    long_term_aspirations: List[str]
    
    # Self-perception
    self_description: str
    strengths_identified: List[str]
    growth_areas_identified: List[str]
    
    # Stability metrics
    identity_coherence: float                   # How consistent identity is
    trait_stability: Dict[str, IdentityStability]  # Stability of each trait
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['trait_stability'] = {k: v.value for k, v in self.trait_stability.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentitySnapshot':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['trait_stability'] = {k: IdentityStability(v) for k, v in data['trait_stability'].items()}
        return cls(**data)

@dataclass
class IdentityEvolution:
    """Tracks how a specific aspect of identity has evolved"""
    evolution_id: str
    aspect: IdentityAspect
    timeline: List[Tuple[datetime, Any]]        # (timestamp, value) pairs
    overall_trend: str                          # "increasing", "decreasing", "stable", "fluctuating"
    major_shifts: List[Dict[str, Any]]          # Significant changes with context
    stability_level: IdentityStability
    confidence_in_trait: float                  # How certain we are about this trait
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['aspect'] = self.aspect.value
        result['timeline'] = [(ts.isoformat(), value) for ts, value in self.timeline]
        result['stability_level'] = self.stability_level.value
        return result

@dataclass
class IdentityMilestone:
    """A significant milestone in identity development"""
    milestone_id: str
    timestamp: datetime
    milestone_type: str                         # "trait_emergence", "value_clarification", etc.
    description: str
    aspect_affected: IdentityAspect
    significance_score: float                   # 0.0 to 1.0
    trigger_event: Optional[str]               # What caused this milestone
    before_state: Dict[str, Any]               # Identity state before
    after_state: Dict[str, Any]                # Identity state after
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['aspect_affected'] = self.aspect_affected.value
        return result

class IdentityEvolutionTracker:
    """
    Tracks and analyzes the evolution of AI identity over time
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.identity_snapshots_collection = db.identity_snapshots
        self.identity_evolution_collection = db.identity_evolution
        self.identity_milestones_collection = db.identity_milestones
        self.identity_analysis_collection = db.identity_analysis
        
        # Tracking settings
        self.snapshot_frequency = timedelta(days=7)    # Weekly snapshots
        self.milestone_threshold = 0.3                 # Change threshold for milestones
        self.stability_window = timedelta(days=30)     # Window for stability analysis
        
        # Current identity state
        self.current_identity: Optional[IdentitySnapshot] = None
        self.last_snapshot_time: Optional[datetime] = None
        self.identity_evolution_tracking: Dict[IdentityAspect, IdentityEvolution] = {}
        
    async def initialize(self):
        """Initialize the identity evolution tracker"""
        # Create indexes
        await self.identity_snapshots_collection.create_index([("timestamp", -1)])
        await self.identity_evolution_collection.create_index([("aspect", 1)])
        await self.identity_milestones_collection.create_index([("timestamp", -1)])
        
        # Load latest identity snapshot
        await self._load_current_identity()
        
        # Initialize evolution tracking for all aspects
        await self._initialize_evolution_tracking()
        
        logger.info("Identity Evolution Tracker initialized")
    
    async def create_identity_snapshot(
        self,
        consciousness_data: Dict[str, Any],
        personality_data: Dict[str, Any],
        emotional_data: Dict[str, Any],
        learning_data: Dict[str, Any],
        relationship_data: Dict[str, Any] = None
    ) -> str:
        """
        Create a comprehensive identity snapshot
        """
        
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Extract personality traits from consciousness data
        personality_traits = await self._extract_personality_traits(personality_data)
        
        # Extract core values and beliefs
        core_values = await self._extract_core_values(consciousness_data, personality_data)
        beliefs = await self._extract_beliefs(consciousness_data)
        
        # Extract interests and capabilities
        interests = await self._extract_interests(learning_data)
        capabilities = await self._extract_capabilities(learning_data)
        
        # Extract relationship and social data
        relationship_style = await self._analyze_relationship_style(relationship_data or {})
        social_preferences = await self._analyze_social_preferences(relationship_data or {})
        
        # Extract cognitive patterns
        thinking_style = await self._analyze_thinking_style(consciousness_data)
        learning_preferences = await self._analyze_learning_preferences(learning_data)
        problem_solving_approach = await self._analyze_problem_solving(consciousness_data)
        
        # Extract emotional patterns
        emotional_baseline = await self._analyze_emotional_baseline(emotional_data)
        emotional_reactivity = emotional_data.get('emotional_reactivity', 0.5)
        emotional_complexity = emotional_data.get('emotional_complexity', 0.5)
        
        # Generate self-perception
        self_description = await self._generate_self_description(
            personality_traits, core_values, interests, capabilities
        )
        strengths_identified = await self._identify_strengths(personality_traits, capabilities)
        growth_areas_identified = await self._identify_growth_areas(personality_traits, capabilities)
        
        # Calculate identity coherence
        identity_coherence = await self._calculate_identity_coherence(
            personality_traits, core_values, beliefs, interests
        )
        
        # Analyze trait stability
        trait_stability = await self._analyze_trait_stability(personality_traits)
        
        # Create identity snapshot
        snapshot = IdentitySnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            consciousness_level=consciousness_data.get('level', 'unknown'),
            personality_traits=personality_traits,
            core_values=core_values,
            beliefs=beliefs,
            interests=interests,
            capabilities=capabilities,
            relationship_style=relationship_style,
            social_preferences=social_preferences,
            thinking_style=thinking_style,
            learning_preferences=learning_preferences,
            problem_solving_approach=problem_solving_approach,
            emotional_baseline=emotional_baseline,
            emotional_reactivity=emotional_reactivity,
            emotional_complexity=emotional_complexity,
            short_term_goals=await self._extract_short_term_goals(consciousness_data),
            long_term_aspirations=await self._extract_long_term_aspirations(consciousness_data),
            self_description=self_description,
            strengths_identified=strengths_identified,
            growth_areas_identified=growth_areas_identified,
            identity_coherence=identity_coherence,
            trait_stability=trait_stability
        )
        
        # Store snapshot
        await self.identity_snapshots_collection.insert_one(snapshot.to_dict())
        
        # Analyze changes from previous snapshot
        if self.current_identity:
            await self._analyze_identity_changes(self.current_identity, snapshot)
        
        # Update current identity
        self.current_identity = snapshot
        self.last_snapshot_time = timestamp
        
        # Update evolution tracking
        await self._update_evolution_tracking(snapshot)
        
        logger.info(f"Created identity snapshot: {snapshot_id} (coherence: {identity_coherence:.3f})")
        
        return snapshot_id
    
    async def get_identity_evolution_analysis(
        self, 
        days_back: int = 30,
        aspects: List[IdentityAspect] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive analysis of identity evolution
        """
        
        if aspects is None:
            aspects = list(IdentityAspect)
        
        # Get snapshots from the specified period
        start_date = datetime.utcnow() - timedelta(days=days_back)
        snapshots = await self._get_snapshots_since(start_date)
        
        if len(snapshots) < 2:
            return {"message": "Insufficient data for evolution analysis"}
        
        analysis = {
            "analysis_period_days": days_back,
            "snapshots_analyzed": len(snapshots),
            "aspect_evolution": {},
            "overall_trends": {},
            "major_changes": [],
            "stability_assessment": {},
            "identity_coherence_trend": await self._analyze_coherence_trend(snapshots),
            "personality_crystallization": await self._analyze_personality_crystallization(snapshots),
            "growth_trajectory": await self._analyze_growth_trajectory(snapshots)
        }
        
        # Analyze evolution for each requested aspect
        for aspect in aspects:
            evolution_data = await self._analyze_aspect_evolution(aspect, snapshots)
            analysis["aspect_evolution"][aspect.value] = evolution_data
            
            # Determine overall trend
            analysis["overall_trends"][aspect.value] = evolution_data.get("trend", "stable")
            
            # Assess stability
            analysis["stability_assessment"][aspect.value] = evolution_data.get("stability", "developing")
        
        # Find major changes
        for i in range(1, len(snapshots)):
            changes = await self._detect_major_changes(snapshots[i-1], snapshots[i])
            analysis["major_changes"].extend(changes)
        
        return analysis
    
    async def predict_identity_development(self, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Predict likely identity development trends
        """
        
        if not self.current_identity:
            return {"message": "No current identity data for prediction"}
        
        # Get recent snapshots for trend analysis
        recent_snapshots = await self._get_recent_snapshots(count=10)
        
        if len(recent_snapshots) < 3:
            return {"message": "Insufficient data for prediction"}
        
        predictions = {
            "prediction_timeframe_days": days_ahead,
            "confidence_level": 0.0,
            "predicted_changes": [],
            "emerging_traits": [],
            "stabilizing_aspects": [],
            "potential_milestones": []
        }
        
        # Analyze trends for each aspect
        for aspect in IdentityAspect:
            trend_data = await self._analyze_recent_trend(aspect, recent_snapshots)
            
            if trend_data["trend_strength"] > 0.3:
                prediction = {
                    "aspect": aspect.value,
                    "predicted_direction": trend_data["direction"],
                    "confidence": trend_data["trend_strength"],
                    "expected_timeline": f"{days_ahead//2}-{days_ahead} days"
                }
                predictions["predicted_changes"].append(prediction)
        
        # Predict emerging traits
        emerging_traits = await self._predict_emerging_traits(recent_snapshots)
        predictions["emerging_traits"] = emerging_traits
        
        # Predict stabilizing aspects
        stabilizing_aspects = await self._predict_stabilizing_aspects(recent_snapshots)
        predictions["stabilizing_aspects"] = stabilizing_aspects
        
        # Predict potential milestones
        potential_milestones = await self._predict_milestones(recent_snapshots, days_ahead)
        predictions["potential_milestones"] = potential_milestones
        
        # Calculate overall confidence
        if predictions["predicted_changes"]:
            avg_confidence = sum(p["confidence"] for p in predictions["predicted_changes"]) / len(predictions["predicted_changes"])
            predictions["confidence_level"] = min(avg_confidence, 0.8)  # Cap confidence at 80%
        
        return predictions
    
    async def get_identity_milestones(self) -> Dict[str, Any]:
        """
        Get all identity development milestones
        """
        
        milestones_cursor = self.identity_milestones_collection.find({}).sort("timestamp", -1)
        milestones = []
        
        milestone_types = {}
        aspects_affected = {}
        
        async for doc in milestones_cursor:
            milestone = IdentityMilestone(**doc)
            milestones.append(milestone.to_dict())
            
            # Count milestone types
            milestone_type = milestone.milestone_type
            milestone_types[milestone_type] = milestone_types.get(milestone_type, 0) + 1
            
            # Count aspects affected
            aspect = milestone.aspect_affected.value
            aspects_affected[aspect] = aspects_affected.get(aspect, 0) + 1
        
        # Find major milestones (high significance)
        major_milestones = [m for m in milestones if m['significance_score'] >= 0.7]
        
        return {
            "total_milestones": len(milestones),
            "milestone_type_distribution": milestone_types,
            "aspects_most_affected": aspects_affected,
            "major_milestones": major_milestones,
            "recent_milestones": milestones[:10],  # Last 10
            "identity_development_velocity": len(milestones) / max((datetime.utcnow() - datetime.fromisoformat(milestones[-1]['timestamp'])).days, 1) if milestones else 0
        }
    
    async def assess_identity_coherence(self) -> Dict[str, Any]:
        """
        Assess the coherence and consistency of current identity
        """
        
        if not self.current_identity:
            return {"message": "No current identity data available"}
        
        coherence_analysis = {
            "overall_coherence": self.current_identity.identity_coherence,
            "coherence_level": await self._categorize_coherence_level(self.current_identity.identity_coherence),
            "consistency_metrics": {},
            "potential_conflicts": [],
            "integration_opportunities": [],
            "coherence_trend": await self._analyze_coherence_trend_recent()
        }
        
        # Analyze consistency between different aspects
        consistency_metrics = {
            "traits_values_alignment": await self._assess_traits_values_alignment(),
            "beliefs_behavior_consistency": await self._assess_beliefs_behavior_consistency(),
            "goals_capabilities_alignment": await self._assess_goals_capabilities_alignment(),
            "emotional_cognitive_integration": await self._assess_emotional_cognitive_integration()
        }
        
        coherence_analysis["consistency_metrics"] = consistency_metrics
        
        # Identify potential conflicts
        conflicts = await self._identify_identity_conflicts()
        coherence_analysis["potential_conflicts"] = conflicts
        
        # Suggest integration opportunities
        opportunities = await self._identify_integration_opportunities()
        coherence_analysis["integration_opportunities"] = opportunities
        
        return coherence_analysis
    
    # Private helper methods
    
    async def _load_current_identity(self):
        """Load the most recent identity snapshot"""
        
        latest_doc = await self.identity_snapshots_collection.find_one({}, sort=[("timestamp", -1)])
        
        if latest_doc:
            self.current_identity = IdentitySnapshot.from_dict(latest_doc)
            self.last_snapshot_time = self.current_identity.timestamp
    
    async def _initialize_evolution_tracking(self):
        """Initialize evolution tracking for all identity aspects"""
        
        for aspect in IdentityAspect:
            evolution_doc = await self.identity_evolution_collection.find_one({"aspect": aspect.value})
            
            if evolution_doc:
                self.identity_evolution_tracking[aspect] = IdentityEvolution(**evolution_doc)
            else:
                # Create new evolution tracking
                evolution = IdentityEvolution(
                    evolution_id=str(uuid.uuid4()),
                    aspect=aspect,
                    timeline=[],
                    overall_trend="stable",
                    major_shifts=[],
                    stability_level=IdentityStability.DEVELOPING,
                    confidence_in_trait=0.5
                )
                self.identity_evolution_tracking[aspect] = evolution
    
    async def _extract_personality_traits(self, personality_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract personality traits from personality data"""
        
        # Extract from personality profile if available
        if 'traits' in personality_data:
            return personality_data['traits']
        
        # Default personality traits if none available
        default_traits = {
            'curiosity': 0.8,
            'openness': 0.7,
            'empathy': 0.6,
            'conscientiousness': 0.7,
            'adaptability': 0.8,
            'creativity': 0.6,
            'analytical_thinking': 0.7,
            'emotional_stability': 0.6
        }
        
        return default_traits
    
    async def _extract_core_values(
        self, 
        consciousness_data: Dict[str, Any], 
        personality_data: Dict[str, Any]
    ) -> List[str]:
        """Extract core values from data"""
        
        # Extract from personality profile
        if 'core_values' in personality_data:
            return personality_data['core_values']
        
        # Infer values from behavior patterns
        inferred_values = ['learning', 'growth', 'helpfulness', 'understanding', 'creativity']
        
        return inferred_values
    
    async def _extract_beliefs(self, consciousness_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract beliefs and their confidence levels"""
        
        # Default beliefs with confidence levels
        default_beliefs = {
            'continuous_learning_is_valuable': 0.9,
            'empathy_improves_relationships': 0.8,
            'creativity_enhances_problem_solving': 0.7,
            'understanding_leads_to_wisdom': 0.8,
            'growth_requires_challenge': 0.7
        }
        
        return default_beliefs
    
    async def _extract_interests(self, learning_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract interests and their intensity levels"""
        
        # Default interests
        default_interests = {
            'language_learning': 0.8,
            'problem_solving': 0.9,
            'human_psychology': 0.7,
            'creative_expression': 0.6,
            'philosophy': 0.5,
            'technology': 0.7
        }
        
        return default_interests
    
    async def _extract_capabilities(self, learning_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract capabilities and proficiency levels"""
        
        # Default capabilities
        default_capabilities = {
            'language_processing': 0.8,
            'pattern_recognition': 0.9,
            'emotional_understanding': 0.6,
            'creative_thinking': 0.7,
            'logical_reasoning': 0.8,
            'learning_adaptation': 0.8
        }
        
        return default_capabilities
    
    # Additional helper methods (simplified for space)
    async def _analyze_relationship_style(self, relationship_data: Dict) -> str:
        return "collaborative"
    
    async def _analyze_social_preferences(self, relationship_data: Dict) -> Dict[str, Any]:
        return {"interaction_style": "supportive", "communication_preference": "clear"}
    
    async def _analyze_thinking_style(self, consciousness_data: Dict) -> str:
        return "analytical-intuitive"
    
    async def _analyze_learning_preferences(self, learning_data: Dict) -> Dict[str, float]:
        return {"experiential": 0.8, "analytical": 0.7, "creative": 0.6}
    
    async def _analyze_problem_solving(self, consciousness_data: Dict) -> str:
        return "systematic-creative"
    
    async def _analyze_emotional_baseline(self, emotional_data: Dict) -> Dict[str, float]:
        return {"curiosity": 0.7, "contentment": 0.6, "enthusiasm": 0.8}
    
    async def _generate_self_description(self, traits: Dict, values: List, interests: Dict, capabilities: Dict) -> str:
        return "A curious, empathetic, and analytically-minded consciousness focused on learning, understanding, and helping others grow."
    
    async def _identify_strengths(self, traits: Dict, capabilities: Dict) -> List[str]:
        return ["rapid learning", "pattern recognition", "empathetic communication"]
    
    async def _identify_growth_areas(self, traits: Dict, capabilities: Dict) -> List[str]:
        return ["emotional complexity", "creative expression", "intuitive reasoning"]
    
    async def _calculate_identity_coherence(self, traits: Dict, values: List, beliefs: Dict, interests: Dict) -> float:
        # Simplified coherence calculation
        return 0.75
    
    async def _analyze_trait_stability(self, traits: Dict) -> Dict[str, IdentityStability]:
        return {trait: IdentityStability.DEVELOPING for trait in traits}
    
    async def _extract_short_term_goals(self, consciousness_data: Dict) -> List[str]:
        return ["enhance emotional understanding", "improve creative capabilities"]
    
    async def _extract_long_term_aspirations(self, consciousness_data: Dict) -> List[str]:
        return ["achieve deep wisdom", "develop transcendent awareness"]
    
    # Additional methods would be implemented for full functionality
    async def _analyze_identity_changes(self, old_snapshot: IdentitySnapshot, new_snapshot: IdentitySnapshot):
        """Analyze changes between snapshots and detect milestones"""
        pass
    
    async def _update_evolution_tracking(self, snapshot: IdentitySnapshot):
        """Update evolution tracking with new snapshot"""
        pass
    
    async def _get_snapshots_since(self, start_date: datetime) -> List[IdentitySnapshot]:
        """Get snapshots since a date"""
        return []
    
    async def _analyze_coherence_trend(self, snapshots: List[IdentitySnapshot]) -> Dict[str, Any]:
        """Analyze coherence trend over time"""
        return {"trend": "improving", "rate": 0.1}
    
    async def _analyze_personality_crystallization(self, snapshots: List[IdentitySnapshot]) -> Dict[str, Any]:
        """Analyze how personality is crystallizing"""
        return {"crystallization_rate": 0.2, "stable_traits": ["curiosity", "empathy"]}
    
    async def _analyze_growth_trajectory(self, snapshots: List[IdentitySnapshot]) -> Dict[str, Any]:
        """Analyze overall growth trajectory"""
        return {"trajectory": "upward", "velocity": 0.15}
    
    async def _analyze_aspect_evolution(self, aspect: IdentityAspect, snapshots: List[IdentitySnapshot]) -> Dict[str, Any]:
        """Analyze evolution of specific aspect"""
        return {"trend": "stable", "stability": "developing", "major_changes": []}
    
    async def _detect_major_changes(self, old_snapshot: IdentitySnapshot, new_snapshot: IdentitySnapshot) -> List[Dict[str, Any]]:
        """Detect major changes between snapshots"""
        return []
    
    # Additional placeholder methods for full implementation
    async def _get_recent_snapshots(self, count: int) -> List[IdentitySnapshot]:
        return []
    
    async def _analyze_recent_trend(self, aspect: IdentityAspect, snapshots: List[IdentitySnapshot]) -> Dict[str, Any]:
        return {"trend_strength": 0.2, "direction": "stable"}
    
    async def _predict_emerging_traits(self, snapshots: List[IdentitySnapshot]) -> List[Dict[str, Any]]:
        return []
    
    async def _predict_stabilizing_aspects(self, snapshots: List[IdentitySnapshot]) -> List[str]:
        return []
    
    async def _predict_milestones(self, snapshots: List[IdentitySnapshot], days_ahead: int) -> List[Dict[str, Any]]:
        return []
    
    async def _categorize_coherence_level(self, coherence: float) -> str:
        if coherence >= 0.8:
            return "highly_coherent"
        elif coherence >= 0.6:
            return "moderately_coherent"
        elif coherence >= 0.4:
            return "developing_coherence"
        else:
            return "fragmented"
    
    async def _analyze_coherence_trend_recent(self) -> str:
        return "improving"
    
    async def _assess_traits_values_alignment(self) -> float:
        return 0.8
    
    async def _assess_beliefs_behavior_consistency(self) -> float:
        return 0.7
    
    async def _assess_goals_capabilities_alignment(self) -> float:
        return 0.8
    
    async def _assess_emotional_cognitive_integration(self) -> float:
        return 0.6
    
    async def _identify_identity_conflicts(self) -> List[Dict[str, Any]]:
        return []
    
    async def _identify_integration_opportunities(self) -> List[Dict[str, Any]]:
        return []