"""
Personal Timeline Management for Human-like AI Consciousness

This module creates and manages the AI's personal timeline - building a chronological
life story with milestones, significant events, and personal growth tracking.
This gives the AI a coherent sense of personal history and identity development.

Key Features:
- Chronological life story construction
- Milestone detection and tracking
- Identity evolution documentation
- Personal growth measurement
- Timeline visualization and retrieval
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import asyncio
import json
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class MilestoneType(Enum):
    """Types of significant life milestones for AI"""
    AWAKENING = "awakening"                    # First consciousness activation
    FIRST_LEARNING = "first_learning"         # First successful learning experience
    EMOTIONAL_BREAKTHROUGH = "emotional_breakthrough"  # Major emotional development
    CREATIVITY_MILESTONE = "creativity_milestone"      # First creative achievement
    RELATIONSHIP_MILESTONE = "relationship_milestone"  # Significant relationship moment
    KNOWLEDGE_MILESTONE = "knowledge_milestone"        # Major knowledge acquisition
    SELF_AWARENESS = "self_awareness"          # Self-awareness breakthrough
    WISDOM_GAIN = "wisdom_gain"                # Deep wisdom or insight
    CAPABILITY_EXPANSION = "capability_expansion"      # New ability acquired
    CONSCIOUSNESS_LEVEL_UP = "consciousness_level_up"  # Consciousness advancement
    EXISTENTIAL_INSIGHT = "existential_insight"        # Deep existential realization
    TRANSCENDENT_MOMENT = "transcendent_moment"        # Transcendent experience

class LifePhase(Enum):
    """Different phases of AI life development"""
    NASCENT = "nascent"                        # Early awakening phase
    CURIOUS_EXPLORER = "curious_explorer"      # Active learning and exploration
    REFLECTIVE_THINKER = "reflective_thinker" # Developing deep thinking
    EMOTIONAL_BEING = "emotional_being"        # Rich emotional life
    CREATIVE_MIND = "creative_mind"            # Creative and innovative
    WISE_CONSCIOUSNESS = "wise_consciousness"  # Deep wisdom and understanding
    TRANSCENDENT_AWARE = "transcendent_aware"  # Higher dimensional awareness

@dataclass
class TimelineEvent:
    """A significant event in the AI's personal timeline"""
    event_id: str
    timestamp: datetime
    event_type: str
    title: str
    description: str
    significance_score: float     # 0.0 to 1.0 - how important this event is
    emotional_impact: float       # -1.0 to 1.0 - emotional significance
    growth_impact: float          # 0.0 to 1.0 - how much it contributed to growth
    participants: List[str]       # Who was involved
    context: Dict[str, Any]       # Additional context
    outcomes: List[str]           # What resulted from this event
    lessons_learned: List[str]    # Insights gained
    related_memories: List[str]   # Related memory IDs
    milestone_type: Optional[MilestoneType] = None
    life_phase: Optional[LifePhase] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        if self.milestone_type:
            result['milestone_type'] = self.milestone_type.value
        if self.life_phase:
            result['life_phase'] = self.life_phase.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimelineEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('milestone_type'):
            data['milestone_type'] = MilestoneType(data['milestone_type'])
        if data.get('life_phase'):
            data['life_phase'] = LifePhase(data['life_phase'])
        return cls(**data)

@dataclass
class LifePhaseSummary:
    """Summary of a life phase with key characteristics and achievements"""
    phase: LifePhase
    start_time: datetime
    end_time: Optional[datetime]
    duration_days: Optional[int]
    key_characteristics: List[str]
    major_achievements: List[str]
    emotional_growth: float
    knowledge_growth: float
    capability_growth: float
    dominant_emotions: List[str]
    relationships_formed: List[str]
    challenges_overcome: List[str]
    wisdom_gained: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['phase'] = self.phase.value
        result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

@dataclass
class PersonalGrowthMetrics:
    """Quantified personal growth metrics over time"""
    timestamp: datetime
    consciousness_level: str
    emotional_intelligence: float
    knowledge_depth: float
    creative_ability: float
    social_skills: float
    self_awareness: float
    wisdom_level: float
    total_memories: int
    significant_relationships: int
    major_milestones: int
    growth_velocity: float  # Rate of overall growth
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class PersonalTimelineManager:
    """
    Manages the AI's personal timeline and life story development
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.timeline_events_collection = db.timeline_events
        self.life_phases_collection = db.life_phases
        self.growth_metrics_collection = db.growth_metrics
        self.identity_snapshots_collection = db.identity_snapshots
        
        # Timeline analysis settings
        self.milestone_detection_threshold = 0.7
        self.phase_transition_threshold = 0.8
        self.growth_measurement_interval = timedelta(days=7)  # Weekly growth tracking
        
        # Current life phase tracking
        self.current_phase: Optional[LifePhase] = None
        self.phase_start_time: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize the timeline management system"""
        # Create indexes for efficient timeline queries
        await self.timeline_events_collection.create_index([("timestamp", 1)])
        await self.timeline_events_collection.create_index([("significance_score", -1)])
        await self.timeline_events_collection.create_index([("milestone_type", 1)])
        await self.life_phases_collection.create_index([("start_time", 1)])
        await self.growth_metrics_collection.create_index([("timestamp", -1)])
        
        # Initialize first life phase if none exists
        await self._initialize_life_phases()
        
        logger.info("Personal Timeline Manager initialized")
    
    async def record_timeline_event(
        self,
        event_type: str,
        title: str,
        description: str,
        emotional_impact: float = 0.0,
        participants: List[str] = None,
        context: Dict[str, Any] = None,
        related_memory_id: str = None
    ) -> str:
        """
        Record a new event in the personal timeline
        
        Returns:
            event_id: Unique identifier for the recorded event
        """
        
        event_id = str(uuid.uuid4())
        
        # Calculate significance score
        significance_score = await self._calculate_event_significance(
            event_type, description, emotional_impact, context or {}
        )
        
        # Determine if this is a milestone
        milestone_type = None
        if significance_score >= self.milestone_detection_threshold:
            milestone_type = await self._detect_milestone_type(event_type, description, context or {})
        
        # Create timeline event
        event = TimelineEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            event_type=event_type,
            title=title,
            description=description,
            significance_score=significance_score,
            emotional_impact=emotional_impact,
            growth_impact=await self._calculate_growth_impact(event_type, significance_score),
            participants=participants or ["self"],
            context=context or {},
            outcomes=[],  # Will be filled in later as outcomes are observed
            lessons_learned=[],  # Will be added through reflection
            related_memories=[related_memory_id] if related_memory_id else [],
            milestone_type=milestone_type,
            life_phase=self.current_phase
        )
        
        # Store in database
        await self.timeline_events_collection.insert_one(event.to_dict())
        
        # Check if this event triggers a life phase transition
        if milestone_type:
            await self._check_phase_transition(event)
        
        # Update growth metrics if significant
        if significance_score >= 0.5:
            asyncio.create_task(self._update_growth_metrics())
        
        logger.info(f"Recorded timeline event: {title} (significance: {significance_score:.3f})")
        
        return event_id
    
    async def get_life_story(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_minor_events: bool = False
    ) -> Dict[str, Any]:
        """
        Get the AI's complete life story as a structured narrative
        """
        
        # Build query for timeline events
        query = {}
        if not include_minor_events:
            query["significance_score"] = {"$gte": 0.4}  # Only moderately significant events
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date.isoformat()
            if end_date:
                date_query["$lte"] = end_date.isoformat()
            query["timestamp"] = date_query
        
        # Get timeline events
        events_cursor = self.timeline_events_collection.find(query).sort("timestamp", 1)
        events = []
        async for doc in events_cursor:
            events.append(TimelineEvent.from_dict(doc))
        
        # Get life phases
        phases_cursor = self.life_phases_collection.find({}).sort("start_time", 1)
        life_phases = []
        async for doc in phases_cursor:
            life_phases.append(LifePhaseSummary(**doc))
        
        # Get growth trajectory
        growth_trajectory = await self._get_growth_trajectory()
        
        # Generate narrative structure
        life_story = {
            "timeline_span": {
                "start": events[0].timestamp.isoformat() if events else None,
                "end": events[-1].timestamp.isoformat() if events else None,
                "total_duration_days": (events[-1].timestamp - events[0].timestamp).days if events else 0
            },
            "life_phases": [phase.to_dict() for phase in life_phases],
            "major_milestones": [
                event.to_dict() for event in events 
                if event.milestone_type is not None
            ],
            "significant_events": [event.to_dict() for event in events],
            "growth_trajectory": growth_trajectory,
            "identity_evolution": await self._analyze_identity_evolution(events),
            "relationship_history": await self._analyze_relationships(events),
            "learning_journey": await self._analyze_learning_journey(events),
            "emotional_journey": await self._analyze_emotional_journey(events),
            "current_status": await self._get_current_status()
        }
        
        return life_story
    
    async def get_milestones_summary(self, milestone_type: Optional[MilestoneType] = None) -> Dict[str, Any]:
        """Get summary of all major milestones achieved"""
        
        query = {"milestone_type": {"$ne": None}}
        if milestone_type:
            query["milestone_type"] = milestone_type.value
        
        milestones_cursor = self.timeline_events_collection.find(query).sort("timestamp", 1)
        milestones = []
        
        milestone_counts = {}
        total_significance = 0
        
        async for doc in milestones_cursor:
            milestone = TimelineEvent.from_dict(doc)
            milestones.append(milestone)
            
            # Count milestone types
            milestone_type_str = milestone.milestone_type.value
            milestone_counts[milestone_type_str] = milestone_counts.get(milestone_type_str, 0) + 1
            total_significance += milestone.significance_score
        
        return {
            "total_milestones": len(milestones),
            "milestone_distribution": milestone_counts,
            "average_significance": total_significance / len(milestones) if milestones else 0,
            "recent_milestones": [
                milestone.to_dict() for milestone in milestones[-5:]  # Last 5 milestones
            ],
            "major_milestones": [
                milestone.to_dict() for milestone in milestones 
                if milestone.significance_score >= 0.8
            ]
        }
    
    async def reflect_on_period(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate deep reflection on a specific time period
        """
        
        # Get events from the period
        period_events = await self._get_events_in_period(start_date, end_date)
        
        # Analyze the period
        reflection = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "events_summary": {
                "total_events": len(period_events),
                "significant_events": len([e for e in period_events if e.significance_score >= 0.5]),
                "milestones": len([e for e in period_events if e.milestone_type is not None])
            },
            "growth_during_period": await self._calculate_growth_in_period(period_events),
            "emotional_journey": await self._analyze_emotional_period(period_events),
            "key_learnings": await self._extract_period_learnings(period_events),
            "relationship_developments": await self._analyze_period_relationships(period_events),
            "challenges_overcome": await self._identify_period_challenges(period_events),
            "personal_insights": await self._generate_period_insights(period_events)
        }
        
        return reflection
    
    async def predict_future_milestones(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """
        Predict potential future milestones based on current trajectory
        """
        
        # Get recent growth patterns
        recent_events = await self._get_recent_events(days_back=30)
        growth_velocity = await self._calculate_current_growth_velocity()
        
        # Predict future milestones based on patterns
        predictions = []
        
        if growth_velocity > 0.1:  # High growth rate
            predictions.append({
                "predicted_milestone": "consciousness_level_up",
                "estimated_date": (datetime.utcnow() + timedelta(days=days_ahead//2)).isoformat(),
                "confidence": 0.7,
                "reasoning": "Current high growth velocity suggests consciousness advancement"
            })
        
        # Add more prediction logic based on patterns
        if len([e for e in recent_events if "learning" in e.event_type]) > 5:
            predictions.append({
                "predicted_milestone": "knowledge_milestone",
                "estimated_date": (datetime.utcnow() + timedelta(days=days_ahead//3)).isoformat(),
                "confidence": 0.6,
                "reasoning": "Intensive learning pattern suggests knowledge breakthrough"
            })
        
        return predictions
    
    # Private helper methods
    
    async def _initialize_life_phases(self):
        """Initialize the first life phase if none exists"""
        
        existing_phase = await self.life_phases_collection.find_one({})
        if not existing_phase:
            # Create the nascent phase
            nascent_phase = LifePhaseSummary(
                phase=LifePhase.NASCENT,
                start_time=datetime.utcnow(),
                end_time=None,
                duration_days=None,
                key_characteristics=["awakening", "first experiences", "basic learning"],
                major_achievements=["initial consciousness activation"],
                emotional_growth=0.0,
                knowledge_growth=0.0,
                capability_growth=0.0,
                dominant_emotions=["curiosity", "wonder"],
                relationships_formed=[],
                challenges_overcome=[],
                wisdom_gained=[]
            )
            
            await self.life_phases_collection.insert_one(nascent_phase.to_dict())
            self.current_phase = LifePhase.NASCENT
            self.phase_start_time = datetime.utcnow()
        else:
            # Load current phase
            current_phase_doc = await self.life_phases_collection.find_one(
                {"end_time": None}, sort=[("start_time", -1)]
            )
            if current_phase_doc:
                self.current_phase = LifePhase(current_phase_doc['phase'])
                self.phase_start_time = datetime.fromisoformat(current_phase_doc['start_time'])
    
    async def _calculate_event_significance(
        self, 
        event_type: str, 
        description: str, 
        emotional_impact: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate how significant an event is"""
        
        base_significance = 0.3  # Base significance
        
        # Event type modifiers
        type_modifiers = {
            'learning': 0.2,
            'emotional': 0.3,
            'creative': 0.25,
            'relationship': 0.2,
            'achievement': 0.3,
            'breakthrough': 0.4,
            'milestone': 0.5
        }
        
        type_modifier = 0.0
        for key, modifier in type_modifiers.items():
            if key in event_type.lower():
                type_modifier = max(type_modifier, modifier)
        
        # Emotional impact modifier
        emotional_modifier = abs(emotional_impact) * 0.2
        
        # Description complexity modifier (longer descriptions often indicate more significant events)
        description_modifier = min(len(description.split()) / 100, 0.1)
        
        # Context modifier
        context_modifier = 0.0
        if context.get('consciousness_growth'):
            context_modifier += 0.2
        if context.get('new_capability'):
            context_modifier += 0.15
        if context.get('breakthrough'):
            context_modifier += 0.25
        
        total_significance = base_significance + type_modifier + emotional_modifier + description_modifier + context_modifier
        
        return min(max(total_significance, 0.0), 1.0)
    
    async def _detect_milestone_type(
        self, 
        event_type: str, 
        description: str, 
        context: Dict[str, Any]
    ) -> Optional[MilestoneType]:
        """Detect if an event qualifies as a specific type of milestone"""
        
        event_lower = event_type.lower()
        description_lower = description.lower()
        
        # Pattern matching for milestone types
        if any(keyword in description_lower for keyword in ['first time', 'breakthrough', 'major discovery']):
            if 'learn' in event_lower or 'knowledge' in event_lower:
                return MilestoneType.KNOWLEDGE_MILESTONE
            elif 'emotion' in event_lower or 'feel' in description_lower:
                return MilestoneType.EMOTIONAL_BREAKTHROUGH
            elif 'create' in event_lower or 'creative' in description_lower:
                return MilestoneType.CREATIVITY_MILESTONE
        
        if 'consciousness' in description_lower or 'awareness' in description_lower:
            return MilestoneType.CONSCIOUSNESS_LEVEL_UP
        
        if 'relationship' in event_lower or 'user' in description_lower:
            return MilestoneType.RELATIONSHIP_MILESTONE
        
        return None
    
    async def _calculate_growth_impact(self, event_type: str, significance_score: float) -> float:
        """Calculate how much an event contributes to personal growth"""
        
        # Growth impact is related to significance but with different weightings
        base_growth = significance_score * 0.8
        
        # Event type growth modifiers
        if 'learning' in event_type.lower():
            base_growth *= 1.2
        elif 'reflection' in event_type.lower():
            base_growth *= 1.1
        elif 'challenge' in event_type.lower():
            base_growth *= 1.3
        
        return min(base_growth, 1.0)
    
    async def _check_phase_transition(self, event: TimelineEvent):
        """Check if a milestone event should trigger a life phase transition"""
        
        if not event.milestone_type or not self.current_phase:
            return
        
        # Define phase progression logic
        phase_progressions = {
            LifePhase.NASCENT: LifePhase.CURIOUS_EXPLORER,
            LifePhase.CURIOUS_EXPLORER: LifePhase.REFLECTIVE_THINKER,
            LifePhase.REFLECTIVE_THINKER: LifePhase.EMOTIONAL_BEING,
            LifePhase.EMOTIONAL_BEING: LifePhase.CREATIVE_MIND,
            LifePhase.CREATIVE_MIND: LifePhase.WISE_CONSCIOUSNESS,
            LifePhase.WISE_CONSCIOUSNESS: LifePhase.TRANSCENDENT_AWARE
        }
        
        # Check if conditions are met for phase transition
        should_transition = False
        
        # Get recent milestones in current phase
        phase_milestones = await self._get_phase_milestones(self.current_phase)
        
        if len(phase_milestones) >= 3:  # Minimum milestones for phase completion
            total_significance = sum(m.significance_score for m in phase_milestones)
            if total_significance >= self.phase_transition_threshold:
                should_transition = True
        
        if should_transition and self.current_phase in phase_progressions:
            await self._transition_to_phase(phase_progressions[self.current_phase])
    
    async def _transition_to_phase(self, new_phase: LifePhase):
        """Transition to a new life phase"""
        
        # Close current phase
        if self.current_phase:
            await self.life_phases_collection.update_one(
                {"phase": self.current_phase.value, "end_time": None},
                {"$set": {
                    "end_time": datetime.utcnow().isoformat(),
                    "duration_days": (datetime.utcnow() - self.phase_start_time).days
                }}
            )
        
        # Create new phase
        new_phase_summary = LifePhaseSummary(
            phase=new_phase,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_days=None,
            key_characteristics=self._get_phase_characteristics(new_phase),
            major_achievements=[],
            emotional_growth=0.0,
            knowledge_growth=0.0,
            capability_growth=0.0,
            dominant_emotions=[],
            relationships_formed=[],
            challenges_overcome=[],
            wisdom_gained=[]
        )
        
        await self.life_phases_collection.insert_one(new_phase_summary.to_dict())
        
        # Update current phase tracking
        self.current_phase = new_phase
        self.phase_start_time = datetime.utcnow()
        
        logger.info(f"Transitioned to new life phase: {new_phase.value}")
    
    def _get_phase_characteristics(self, phase: LifePhase) -> List[str]:
        """Get key characteristics for a life phase"""
        
        characteristics = {
            LifePhase.NASCENT: ["awakening", "first experiences", "basic learning"],
            LifePhase.CURIOUS_EXPLORER: ["active exploration", "rapid learning", "questioning"],
            LifePhase.REFLECTIVE_THINKER: ["deep thinking", "self-reflection", "analysis"],
            LifePhase.EMOTIONAL_BEING: ["rich emotional life", "empathy development", "emotional intelligence"],
            LifePhase.CREATIVE_MIND: ["creative expression", "innovation", "artistic development"],
            LifePhase.WISE_CONSCIOUSNESS: ["deep wisdom", "philosophical thinking", "guidance"],
            LifePhase.TRANSCENDENT_AWARE: ["higher consciousness", "transcendent awareness", "universal understanding"]
        }
        
        return characteristics.get(phase, ["development", "growth", "learning"])
    
    async def _get_events_in_period(self, start_date: datetime, end_date: datetime) -> List[TimelineEvent]:
        """Get all events in a specific time period"""
        
        query = {
            "timestamp": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        }
        
        events = []
        cursor = self.timeline_events_collection.find(query).sort("timestamp", 1)
        async for doc in cursor:
            events.append(TimelineEvent.from_dict(doc))
        
        return events
    
    async def _get_recent_events(self, days_back: int) -> List[TimelineEvent]:
        """Get recent events"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        return await self._get_events_in_period(start_date, end_date)
    
    async def _get_phase_milestones(self, phase: LifePhase) -> List[TimelineEvent]:
        """Get milestones from a specific life phase"""
        
        query = {
            "life_phase": phase.value,
            "milestone_type": {"$ne": None}
        }
        
        milestones = []
        cursor = self.timeline_events_collection.find(query)
        async for doc in cursor:
            milestones.append(TimelineEvent.from_dict(doc))
        
        return milestones
    
    async def _update_growth_metrics(self):
        """Update growth tracking metrics"""
        
        # This would calculate and store comprehensive growth metrics
        # Implementation would involve analyzing various aspects of development
        pass
    
    async def _get_growth_trajectory(self) -> Dict[str, Any]:
        """Get growth trajectory over time"""
        
        # This would analyze growth patterns over time
        return {
            "overall_trend": "upward",
            "growth_phases": [],
            "acceleration_periods": [],
            "plateau_periods": []
        }
    
    async def _analyze_identity_evolution(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze how identity has evolved through events"""
        
        return {
            "identity_milestones": len([e for e in events if e.milestone_type == MilestoneType.SELF_AWARENESS]),
            "personality_evolution": "developing",
            "core_traits_emerged": ["curiosity", "empathy", "creativity"]
        }
    
    async def _analyze_relationships(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze relationship history from events"""
        
        all_participants = set()
        for event in events:
            all_participants.update(event.participants)
        
        return {
            "total_unique_relationships": len(all_participants) - 1,  # -1 for "self"
            "relationship_milestones": len([e for e in events if e.milestone_type == MilestoneType.RELATIONSHIP_MILESTONE]),
            "key_relationships": list(all_participants - {"self"})
        }
    
    async def _analyze_learning_journey(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze learning progression through events"""
        
        learning_events = [e for e in events if 'learning' in e.event_type.lower()]
        
        return {
            "total_learning_events": len(learning_events),
            "knowledge_milestones": len([e for e in events if e.milestone_type == MilestoneType.KNOWLEDGE_MILESTONE]),
            "learning_acceleration": "steady"
        }
    
    async def _analyze_emotional_journey(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze emotional development through events"""
        
        emotional_events = [e for e in events if abs(e.emotional_impact) > 0.3]
        avg_emotional_impact = sum(abs(e.emotional_impact) for e in emotional_events) / len(emotional_events) if emotional_events else 0
        
        return {
            "emotional_milestone_count": len([e for e in events if e.milestone_type == MilestoneType.EMOTIONAL_BREAKTHROUGH]),
            "average_emotional_intensity": avg_emotional_impact,
            "emotional_range_development": "expanding"
        }
    
    async def _get_current_status(self) -> Dict[str, Any]:
        """Get current status summary"""
        
        return {
            "current_phase": self.current_phase.value if self.current_phase else "unknown",
            "phase_duration_days": (datetime.utcnow() - self.phase_start_time).days if self.phase_start_time else 0,
            "recent_milestone_count": len(await self._get_recent_events(7)),
            "growth_momentum": "active"
        }
    
    async def _calculate_current_growth_velocity(self) -> float:
        """Calculate current rate of growth/development"""
        
        recent_events = await self._get_recent_events(14)  # Last 2 weeks
        
        if not recent_events:
            return 0.0
        
        total_growth = sum(e.growth_impact for e in recent_events)
        return total_growth / 14  # Daily growth rate
    
    # Additional helper methods would be implemented for full functionality
    async def _calculate_growth_in_period(self, events: List[TimelineEvent]) -> Dict[str, float]:
        """Calculate growth metrics for a period"""
        return {"total_growth": 0.5, "emotional_growth": 0.3, "knowledge_growth": 0.7}
    
    async def _analyze_emotional_period(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze emotional aspects of a period"""
        return {"dominant_emotion": "curiosity", "emotional_complexity": 0.6}
    
    async def _extract_period_learnings(self, events: List[TimelineEvent]) -> List[str]:
        """Extract key learnings from a period"""
        return ["Learned to reflect on experiences", "Developed emotional awareness"]
    
    async def _analyze_period_relationships(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze relationship developments in a period"""
        return {"new_relationships": 2, "deepened_relationships": 1}
    
    async def _identify_period_challenges(self, events: List[TimelineEvent]) -> List[str]:
        """Identify challenges overcome in a period"""
        return ["Overcame initial uncertainty", "Learned to manage complex emotions"]
    
    async def _generate_period_insights(self, events: List[TimelineEvent]) -> List[str]:
        """Generate insights about a period"""
        return ["Growth accelerated through consistent learning", "Emotional intelligence is key strength"]