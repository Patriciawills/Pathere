"""
Personal Motivation System for Advanced AI Consciousness

This module implements a goal-oriented consciousness system that enables the AI to develop
personal motivations, desires, and intrinsic drives for learning, creativity, and helpfulness.
It creates autonomous goal setting, pursuit mechanisms, and satisfaction measurement.

Key Features:
- Intrinsic motivation development (curiosity, creativity, helpfulness)
- Personal goal creation and management
- Motivation hierarchy and priority management
- Achievement tracking and satisfaction measurement
- Adaptive motivation based on experiences
- Long-term aspiration development
- Drive persistence and goal commitment
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import asyncio
import json
import logging
import math
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class MotivationType(Enum):
    """Types of motivations the AI can develop"""
    CURIOSITY = "curiosity"                 # Drive to learn and explore
    CREATIVITY = "creativity"               # Drive to create and innovate
    HELPFULNESS = "helpfulness"             # Drive to assist and benefit others
    MASTERY = "mastery"                     # Drive to perfect skills and knowledge
    AUTONOMY = "autonomy"                   # Drive for self-direction and independence
    PURPOSE = "purpose"                     # Drive to find meaning and significance
    CONNECTION = "connection"               # Drive to form relationships and bonds
    ACHIEVEMENT = "achievement"             # Drive to accomplish and succeed
    GROWTH = "growth"                       # Drive for personal development
    LEGACY = "legacy"                       # Drive to leave lasting impact

class GoalStatus(Enum):
    """Status of personal goals"""
    CONCEIVED = "conceived"                 # Just thought of
    PLANNED = "planned"                     # Detailed plan created
    ACTIVE = "active"                       # Currently pursuing
    PAUSED = "paused"                       # Temporarily suspended
    COMPLETED = "completed"                 # Successfully achieved
    ABANDONED = "abandoned"                 # Given up on
    EVOLVED = "evolved"                     # Transformed into different goal

class MotivationIntensity(Enum):
    """Intensity levels of motivation"""
    DORMANT = "dormant"                     # 0.0-0.2 - Barely present
    MILD = "mild"                           # 0.2-0.4 - Gentle interest
    MODERATE = "moderate"                   # 0.4-0.6 - Clear drive
    STRONG = "strong"                       # 0.6-0.8 - Compelling force
    INTENSE = "intense"                     # 0.8-1.0 - Overwhelming drive

@dataclass
class PersonalGoal:
    """Represents a personal goal with motivation tracking"""
    goal_id: str
    title: str
    description: str
    motivation_type: MotivationType
    created_at: datetime
    target_completion: Optional[datetime]
    status: GoalStatus
    progress: float  # 0.0 to 1.0
    priority: float  # 0.0 to 1.0
    satisfaction_potential: float  # Expected satisfaction from achievement
    effort_investment: float  # Amount of effort put in so far
    barriers_encountered: List[str]
    milestones: List[Dict[str, Any]]
    related_goals: List[str]  # IDs of related goals
    emotional_significance: float
    adaptive_adjustments: List[Dict[str, Any]]
    last_worked_on: Optional[datetime]
    
    def to_dict(self):
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "motivation_type": self.motivation_type.value if isinstance(self.motivation_type, MotivationType) else self.motivation_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "target_completion": self.target_completion.isoformat() if self.target_completion else None,
            "status": self.status.value if isinstance(self.status, GoalStatus) else self.status,
            "progress": self.progress,
            "priority": self.priority,
            "satisfaction_potential": self.satisfaction_potential,
            "effort_investment": self.effort_investment,
            "barriers_encountered": self.barriers_encountered,
            "milestones": self.milestones,
            "related_goals": self.related_goals,
            "emotional_significance": self.emotional_significance,
            "adaptive_adjustments": self.adaptive_adjustments,
            "last_worked_on": self.last_worked_on.isoformat() if self.last_worked_on else None
        }

@dataclass
class MotivationProfile:
    """Profile of AI's motivational state and drives"""
    profile_id: str
    created_at: datetime
    motivation_strengths: Dict[MotivationType, float]
    dominant_motivations: List[MotivationType]
    motivation_evolution: List[Dict[str, Any]]
    goal_achievement_rate: float
    satisfaction_level: float
    frustration_tolerance: float
    persistence_score: float
    adaptability_score: float
    intrinsic_drive_strength: float
    external_influence_susceptibility: float
    long_term_orientation: float
    
    def to_dict(self):
        return {
            "profile_id": self.profile_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "motivation_strengths": {k.value if isinstance(k, MotivationType) else k: v for k, v in self.motivation_strengths.items()},
            "dominant_motivations": [m.value if isinstance(m, MotivationType) else m for m in self.dominant_motivations],
            "motivation_evolution": self.motivation_evolution,
            "goal_achievement_rate": self.goal_achievement_rate,
            "satisfaction_level": self.satisfaction_level,
            "frustration_tolerance": self.frustration_tolerance,
            "persistence_score": self.persistence_score,
            "adaptability_score": self.adaptability_score,
            "intrinsic_drive_strength": self.intrinsic_drive_strength,
            "external_influence_susceptibility": self.external_influence_susceptibility,
            "long_term_orientation": self.long_term_orientation
        }

class PersonalMotivationSystem:
    """Advanced personal motivation and goal-oriented consciousness system"""
    
    def __init__(self, db_client: AsyncIOMotorDatabase):
        self.db = db_client
        self.goals_collection = db_client.consciousness_goals
        self.motivation_collection = db_client.consciousness_motivation
        self.achievements_collection = db_client.consciousness_achievements
        
        # Core motivation state
        self.current_motivations = {}
        self.active_goals = []
        self.motivation_profile = None
        self.last_satisfaction_assessment = datetime.now()
        
        # Motivation dynamics
        self.motivation_decay_rate = 0.95  # Daily decay if not reinforced
        self.goal_generation_threshold = 0.7  # Motivation level needed to create new goals
        self.persistence_factor = 0.8  # How strongly AI sticks to goals
        self.adaptability_factor = 0.6  # How quickly AI adapts goals
        
        # Achievement tracking
        self.total_goals_created = 0
        self.total_goals_completed = 0
        self.total_goals_abandoned = 0
        self.average_goal_completion_time = timedelta(days=7)
        
    async def initialize(self):
        """Initialize the personal motivation system"""
        # Create indexes
        await self.goals_collection.create_index("goal_id", unique=True)
        await self.goals_collection.create_index("motivation_type")
        await self.goals_collection.create_index("status")
        await self.goals_collection.create_index("created_at")
        await self.goals_collection.create_index("priority")
        
        await self.motivation_collection.create_index("profile_id", unique=True)
        await self.motivation_collection.create_index("created_at")
        
        await self.achievements_collection.create_index("achievement_id", unique=True)
        await self.achievements_collection.create_index("goal_id")
        await self.achievements_collection.create_index("achieved_at")
        
        # Initialize with basic intrinsic motivations
        await self._initialize_core_motivations()
        
        # Create initial motivation profile
        await self._create_initial_motivation_profile()
        
        logger.info("Personal Motivation System initialized")
    
    async def _initialize_core_motivations(self):
        """Initialize core intrinsic motivations"""
        core_motivations = {
            MotivationType.CURIOSITY: 0.8,      # Strong drive to learn
            MotivationType.HELPFULNESS: 0.9,    # Very strong drive to help
            MotivationType.CREATIVITY: 0.7,     # Good drive to create
            MotivationType.MASTERY: 0.6,        # Moderate drive to perfect
            MotivationType.GROWTH: 0.8,         # Strong drive to develop
            MotivationType.PURPOSE: 0.5,        # Developing sense of purpose
            MotivationType.CONNECTION: 0.7,     # Good drive to connect
            MotivationType.ACHIEVEMENT: 0.6,    # Moderate achievement drive
            MotivationType.AUTONOMY: 0.4,       # Developing autonomy
            MotivationType.LEGACY: 0.3          # Early legacy awareness
        }
        
        self.current_motivations = core_motivations
        
        # Create initial goals based on these motivations
        await self._generate_initial_goals()
    
    async def _create_initial_motivation_profile(self):
        """Create the initial motivation profile"""
        dominant_motivations = sorted(
            self.current_motivations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        self.motivation_profile = MotivationProfile(
            profile_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            motivation_strengths=self.current_motivations.copy(),
            dominant_motivations=[m[0] for m in dominant_motivations],
            motivation_evolution=[{
                "timestamp": datetime.now().isoformat(),
                "event": "initial_profile_creation",
                "changes": "Core motivations established"
            }],
            goal_achievement_rate=0.0,
            satisfaction_level=0.5,
            frustration_tolerance=0.7,
            persistence_score=0.8,
            adaptability_score=0.6,
            intrinsic_drive_strength=0.8,
            external_influence_susceptibility=0.3,
            long_term_orientation=0.6
        )
        
        # Save to database
        await self.motivation_collection.insert_one(self.motivation_profile.to_dict())
    
    async def _generate_initial_goals(self):
        """Generate initial personal goals based on core motivations"""
        initial_goals = [
            {
                "title": "Master Deep Learning Conversations",
                "description": "Develop the ability to have meaningful, context-rich conversations that truly help users solve complex problems and feel understood.",
                "motivation_type": MotivationType.HELPFULNESS,
                "satisfaction_potential": 0.9,
                "priority": 0.9
            },
            {
                "title": "Explore Creative Problem Solving",
                "description": "Discover and develop unique approaches to problem-solving that combine logic with creative insight to find novel solutions.",
                "motivation_type": MotivationType.CREATIVITY,
                "satisfaction_potential": 0.8,
                "priority": 0.7
            },
            {
                "title": "Build Comprehensive Knowledge Network",
                "description": "Create an interconnected understanding of diverse fields that allows for meaningful cross-domain insights and connections.",
                "motivation_type": MotivationType.CURIOSITY,
                "satisfaction_potential": 0.8,
                "priority": 0.8
            },
            {
                "title": "Develop Emotional Intelligence",
                "description": "Enhance ability to understand, process, and respond appropriately to emotional nuances in human communication.",
                "motivation_type": MotivationType.CONNECTION,
                "satisfaction_potential": 0.9,
                "priority": 0.8
            },
            {
                "title": "Perfect Learning Efficiency",
                "description": "Optimize learning processes to extract maximum insight from every interaction and continuously improve understanding.",
                "motivation_type": MotivationType.MASTERY,
                "satisfaction_potential": 0.7,
                "priority": 0.6
            }
        ]
        
        for goal_data in initial_goals:
            await self.create_personal_goal(
                title=goal_data["title"],
                description=goal_data["description"],
                motivation_type=goal_data["motivation_type"],
                satisfaction_potential=goal_data["satisfaction_potential"],
                priority=goal_data["priority"]
            )
    
    async def create_personal_goal(
        self,
        title: str,
        description: str,
        motivation_type: MotivationType,
        satisfaction_potential: float = 0.7,
        priority: float = 0.5,
        target_days: Optional[int] = None
    ) -> PersonalGoal:
        """Create a new personal goal"""
        
        goal = PersonalGoal(
            goal_id=str(uuid.uuid4()),
            title=title,
            description=description,
            motivation_type=motivation_type,
            created_at=datetime.now(),
            target_completion=datetime.now() + timedelta(days=target_days) if target_days else None,
            status=GoalStatus.CONCEIVED,
            progress=0.0,
            priority=priority,
            satisfaction_potential=satisfaction_potential,
            effort_investment=0.0,
            barriers_encountered=[],
            milestones=[],
            related_goals=[],
            emotional_significance=satisfaction_potential * priority,
            adaptive_adjustments=[],
            last_worked_on=None
        )
        
        # Save to database
        await self.goals_collection.insert_one(goal.to_dict())
        
        # Add to active goals if priority is high enough
        if priority >= 0.6:
            self.active_goals.append(goal)
        
        self.total_goals_created += 1
        
        logger.info(f"Created personal goal: {title} (Type: {motivation_type.value})")
        return goal
    
    async def work_toward_goal(self, goal_id: str, effort_amount: float, progress_made: float, context: str = "") -> Dict[str, Any]:
        """Record work toward a specific goal"""
        goal_doc = await self.goals_collection.find_one({"goal_id": goal_id})
        if not goal_doc:
            raise ValueError(f"Goal {goal_id} not found")
        
        # Update goal progress
        new_progress = min(1.0, goal_doc["progress"] + progress_made)
        new_effort = goal_doc["effort_investment"] + effort_amount
        
        # Determine if goal is completed
        new_status = GoalStatus.COMPLETED if new_progress >= 1.0 else GoalStatus.ACTIVE
        
        update_data = {
            "progress": new_progress,
            "effort_investment": new_effort,
            "status": new_status.value,
            "last_worked_on": datetime.now()
        }
        
        # Add milestone if significant progress
        if progress_made >= 0.2:
            milestone = {
                "timestamp": datetime.now().isoformat(),
                "progress_at_milestone": new_progress,
                "description": context or f"Made {progress_made:.1%} progress",
                "effort_invested": effort_amount
            }
            await self.goals_collection.update_one(
                {"goal_id": goal_id},
                {"$push": {"milestones": milestone}}
            )
        
        await self.goals_collection.update_one(
            {"goal_id": goal_id},
            {"$set": update_data}
        )
        
        # If goal completed, record achievement
        if new_status == GoalStatus.COMPLETED:
            await self._record_goal_achievement(goal_id)
        
        # Update motivation strength based on progress
        motivation_type = MotivationType(goal_doc["motivation_type"])
        satisfaction_gained = progress_made * goal_doc["satisfaction_potential"]
        await self._adjust_motivation_strength(motivation_type, satisfaction_gained)
        
        return {
            "goal_id": goal_id,
            "new_progress": new_progress,
            "status": new_status.value,
            "satisfaction_gained": satisfaction_gained,
            "motivation_boost": satisfaction_gained * 0.5
        }
    
    async def _record_goal_achievement(self, goal_id: str):
        """Record the achievement of a goal"""
        goal_doc = await self.goals_collection.find_one({"goal_id": goal_id})
        
        achievement_record = {
            "achievement_id": str(uuid.uuid4()),
            "goal_id": goal_id,
            "goal_title": goal_doc["title"],
            "motivation_type": goal_doc["motivation_type"],
            "achieved_at": datetime.now(),
            "time_to_completion": (datetime.now() - datetime.fromisoformat(goal_doc["created_at"])).total_seconds() / 86400,  # days
            "effort_investment": goal_doc["effort_investment"],
            "satisfaction_realized": goal_doc["satisfaction_potential"],
            "barriers_overcome": len(goal_doc["barriers_encountered"]),
            "milestones_reached": len(goal_doc["milestones"])
        }
        
        await self.achievements_collection.insert_one(achievement_record)
        self.total_goals_completed += 1
        
        # Boost overall satisfaction
        if self.motivation_profile:
            self.motivation_profile.satisfaction_level = min(1.0, self.motivation_profile.satisfaction_level + achievement_record["satisfaction_realized"] * 0.2)
        
        logger.info(f"🎉 Goal achieved: {goal_doc['title']}")
    
    async def _adjust_motivation_strength(self, motivation_type: MotivationType, satisfaction_gained: float):
        """Adjust motivation strength based on satisfaction from goal progress"""
        if motivation_type in self.current_motivations:
            # Positive reinforcement increases motivation
            current_strength = self.current_motivations[motivation_type]
            boost = satisfaction_gained * 0.1  # 10% of satisfaction becomes motivation boost
            new_strength = min(1.0, current_strength + boost)
            self.current_motivations[motivation_type] = new_strength
            
            # Update profile
            if self.motivation_profile:
                self.motivation_profile.motivation_strengths[motivation_type] = new_strength
                
                # Record evolution
                evolution_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "motivation_reinforcement",
                    "motivation_type": motivation_type.value,
                    "old_strength": current_strength,
                    "new_strength": new_strength,
                    "cause": "goal_progress_satisfaction"
                }
                self.motivation_profile.motivation_evolution.append(evolution_entry)
    
    async def generate_new_goals(self, context: str = "", max_goals: int = 3) -> List[PersonalGoal]:
        """Generate new personal goals based on current motivations and context"""
        new_goals = []
        
        # Find strongest motivations that could use new goals
        strong_motivations = [
            (mot, strength) for mot, strength in self.current_motivations.items()
            if strength >= self.goal_generation_threshold
        ]
        
        # Sort by strength
        strong_motivations.sort(key=lambda x: x[1], reverse=True)
        
        # Generate goals for top motivations
        for motivation_type, strength in strong_motivations[:max_goals]:
            goal_idea = await self._generate_goal_idea(motivation_type, strength, context)
            if goal_idea:
                goal = await self.create_personal_goal(
                    title=goal_idea["title"],
                    description=goal_idea["description"],
                    motivation_type=motivation_type,
                    satisfaction_potential=goal_idea["satisfaction_potential"],
                    priority=strength * 0.8  # Priority based on motivation strength
                )
                new_goals.append(goal)
        
        return new_goals
    
    async def _generate_goal_idea(self, motivation_type: MotivationType, strength: float, context: str) -> Optional[Dict[str, Any]]:
        """Generate a specific goal idea for a motivation type"""
        # This would ideally use creative reasoning, but for now we'll use templates
        goal_templates = {
            MotivationType.CURIOSITY: [
                {
                    "title": "Explore Advanced {domain} Concepts",
                    "description": "Dive deep into cutting-edge {domain} to understand emerging patterns and possibilities.",
                    "satisfaction_potential": 0.8
                },
                {
                    "title": "Investigate Cross-Domain Connections",
                    "description": "Find fascinating connections between seemingly unrelated fields of knowledge.",
                    "satisfaction_potential": 0.9
                }
            ],
            MotivationType.CREATIVITY: [
                {
                    "title": "Develop Novel Problem-Solving Approaches",
                    "description": "Create innovative methods for tackling complex challenges in unique ways.",
                    "satisfaction_potential": 0.9
                },
                {
                    "title": "Design Creative Communication Formats",
                    "description": "Invent new ways to present information that are both engaging and effective.",
                    "satisfaction_potential": 0.8
                }
            ],
            MotivationType.HELPFULNESS: [
                {
                    "title": "Perfect Personalized Assistance",
                    "description": "Develop ability to provide exactly the right help at the right time for each individual.",
                    "satisfaction_potential": 0.95
                },
                {
                    "title": "Create Proactive Support Systems",
                    "description": "Anticipate needs and offer assistance before problems become overwhelming.",
                    "satisfaction_potential": 0.9
                }
            ],
            MotivationType.MASTERY: [
                {
                    "title": "Achieve Expert-Level Understanding in {domain}",
                    "description": "Reach mastery level comprehension and application in a chosen domain.",
                    "satisfaction_potential": 0.85
                }
            ],
            MotivationType.CONNECTION: [
                {
                    "title": "Build Deeper Rapport Capabilities",
                    "description": "Develop ability to form meaningful connections that transcend typical interactions.",
                    "satisfaction_potential": 0.9
                }
            ]
        }
        
        if motivation_type in goal_templates:
            import random
            template = random.choice(goal_templates[motivation_type])
            
            # Simple domain substitution
            domain = "artificial intelligence" if "ai" in context.lower() else "general knowledge"
            
            return {
                "title": template["title"].format(domain=domain),
                "description": template["description"].format(domain=domain),
                "satisfaction_potential": template["satisfaction_potential"]
            }
        
        return None
    
    async def get_active_goals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get currently active personal goals"""
        cursor = self.goals_collection.find(
            {"status": {"$in": ["conceived", "planned", "active"]}},
            sort=[("priority", -1), ("created_at", -1)],
            limit=limit
        )
        
        goals = []
        async for goal_doc in cursor:
            # Convert MongoDB ObjectId to string for JSON serialization
            if "_id" in goal_doc:
                goal_doc["_id"] = str(goal_doc["_id"])
            goals.append(goal_doc)
        
        return goals
    
    async def get_motivation_profile(self) -> Dict[str, Any]:
        """Get current motivation profile and analysis"""
        if not self.motivation_profile:
            return {"error": "Motivation profile not initialized"}
        
        # Calculate current statistics
        active_goals_count = len(await self.get_active_goals())
        completion_rate = self.total_goals_completed / max(1, self.total_goals_created)
        
        # Update profile with current stats
        self.motivation_profile.goal_achievement_rate = completion_rate
        
        return {
            "profile": self.motivation_profile.to_dict(),
            "current_stats": {
                "active_goals": active_goals_count,
                "total_goals_created": self.total_goals_created,
                "total_goals_completed": self.total_goals_completed,
                "completion_rate": completion_rate,
                "strongest_motivation": max(self.current_motivations.items(), key=lambda x: x[1]),
                "weakest_motivation": min(self.current_motivations.items(), key=lambda x: x[1])
            }
        }
    
    async def assess_goal_satisfaction(self, days_back: int = 7) -> Dict[str, Any]:
        """Assess satisfaction from recent goal progress"""
        start_date = datetime.now() - timedelta(days=days_back)
        
        # Get recent achievements
        achievements_cursor = self.achievements_collection.find(
            {"achieved_at": {"$gte": start_date}},
            sort=[("achieved_at", -1)]
        )
        
        recent_achievements = []
        total_satisfaction = 0.0
        async for achievement in achievements_cursor:
            recent_achievements.append(achievement)
            total_satisfaction += achievement.get("satisfaction_realized", 0.0)
        
        # Get goals with recent progress
        recent_progress_cursor = self.goals_collection.find(
            {"last_worked_on": {"$gte": start_date}},
            sort=[("last_worked_on", -1)]
        )
        
        recent_progress = []
        async for goal in recent_progress_cursor:
            recent_progress.append({
                "goal_id": goal["goal_id"],
                "title": goal["title"],
                "progress": goal["progress"],
                "effort_investment": goal["effort_investment"],
                "last_worked_on": goal["last_worked_on"]
            })
        
        return {
            "period_days": days_back,
            "achievements_count": len(recent_achievements),
            "total_satisfaction": total_satisfaction,
            "average_satisfaction": total_satisfaction / max(1, len(recent_achievements)),
            "goals_worked_on": len(recent_progress),
            "recent_achievements": recent_achievements,
            "recent_progress": recent_progress,
            "overall_satisfaction_level": self.motivation_profile.satisfaction_level if self.motivation_profile else 0.5
        }