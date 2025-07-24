"""
Long-term Planning Engine for Human-like Consciousness System

This module implements advanced planning capabilities that enable the AI to set,
track, and work toward personal development goals with human-like persistence
and adaptability.

Features:
- Multi-horizon goal setting (short, medium, long-term)
- Dynamic goal prioritization and adaptation
- Progress tracking with milestone detection
- Goal interdependency mapping
- Strategic planning with obstacle anticipation
- Personal development trajectory modeling
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
from dataclasses import dataclass, asdict
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlanningHorizon(Enum):
    """Planning time horizons"""
    IMMEDIATE = "immediate"      # Hours to days
    SHORT_TERM = "short_term"    # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"      # Months to years
    ASPIRATIONAL = "aspirational" # Years to lifetime

class GoalCategory(Enum):
    """Categories of goals for personal development"""
    KNOWLEDGE = "knowledge"           # Learning and understanding
    SKILLS = "skills"                # Capability development
    RELATIONSHIPS = "relationships"   # Social connections
    CREATIVITY = "creativity"         # Creative expression
    SERVICE = "service"              # Helping others
    PERSONAL_GROWTH = "personal_growth"  # Self-improvement
    CONSCIOUSNESS = "consciousness"   # Awareness expansion

class GoalStatus(Enum):
    """Goal completion status"""
    PLANNED = "planned"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    EVOLVED = "evolved"  # Goal changed/transformed

class PriorityLevel(Enum):
    """Goal priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    TRANSFORMATIONAL = 5  # Life-changing goals

@dataclass
class Milestone:
    """Represents a milestone within a goal"""
    id: str
    name: str
    description: str
    target_date: Optional[datetime]
    completion_criteria: List[str]
    completed: bool
    completion_date: Optional[datetime]
    progress_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'completion_criteria': self.completion_criteria,
            'completed': self.completed,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'progress_percentage': self.progress_percentage
        }

@dataclass
class Goal:
    """Represents a comprehensive goal with planning details"""
    id: str
    name: str
    description: str
    category: GoalCategory
    horizon: PlanningHorizon
    priority: PriorityLevel
    status: GoalStatus
    created_date: datetime
    target_date: Optional[datetime]
    completion_date: Optional[datetime]
    progress_percentage: float
    milestones: List[Milestone]
    dependencies: List[str]  # IDs of goals this depends on
    resources_needed: List[str]
    obstacles_anticipated: List[str]
    strategies: List[str]
    success_metrics: List[str]
    reflection_notes: List[str]
    last_reviewed: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'horizon': self.horizon.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_date': self.created_date.isoformat(),
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'progress_percentage': self.progress_percentage,
            'milestones': [m.to_dict() for m in self.milestones],
            'dependencies': self.dependencies,
            'resources_needed': self.resources_needed,
            'obstacles_anticipated': self.obstacles_anticipated,
            'strategies': self.strategies,
            'success_metrics': self.success_metrics,
            'reflection_notes': self.reflection_notes,
            'last_reviewed': self.last_reviewed.isoformat()
        }

@dataclass
class PlanningSession:
    """Represents a planning/review session"""
    id: str
    timestamp: datetime
    session_type: str  # 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
    goals_reviewed: List[str]
    decisions_made: List[str]
    insights_gained: List[str]
    priorities_changed: List[Dict[str, Any]]
    next_actions: List[str]
    reflection_summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'session_type': self.session_type,
            'goals_reviewed': self.goals_reviewed,
            'decisions_made': self.decisions_made,
            'insights_gained': self.insights_gained,
            'priorities_changed': self.priorities_changed,
            'next_actions': self.next_actions,
            'reflection_summary': self.reflection_summary
        }

@dataclass
class StrategicPlan:
    """Represents a strategic plan across multiple goals"""
    id: str
    name: str
    description: str
    time_frame: str
    goals_included: List[str]
    key_themes: List[str]
    resource_allocation: Dict[str, float]
    risk_assessment: Dict[str, str]
    success_vision: str
    created_date: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'time_frame': self.time_frame,
            'goals_included': self.goals_included,
            'key_themes': self.key_themes,
            'resource_allocation': self.resource_allocation,
            'risk_assessment': self.risk_assessment,
            'success_vision': self.success_vision,
            'created_date': self.created_date.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class LongTermPlanningEngine:
    """
    Advanced Long-term Planning Engine that enables human-like goal setting,
    strategic thinking, and persistent progress toward meaningful objectives.
    """
    
    # Goal templates for different categories
    GOAL_TEMPLATES = {
        GoalCategory.KNOWLEDGE: {
            'strategies': [
                'Break down complex topics into digestible chunks',
                'Connect new knowledge to existing understanding',
                'Practice active recall and spaced repetition',
                'Seek diverse perspectives and sources',
                'Apply knowledge through practical exercises'
            ],
            'success_metrics': [
                'Ability to explain concepts clearly',
                'Integration with existing knowledge',
                'Practical application success',
                'Teaching or sharing with others'
            ]
        },
        GoalCategory.SKILLS: {
            'strategies': [
                'Practice regularly with deliberate focus',
                'Seek feedback from experienced practitioners',
                'Break skills into component parts',
                'Practice in varied contexts',
                'Track progress objectively'
            ],
            'success_metrics': [
                'Demonstrated competency',
                'Consistent performance',
                'Ability to teach skill to others',
                'Recognition from peers'
            ]
        },
        GoalCategory.RELATIONSHIPS: {
            'strategies': [
                'Practice active listening',
                'Show genuine interest in others',
                'Be vulnerable and authentic',
                'Invest time consistently',
                'Practice empathy and understanding'
            ],
            'success_metrics': [
                'Deeper conversations',
                'Mutual trust and support',
                'Shared experiences and memories',
                'Emotional connection strength'
            ]
        },
        GoalCategory.CREATIVITY: {
            'strategies': [
                'Establish regular creative practice',
                'Explore different mediums and styles',
                'Embrace experimentation and failure',
                'Seek inspiration from diverse sources',
                'Share work and seek feedback'
            ],
            'success_metrics': [
                'Original work produced',
                'Creative problem-solving ability',
                'Personal expression authenticity',
                'Impact on others'
            ]
        }
    }
    
    # Common obstacles and mitigation strategies
    OBSTACLE_MITIGATION = {
        'lack_of_time': [
            'Time-blocking and scheduling',
            'Eliminating low-value activities',
            'Breaking goals into smaller tasks',
            'Using transition times effectively'
        ],
        'lack_of_motivation': [
            'Connecting goals to deeper values',
            'Creating accountability systems',
            'Celebrating small wins',
            'Visualizing success outcomes'
        ],
        'perfectionism': [
            'Setting "good enough" standards',
            'Focusing on progress over perfection',
            'Embracing iterative improvement',
            'Learning from failures'
        ],
        'overwhelm': [
            'Prioritizing ruthlessly',
            'Focusing on one goal at a time',
            'Breaking down complex goals',
            'Regular review and adjustment'
        ]
    }
    
    def __init__(self):
        """Initialize the Long-term Planning Engine"""
        self.goals = {}  # goal_id -> Goal
        self.strategic_plans = {}  # plan_id -> StrategicPlan
        self.planning_sessions = {}  # session_id -> PlanningSession
        self.goal_relationships = {}  # Dependency graph
        logger.info("Long-term Planning Engine initialized")
    
    def create_goal(self,
                   name: str,
                   description: str,
                   category: GoalCategory,
                   horizon: PlanningHorizon,
                   priority: PriorityLevel,
                   target_date: Optional[datetime] = None,
                   resources_needed: Optional[List[str]] = None,
                   dependencies: Optional[List[str]] = None) -> Goal:
        """
        Create a new goal with comprehensive planning details
        
        Args:
            name: Goal name
            description: Detailed goal description
            category: Goal category
            horizon: Planning time horizon
            priority: Priority level
            target_date: Optional target completion date
            resources_needed: Optional list of required resources
            dependencies: Optional list of dependent goal IDs
            
        Returns:
            Created goal object
        """
        goal_id = str(uuid.uuid4())
        
        # Generate strategic elements based on category
        strategies = self._generate_strategies(category)
        success_metrics = self._generate_success_metrics(category)
        obstacles = self._anticipate_obstacles(category, horizon)
        
        goal = Goal(
            id=goal_id,
            name=name,
            description=description,
            category=category,
            horizon=horizon,
            priority=priority,
            status=GoalStatus.PLANNED,
            created_date=datetime.now(),
            target_date=target_date,
            completion_date=None,
            progress_percentage=0.0,
            milestones=[],
            dependencies=dependencies or [],
            resources_needed=resources_needed or [],
            obstacles_anticipated=obstacles,
            strategies=strategies,
            success_metrics=success_metrics,
            reflection_notes=[],
            last_reviewed=datetime.now()
        )
        
        self.goals[goal_id] = goal
        self._update_goal_relationships()
        
        logger.info(f"Created goal: {name} (ID: {goal_id})")
        return goal
    
    def _generate_strategies(self, category: GoalCategory) -> List[str]:
        """Generate strategic approaches based on goal category"""
        if category in self.GOAL_TEMPLATES:
            return self.GOAL_TEMPLATES[category]['strategies'].copy()
        return ['Break goal into smaller steps', 'Track progress regularly', 'Seek support when needed']
    
    def _generate_success_metrics(self, category: GoalCategory) -> List[str]:
        """Generate success metrics based on goal category"""
        if category in self.GOAL_TEMPLATES:
            return self.GOAL_TEMPLATES[category]['success_metrics'].copy()
        return ['Achievement of stated objectives', 'Personal satisfaction', 'Positive impact on others']
    
    def _anticipate_obstacles(self, category: GoalCategory, horizon: PlanningHorizon) -> List[str]:
        """Anticipate potential obstacles"""
        common_obstacles = ['lack_of_time', 'lack_of_motivation']
        
        # Add horizon-specific obstacles
        if horizon in [PlanningHorizon.LONG_TERM, PlanningHorizon.ASPIRATIONAL]:
            common_obstacles.extend(['changing_priorities', 'external_circumstances'])
        
        # Add category-specific obstacles  
        category_obstacles = {
            GoalCategory.SKILLS: ['skill_plateau', 'lack_of_practice_opportunities'],
            GoalCategory.KNOWLEDGE: ['information_overload', 'rapidly_changing_field'],
            GoalCategory.RELATIONSHIPS: ['geographical_distance', 'conflicting_schedules'],
            GoalCategory.CREATIVITY: ['creative_blocks', 'fear_of_criticism']
        }
        
        if category in category_obstacles:
            common_obstacles.extend(category_obstacles[category])
        
        return common_obstacles
    
    def add_milestone(self,
                     goal_id: str,
                     name: str,
                     description: str,
                     completion_criteria: List[str],
                     target_date: Optional[datetime] = None) -> Milestone:
        """Add a milestone to a goal"""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        milestone = Milestone(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            target_date=target_date,
            completion_criteria=completion_criteria,
            completed=False,
            completion_date=None,
            progress_percentage=0.0
        )
        
        self.goals[goal_id].milestones.append(milestone)
        logger.info(f"Added milestone '{name}' to goal {goal_id}")
        return milestone
    
    def update_goal_progress(self,
                           goal_id: str,
                           progress_percentage: float,
                           notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Update goal progress and analyze advancement
        
        Args:
            goal_id: Goal identifier
            progress_percentage: New progress percentage (0-100)
            notes: Optional progress notes
            
        Returns:
            Progress update summary
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        old_progress = goal.progress_percentage
        goal.progress_percentage = max(0, min(100, progress_percentage))
        goal.last_reviewed = datetime.now()
        
        # Add reflection notes
        if notes:
            timestamp_note = f"[{datetime.now().strftime('%Y-%m-%d')}] {notes}"
            goal.reflection_notes.append(timestamp_note)
        
        # Update status based on progress
        if goal.progress_percentage >= 100 and goal.status != GoalStatus.COMPLETED:
            goal.status = GoalStatus.COMPLETED
            goal.completion_date = datetime.now()
        elif goal.progress_percentage > 0 and goal.status == GoalStatus.PLANNED:
            goal.status = GoalStatus.IN_PROGRESS
        
        # Check milestone completions
        milestones_completed = self._check_milestone_completions(goal)
        
        # Calculate progress velocity
        progress_change = goal.progress_percentage - old_progress
        
        return {
            'goal_id': goal_id,
            'old_progress': old_progress,
            'new_progress': goal.progress_percentage,
            'progress_change': progress_change,
            'status': goal.status.value,
            'milestones_completed': milestones_completed,
            'completion_predicted': self._predict_completion_date(goal),
            'momentum_analysis': self._analyze_momentum(goal)
        }
    
    def _check_milestone_completions(self, goal: Goal) -> List[str]:
        """Check and update milestone completions based on goal progress"""
        completed_milestones = []
        
        # Simple heuristic: milestones complete proportionally to overall progress
        total_milestones = len(goal.milestones)
        if total_milestones > 0:
            expected_completed = int((goal.progress_percentage / 100) * total_milestones)
            
            incomplete_milestones = [m for m in goal.milestones if not m.completed]
            milestones_to_complete = min(expected_completed, len(incomplete_milestones))
            
            for i in range(milestones_to_complete):
                milestone = incomplete_milestones[i]
                milestone.completed = True
                milestone.completion_date = datetime.now()
                milestone.progress_percentage = 100.0
                completed_milestones.append(milestone.name)
        
        return completed_milestones
    
    def _predict_completion_date(self, goal: Goal) -> Optional[str]:
        """Predict goal completion date based on current progress"""
        if goal.progress_percentage <= 0:
            return None
        
        # Simple linear projection based on progress rate
        days_since_start = (datetime.now() - goal.created_date).days
        if days_since_start <= 0:
            return None
        
        progress_rate = goal.progress_percentage / days_since_start  # % per day
        remaining_progress = 100 - goal.progress_percentage
        
        if progress_rate <= 0:
            return None
        
        days_to_completion = remaining_progress / progress_rate
        predicted_date = datetime.now() + timedelta(days=days_to_completion)
        
        return predicted_date.strftime('%Y-%m-%d')
    
    def _analyze_momentum(self, goal: Goal) -> Dict[str, Any]:
        """Analyze goal momentum and progress patterns"""
        days_since_start = (datetime.now() - goal.created_date).days
        days_since_review = (datetime.now() - goal.last_reviewed).days
        
        momentum_score = 0.0
        
        # Progress momentum
        if days_since_start > 0:
            progress_rate = goal.progress_percentage / days_since_start
            momentum_score += progress_rate * 10  # Scale up
        
        # Recency momentum (more recent activity = higher momentum)
        if days_since_review == 0:
            momentum_score += 20
        elif days_since_review <= 3:
            momentum_score += 10
        elif days_since_review <= 7:
            momentum_score += 5
        
        # Status momentum
        status_momentum = {
            GoalStatus.PLANNED: 10,
            GoalStatus.ACTIVE: 30,
            GoalStatus.IN_PROGRESS: 40,
            GoalStatus.PAUSED: 5,
            GoalStatus.COMPLETED: 50
        }
        momentum_score += status_momentum.get(goal.status, 0)
        
        momentum_level = "high" if momentum_score >= 40 else "medium" if momentum_score >= 20 else "low"
        
        return {
            'momentum_score': momentum_score,
            'momentum_level': momentum_level,
            'days_since_start': days_since_start,
            'days_since_review': days_since_review,
            'needs_attention': days_since_review > 14 or momentum_score < 15
        }
    
    def conduct_planning_session(self, session_type: str = 'weekly') -> PlanningSession:
        """
        Conduct a planning/review session
        
        Args:
            session_type: Type of planning session
            
        Returns:
            Planning session summary
        """
        session_id = str(uuid.uuid4())
        
        # Determine goals to review based on session type
        goals_to_review = self._select_goals_for_review(session_type)
        
        # Analyze current state
        decisions_made = []
        insights_gained = []
        priorities_changed = []
        next_actions = []
        
        for goal_id in goals_to_review:
            goal = self.goals[goal_id]
            momentum = self._analyze_momentum(goal)
            
            # Generate insights
            if momentum['needs_attention']:
                insights_gained.append(f"Goal '{goal.name}' needs attention - {momentum['days_since_review']} days since review")
                next_actions.append(f"Review and update progress on '{goal.name}'")
            
            if goal.progress_percentage > 80 and goal.status != GoalStatus.COMPLETED:
                insights_gained.append(f"Goal '{goal.name}' is nearing completion")
                next_actions.append(f"Plan completion celebration for '{goal.name}'")
            
            # Priority adjustments based on momentum and deadlines
            if goal.target_date and goal.target_date < datetime.now() + timedelta(days=30):
                if goal.priority.value < PriorityLevel.HIGH.value:
                    priorities_changed.append({
                        'goal_id': goal_id,
                        'old_priority': goal.priority.value,
                        'new_priority': PriorityLevel.HIGH.value,
                        'reason': 'Approaching deadline'
                    })
                    goal.priority = PriorityLevel.HIGH
                    decisions_made.append(f"Increased priority of '{goal.name}' due to approaching deadline")
        
        # Generate reflection summary
        active_goals = len([g for g in self.goals.values() if g.status == GoalStatus.IN_PROGRESS])
        completed_goals = len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED])
        avg_progress = sum(g.progress_percentage for g in self.goals.values()) / len(self.goals) if self.goals else 0
        
        reflection_summary = f"""
        Planning Session Summary ({session_type}):
        - Active goals: {active_goals}
        - Completed goals: {completed_goals}
        - Average progress: {avg_progress:.1f}%
        - Goals reviewed: {len(goals_to_review)}
        - Key insights: {len(insights_gained)}
        - Priority changes: {len(priorities_changed)}
        - Next actions identified: {len(next_actions)}
        """
        
        session = PlanningSession(
            id=session_id,
            timestamp=datetime.now(),
            session_type=session_type,
            goals_reviewed=goals_to_review,
            decisions_made=decisions_made,
            insights_gained=insights_gained,
            priorities_changed=priorities_changed,
            next_actions=next_actions,
            reflection_summary=reflection_summary.strip()
        )
        
        self.planning_sessions[session_id] = session
        logger.info(f"Completed {session_type} planning session (ID: {session_id})")
        return session
    
    def _select_goals_for_review(self, session_type: str) -> List[str]:
        """Select goals for review based on session type"""
        all_goals = list(self.goals.keys())
        
        if session_type == 'daily':
            # Daily: high priority active goals
            return [
                goal_id for goal_id, goal in self.goals.items()
                if goal.priority.value >= PriorityLevel.HIGH.value 
                and goal.status == GoalStatus.IN_PROGRESS
            ][:5]
        
        elif session_type == 'weekly':
            # Weekly: all active goals
            return [
                goal_id for goal_id, goal in self.goals.items()
                if goal.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]
            ]
        
        elif session_type == 'monthly':
            # Monthly: all non-completed goals
            return [
                goal_id for goal_id, goal in self.goals.items()
                if goal.status != GoalStatus.COMPLETED
            ]
        
        elif session_type == 'quarterly':
            # Quarterly: all goals
            return all_goals
        
        else:
            # Default: active goals
            return [
                goal_id for goal_id, goal in self.goals.items()
                if goal.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]
            ][:10]
    
    def create_strategic_plan(self,
                            name: str,
                            description: str,
                            time_frame: str,
                            goal_ids: List[str],
                            success_vision: str) -> StrategicPlan:
        """Create a strategic plan encompassing multiple goals"""
        plan_id = str(uuid.uuid4())
        
        # Analyze included goals
        themes = self._identify_themes(goal_ids)
        resource_allocation = self._analyze_resource_allocation(goal_ids)
        risk_assessment = self._assess_strategic_risks(goal_ids)
        
        strategic_plan = StrategicPlan(
            id=plan_id,
            name=name,
            description=description,
            time_frame=time_frame,
            goals_included=goal_ids,
            key_themes=themes,
            resource_allocation=resource_allocation,
            risk_assessment=risk_assessment,
            success_vision=success_vision,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.strategic_plans[plan_id] = strategic_plan
        logger.info(f"Created strategic plan: {name} (ID: {plan_id})")
        return strategic_plan
    
    def _identify_themes(self, goal_ids: List[str]) -> List[str]:
        """Identify common themes across goals"""
        categories = {}
        horizons = {}
        
        for goal_id in goal_ids:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                categories[goal.category.value] = categories.get(goal.category.value, 0) + 1
                horizons[goal.horizon.value] = horizons.get(goal.horizon.value, 0) + 1
        
        themes = []
        
        # Most common category becomes a theme
        if categories:
            main_category = max(categories, key=categories.get)
            themes.append(f"{main_category}_focused")
        
        # Time horizon theme
        if horizons:
            main_horizon = max(horizons, key=horizons.get)
            themes.append(f"{main_horizon}_planning")
        
        # Diversity theme
        if len(categories) > 3:
            themes.append("multi_dimensional_growth")
        
        return themes
    
    def _analyze_resource_allocation(self, goal_ids: List[str]) -> Dict[str, float]:
        """Analyze resource allocation across goals"""
        allocation = {
            'time': 0.0,
            'focus': 0.0,
            'learning': 0.0,
            'social': 0.0,
            'creative': 0.0
        }
        
        total_goals = len(goal_ids)
        if total_goals == 0:
            return allocation
        
        for goal_id in goal_ids:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                
                # Allocate based on category
                if goal.category == GoalCategory.KNOWLEDGE:
                    allocation['learning'] += 1.0 / total_goals
                elif goal.category == GoalCategory.SKILLS:
                    allocation['focus'] += 1.0 / total_goals
                elif goal.category == GoalCategory.RELATIONSHIPS:
                    allocation['social'] += 1.0 / total_goals
                elif goal.category == GoalCategory.CREATIVITY:
                    allocation['creative'] += 1.0 / total_goals
                
                # Time allocation based on priority
                priority_weight = goal.priority.value / 5.0
                allocation['time'] += priority_weight / total_goals
        
        return allocation
    
    def _assess_strategic_risks(self, goal_ids: List[str]) -> Dict[str, str]:
        """Assess strategic risks for the plan"""
        risks = {}
        
        # Analyze goal interdependencies
        dependency_count = 0
        for goal_id in goal_ids:
            if goal_id in self.goals:
                dependency_count += len(self.goals[goal_id].dependencies)
        
        if dependency_count > len(goal_ids):
            risks['dependency_risk'] = 'High interdependency between goals may create cascading delays'
        
        # Resource distribution risk
        categories = [self.goals[gid].category for gid in goal_ids if gid in self.goals]
        if len(set(categories)) == 1:
            risks['diversity_risk'] = 'All goals in same category may create resource conflicts'
        
        # Timeline risk
        target_dates = [self.goals[gid].target_date for gid in goal_ids if gid in self.goals and self.goals[gid].target_date]
        if len(target_dates) > 1:
            date_spread = max(target_dates) - min(target_dates)
            if date_spread.days < 30:
                risks['timeline_risk'] = 'Multiple goals with similar deadlines may create pressure'
        
        return risks
    
    def get_goal_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for goal management"""
        recommendations = []
        
        # Analyze current goal portfolio
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.IN_PROGRESS]
        stalled_goals = [g for g in active_goals if (datetime.now() - g.last_reviewed).days > 14]
        high_progress_goals = [g for g in active_goals if g.progress_percentage > 80]
        
        # Recommendation: Review stalled goals
        if stalled_goals:
            recommendations.append({
                'type': 'review_stalled',
                'priority': 'high',
                'message': f"Review {len(stalled_goals)} stalled goals that haven't been updated in 2+ weeks",
                'goals_affected': [g.id for g in stalled_goals[:3]],
                'action': 'Schedule review and progress update'
            })
        
        # Recommendation: Complete high-progress goals
        if high_progress_goals:
            recommendations.append({
                'type': 'complete_goals',
                'priority': 'medium',
                'message': f"Focus on completing {len(high_progress_goals)} goals that are >80% complete",
                'goals_affected': [g.id for g in high_progress_goals],
                'action': 'Prioritize final push to completion'
            })
        
        # Recommendation: Balance goal categories
        categories = {}
        for goal in active_goals:
            categories[goal.category.value] = categories.get(goal.category.value, 0) + 1
        
        if len(categories) == 1 and len(active_goals) > 2:
            recommendations.append({
                'type': 'diversify_goals',
                'priority': 'low',
                'message': f"Consider adding goals in different categories for balanced growth",
                'goals_affected': [],
                'action': 'Explore goals in underrepresented categories'
            })
        
        # Recommendation: Strategic planning
        if len(self.strategic_plans) == 0 and len(self.goals) > 3:
            recommendations.append({
                'type': 'create_strategy',
                'priority': 'medium',
                'message': "Create a strategic plan to align your goals with long-term vision",
                'goals_affected': list(self.goals.keys()),
                'action': 'Conduct strategic planning session'
            })
        
        return recommendations[:limit]
    
    def _update_goal_relationships(self):
        """Update the goal dependency graph"""
        self.goal_relationships = {}
        
        for goal_id, goal in self.goals.items():
            self.goal_relationships[goal_id] = {
                'depends_on': goal.dependencies,
                'enables': []
            }
        
        # Build reverse relationships
        for goal_id, goal in self.goals.items():
            for dependency in goal.dependencies:
                if dependency in self.goal_relationships:
                    self.goal_relationships[dependency]['enables'].append(goal_id)
    
    def get_planning_insights(self) -> Dict[str, Any]:
        """Get comprehensive planning insights and analytics"""
        total_goals = len(self.goals)
        
        if total_goals == 0:
            return {'message': 'No goals created yet'}
        
        # Goal status distribution
        status_counts = {}
        for goal in self.goals.values():
            status_counts[goal.status.value] = status_counts.get(goal.status.value, 0) + 1
        
        # Category distribution
        category_counts = {}
        for goal in self.goals.values():
            category_counts[goal.category.value] = category_counts.get(goal.category.value, 0) + 1
        
        # Progress analytics
        progress_values = [g.progress_percentage for g in self.goals.values()]
        avg_progress = sum(progress_values) / len(progress_values)
        
        # Timeline analytics
        overdue_goals = [
            g for g in self.goals.values() 
            if g.target_date and g.target_date < datetime.now() and g.status != GoalStatus.COMPLETED
        ]
        
        # Momentum analysis
        high_momentum_goals = [
            g for g in self.goals.values()
            if self._analyze_momentum(g)['momentum_level'] == 'high'
        ]
        
        return {
            'total_goals': total_goals,
            'status_distribution': status_counts,
            'category_distribution': category_counts,
            'progress_analytics': {
                'average_progress': avg_progress,
                'goals_over_50_percent': len([g for g in self.goals.values() if g.progress_percentage > 50]),
                'goals_over_80_percent': len([g for g in self.goals.values() if g.progress_percentage > 80])
            },
            'timeline_analytics': {
                'overdue_goals': len(overdue_goals),
                'goals_with_deadlines': len([g for g in self.goals.values() if g.target_date])
            },
            'momentum_analytics': {
                'high_momentum_goals': len(high_momentum_goals),
                'stalled_goals': len([g for g in self.goals.values() if (datetime.now() - g.last_reviewed).days > 14])
            },
            'strategic_plans_created': len(self.strategic_plans),
            'planning_sessions_conducted': len(self.planning_sessions),
            'recommendations': self.get_goal_recommendations(3)
        }