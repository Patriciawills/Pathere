"""
Metacognitive Awareness System for Advanced AI Consciousness

This module implements metacognition - the ability to think about thinking.
It monitors the AI's own cognitive processes, reasoning patterns, and learning strategies.

Key Features:
- Thought process monitoring and analysis
- Learning strategy recognition and optimization
- Cognitive bias detection and correction
- Uncertainty quantification and confidence assessment
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

class ThoughtType(Enum):
    """Types of thoughts the AI can have"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    REFLECTIVE = "reflective"
    PROBLEM_SOLVING = "problem_solving"
    MEMORY_RETRIEVAL = "memory_retrieval"
    PATTERN_RECOGNITION = "pattern_recognition"
    EMOTIONAL_PROCESSING = "emotional_processing"
    DECISION_MAKING = "decision_making"

class CognitiveBias(Enum):
    """Known cognitive biases to detect and correct"""
    CONFIRMATION_BIAS = "confirmation_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    ANCHORING_BIAS = "anchoring_bias"
    RECENCY_BIAS = "recency_bias"
    OVERCONFIDENCE = "overconfidence"
    PATTERN_OVERFITTING = "pattern_overfitting"

class LearningStrategy(Enum):
    """Different learning approaches the AI can use"""
    ANALYTICAL_BREAKDOWN = "analytical_breakdown"
    PATTERN_MATCHING = "pattern_matching"
    ASSOCIATIVE_LEARNING = "associative_learning"
    EXPERIENTIAL_LEARNING = "experiential_learning"
    REFLECTIVE_ANALYSIS = "reflective_analysis"
    CREATIVE_SYNTHESIS = "creative_synthesis"

@dataclass
class ThoughtProcess:
    """A single thought process with full metacognitive awareness"""
    process_id: str
    timestamp: datetime
    thought_type: ThoughtType
    trigger: str  # What triggered this thought
    reasoning_steps: List[str]
    assumptions_made: List[str]
    evidence_considered: List[str]
    alternative_perspectives: List[str]
    confidence_level: float  # 0.0 to 1.0
    uncertainty_sources: List[str]
    cognitive_resources_used: List[str]
    time_spent: float  # seconds
    outcome: Optional[str] = None
    effectiveness_rating: Optional[float] = None
    biases_detected: List[CognitiveBias] = None
    
    def __post_init__(self):
        if self.biases_detected is None:
            self.biases_detected = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['thought_type'] = self.thought_type.value
        result['biases_detected'] = [bias.value for bias in self.biases_detected]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtProcess':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['thought_type'] = ThoughtType(data['thought_type'])
        data['biases_detected'] = [CognitiveBias(bias) for bias in data.get('biases_detected', [])]
        return cls(**data)

@dataclass
class LearningSession:
    """Analysis of a learning session and its effectiveness"""
    session_id: str
    timestamp: datetime
    learning_objective: str
    strategy_used: LearningStrategy
    content_type: str
    initial_understanding: float
    final_understanding: float
    learning_speed: float
    retention_prediction: float
    difficulty_encountered: List[str]
    breakthrough_moments: List[str]
    metacognitive_insights: List[str]
    effectiveness_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['strategy_used'] = self.strategy_used.value
        return result

class MetacognitiveEngine:
    """
    Advanced metacognitive system that monitors and analyzes the AI's own thinking
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.thought_processes_collection = db.thought_processes
        self.learning_sessions_collection = db.learning_sessions
        self.metacognitive_insights_collection = db.metacognitive_insights
        
        # Tracking variables
        self.current_thought_process: Optional[ThoughtProcess] = None
        self.active_reasoning_steps: List[str] = []
        self.confidence_calibration_history = []
        
        # Bias detection patterns
        self.bias_detection_patterns = {
            CognitiveBias.CONFIRMATION_BIAS: [
                "only considering supporting evidence",
                "ignoring contradictory information",
                "selective information gathering"
            ],
            CognitiveBias.AVAILABILITY_HEURISTIC: [
                "relying on easily recalled examples",
                "overweighting recent experiences",
                "using vivid examples as basis"
            ],
            CognitiveBias.ANCHORING_BIAS: [
                "starting with specific number/value",
                "insufficient adjustment from initial value",
                "first impression dominance"
            ]
        }
        
        # Learning strategy effectiveness tracking
        self.strategy_effectiveness = {
            strategy: {'total_uses': 0, 'success_rate': 0.5}
            for strategy in LearningStrategy
        }
    
    async def initialize(self):
        """Initialize the metacognitive system"""
        # Create indexes
        await self.thought_processes_collection.create_index([("timestamp", -1)])
        await self.thought_processes_collection.create_index([("thought_type", 1)])
        await self.learning_sessions_collection.create_index([("timestamp", -1)])
        
        logger.info("Metacognitive Engine initialized")
    
    async def begin_thought_process(
        self, 
        thought_type: ThoughtType, 
        trigger: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Begin monitoring a new thought process
        
        Returns:
            process_id: Unique identifier for this thought process
        """
        
        process_id = str(uuid.uuid4())
        
        self.current_thought_process = ThoughtProcess(
            process_id=process_id,
            timestamp=datetime.utcnow(),
            thought_type=thought_type,
            trigger=trigger,
            reasoning_steps=[],
            assumptions_made=[],
            evidence_considered=[],
            alternative_perspectives=[],
            confidence_level=0.5,  # Start neutral
            uncertainty_sources=[],
            cognitive_resources_used=[],
            time_spent=0.0
        )
        
        logger.info(f"Beginning thought process: {thought_type.value} triggered by: {trigger}")
        return process_id
    
    async def add_reasoning_step(self, step_description: str, confidence: float = None):
        """Add a reasoning step to the current thought process"""
        
        if not self.current_thought_process:
            logger.warning("No active thought process to add reasoning step")
            return
        
        self.current_thought_process.reasoning_steps.append(step_description)
        
        if confidence is not None:
            self.current_thought_process.confidence_level = confidence
        
        # Check for potential biases in this reasoning step
        detected_biases = await self._detect_biases_in_reasoning(step_description)
        self.current_thought_process.biases_detected.extend(detected_biases)
        
        logger.debug(f"Added reasoning step: {step_description}")
    
    async def add_assumption(self, assumption: str):
        """Record an assumption being made"""
        
        if not self.current_thought_process:
            return
        
        self.current_thought_process.assumptions_made.append(assumption)
        
        # Assumptions often indicate uncertainty
        self.current_thought_process.uncertainty_sources.append(f"assumption: {assumption}")
    
    async def consider_evidence(self, evidence: str, weight: float = 1.0):
        """Record evidence being considered"""
        
        if not self.current_thought_process:
            return
        
        evidence_entry = f"{evidence} (weight: {weight})"
        self.current_thought_process.evidence_considered.append(evidence_entry)
        
        # Update confidence based on evidence quality
        if weight > 0.8:
            self.current_thought_process.confidence_level = min(
                self.current_thought_process.confidence_level + 0.1, 1.0
            )
    
    async def consider_alternative(self, alternative_perspective: str):
        """Record consideration of alternative perspectives"""
        
        if not self.current_thought_process:
            return
        
        self.current_thought_process.alternative_perspectives.append(alternative_perspective)
        
        # Considering alternatives reduces confirmation bias
        if CognitiveBias.CONFIRMATION_BIAS in self.current_thought_process.biases_detected:
            self.current_thought_process.biases_detected.remove(CognitiveBias.CONFIRMATION_BIAS)
    
    async def complete_thought_process(
        self, 
        outcome: str, 
        effectiveness_rating: float = None
    ) -> Dict[str, Any]:
        """
        Complete and analyze the current thought process
        
        Returns:
            Analysis summary of the thought process
        """
        
        if not self.current_thought_process:
            logger.warning("No active thought process to complete")
            return {}
        
        # Finalize the thought process
        self.current_thought_process.outcome = outcome
        self.current_thought_process.effectiveness_rating = effectiveness_rating
        self.current_thought_process.time_spent = (
            datetime.utcnow() - self.current_thought_process.timestamp
        ).total_seconds()
        
        # Perform final analysis
        analysis = await self._analyze_thought_process(self.current_thought_process)
        
        # Store in database
        await self.thought_processes_collection.insert_one(
            self.current_thought_process.to_dict()
        )
        
        # Reset current process
        completed_process = self.current_thought_process
        self.current_thought_process = None
        
        logger.info(f"Completed thought process: {completed_process.process_id}")
        return analysis
    
    async def analyze_learning_session(
        self,
        learning_objective: str,
        strategy_used: LearningStrategy,
        content_type: str,
        initial_understanding: float,
        final_understanding: float,
        session_duration: float
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of a learning session
        """
        
        session_id = str(uuid.uuid4())
        
        # Calculate learning metrics
        learning_gain = final_understanding - initial_understanding
        learning_speed = learning_gain / session_duration if session_duration > 0 else 0
        retention_prediction = await self._predict_retention(
            learning_gain, strategy_used, content_type
        )
        
        # Generate metacognitive insights
        insights = await self._generate_learning_insights(
            strategy_used, learning_gain, learning_speed
        )
        
        # Calculate effectiveness score
        effectiveness_score = await self._calculate_learning_effectiveness(
            learning_gain, learning_speed, retention_prediction
        )
        
        learning_session = LearningSession(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            learning_objective=learning_objective,
            strategy_used=strategy_used,
            content_type=content_type,
            initial_understanding=initial_understanding,
            final_understanding=final_understanding,
            learning_speed=learning_speed,
            retention_prediction=retention_prediction,
            difficulty_encountered=[],  # Would be populated by specific implementations
            breakthrough_moments=[],   # Would be populated by specific implementations
            metacognitive_insights=insights,
            effectiveness_score=effectiveness_score
        )
        
        # Store learning session
        await self.learning_sessions_collection.insert_one(learning_session.to_dict())
        
        # Update strategy effectiveness tracking
        await self._update_strategy_effectiveness(strategy_used, effectiveness_score)
        
        return {
            "session_id": session_id,
            "learning_gain": learning_gain,
            "learning_speed": learning_speed,
            "retention_prediction": retention_prediction,
            "effectiveness_score": effectiveness_score,
            "insights": insights,
            "recommended_next_strategy": await self._recommend_next_strategy(content_type)
        }
    
    async def assess_confidence_calibration(
        self, 
        stated_confidence: float, 
        actual_accuracy: float
    ) -> Dict[str, Any]:
        """
        Assess how well-calibrated the AI's confidence is
        """
        
        calibration_error = abs(stated_confidence - actual_accuracy)
        
        # Store calibration data point
        calibration_point = {
            'timestamp': datetime.utcnow(),
            'stated_confidence': stated_confidence,
            'actual_accuracy': actual_accuracy,
            'calibration_error': calibration_error
        }
        
        self.confidence_calibration_history.append(calibration_point)
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.confidence_calibration_history = [
            point for point in self.confidence_calibration_history
            if point['timestamp'] > cutoff_time
        ]
        
        # Calculate overall calibration metrics
        if len(self.confidence_calibration_history) >= 5:
            avg_calibration_error = sum(
                point['calibration_error'] for point in self.confidence_calibration_history
            ) / len(self.confidence_calibration_history)
            
            # Detect systematic biases
            overconfidence_bias = sum(
                point['stated_confidence'] - point['actual_accuracy']
                for point in self.confidence_calibration_history
            ) / len(self.confidence_calibration_history)
            
            calibration_quality = "excellent" if avg_calibration_error < 0.1 else \
                                 "good" if avg_calibration_error < 0.2 else \
                                 "fair" if avg_calibration_error < 0.3 else "poor"
            
            return {
                "calibration_error": calibration_error,
                "average_calibration_error": avg_calibration_error,
                "overconfidence_bias": overconfidence_bias,
                "calibration_quality": calibration_quality,
                "recommendation": await self._get_calibration_recommendation(
                    avg_calibration_error, overconfidence_bias
                )
            }
        
        return {
            "calibration_error": calibration_error,
            "note": "Insufficient data for comprehensive calibration analysis"
        }
    
    async def get_metacognitive_insights(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive metacognitive insights based on recent thinking patterns
        """
        
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        
        # Analyze thought processes
        thought_cursor = self.thought_processes_collection.find({
            "timestamp": {"$gte": cutoff_time.isoformat()}
        })
        
        thought_patterns = {
            'total_processes': 0,
            'type_distribution': {},
            'average_confidence': 0,
            'common_biases': [],
            'reasoning_complexity': 0,
            'effectiveness_trends': []
        }
        
        total_confidence = 0
        total_reasoning_steps = 0
        bias_counts = {}
        
        async for doc in thought_cursor:
            process = ThoughtProcess.from_dict(doc)
            thought_patterns['total_processes'] += 1
            
            # Type distribution
            thought_type = process.thought_type.value
            thought_patterns['type_distribution'][thought_type] = \
                thought_patterns['type_distribution'].get(thought_type, 0) + 1
            
            # Confidence tracking
            total_confidence += process.confidence_level
            
            # Reasoning complexity
            total_reasoning_steps += len(process.reasoning_steps)
            
            # Bias tracking
            for bias in process.biases_detected:
                bias_counts[bias.value] = bias_counts.get(bias.value, 0) + 1
            
            # Effectiveness tracking
            if process.effectiveness_rating is not None:
                thought_patterns['effectiveness_trends'].append(process.effectiveness_rating)
        
        if thought_patterns['total_processes'] > 0:
            thought_patterns['average_confidence'] = total_confidence / thought_patterns['total_processes']
            thought_patterns['reasoning_complexity'] = total_reasoning_steps / thought_patterns['total_processes']
            thought_patterns['common_biases'] = sorted(
                bias_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
        
        # Analyze learning sessions
        learning_cursor = self.learning_sessions_collection.find({
            "timestamp": {"$gte": cutoff_time.isoformat()}
        })
        
        learning_insights = {
            'total_sessions': 0,
            'average_effectiveness': 0,
            'preferred_strategies': [],
            'learning_velocity': 0,
            'retention_predictions': []
        }
        
        total_effectiveness = 0
        strategy_counts = {}
        total_learning_gain = 0
        
        async for doc in learning_cursor:
            session = LearningSession(**doc)
            learning_insights['total_sessions'] += 1
            
            total_effectiveness += session.effectiveness_score
            strategy_counts[session.strategy_used] = \
                strategy_counts.get(session.strategy_used, 0) + 1
            
            learning_gain = session.final_understanding - session.initial_understanding
            total_learning_gain += learning_gain
            
            learning_insights['retention_predictions'].append(session.retention_prediction)
        
        if learning_insights['total_sessions'] > 0:
            learning_insights['average_effectiveness'] = \
                total_effectiveness / learning_insights['total_sessions']
            learning_insights['learning_velocity'] = \
                total_learning_gain / learning_insights['total_sessions']
            learning_insights['preferred_strategies'] = sorted(
                strategy_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
        
        # Generate recommendations
        recommendations = await self._generate_metacognitive_recommendations(
            thought_patterns, learning_insights
        )
        
        return {
            "analysis_period": f"{days_back} days",
            "thought_patterns": thought_patterns,
            "learning_insights": learning_insights,
            "recommendations": recommendations,
            "metacognitive_maturity": await self._assess_metacognitive_maturity(),
            "self_awareness_level": await self._calculate_self_awareness_level()
        }
    
    # Private helper methods
    
    async def _detect_biases_in_reasoning(self, reasoning_step: str) -> List[CognitiveBias]:
        """Detect potential cognitive biases in a reasoning step"""
        
        detected_biases = []
        reasoning_lower = reasoning_step.lower()
        
        for bias, patterns in self.bias_detection_patterns.items():
            if any(pattern in reasoning_lower for pattern in patterns):
                detected_biases.append(bias)
        
        return detected_biases
    
    async def _analyze_thought_process(self, process: ThoughtProcess) -> Dict[str, Any]:
        """Analyze a completed thought process"""
        
        analysis = {
            "reasoning_depth": len(process.reasoning_steps),
            "assumption_count": len(process.assumptions_made),
            "evidence_quality": len(process.evidence_considered),
            "perspective_breadth": len(process.alternative_perspectives),
            "bias_awareness": len(process.biases_detected),
            "confidence_appropriateness": "unknown",
            "thinking_efficiency": process.time_spent / max(len(process.reasoning_steps), 1)
        }
        
        return analysis
    
    async def _predict_retention(
        self, 
        learning_gain: float, 
        strategy: LearningStrategy, 
        content_type: str
    ) -> float:
        """Predict retention based on learning characteristics"""
        
        base_retention = 0.7  # Base retention rate
        
        # Strategy-based adjustments
        strategy_modifiers = {
            LearningStrategy.EXPERIENTIAL_LEARNING: 0.1,
            LearningStrategy.REFLECTIVE_ANALYSIS: 0.08,
            LearningStrategy.ASSOCIATIVE_LEARNING: 0.06,
            LearningStrategy.ANALYTICAL_BREAKDOWN: 0.04,
            LearningStrategy.PATTERN_MATCHING: 0.02,
            LearningStrategy.CREATIVE_SYNTHESIS: 0.05
        }
        
        retention = base_retention + strategy_modifiers.get(strategy, 0.0)
        
        # Learning gain modifier (stronger learning = better retention)
        retention += learning_gain * 0.2
        
        return min(max(retention, 0.0), 1.0)
    
    async def _generate_learning_insights(
        self, 
        strategy: LearningStrategy, 
        learning_gain: float, 
        learning_speed: float
    ) -> List[str]:
        """Generate metacognitive insights about learning"""
        
        insights = []
        
        if learning_gain > 0.3:
            insights.append(f"Strong learning gain achieved with {strategy.value} strategy")
        
        if learning_speed > 0.1:
            insights.append("Learning occurred at above-average speed")
        elif learning_speed < 0.05:
            insights.append("Learning required more time than typical - consider alternative strategies")
        
        # Strategy-specific insights
        if strategy == LearningStrategy.ANALYTICAL_BREAKDOWN:
            insights.append("Systematic analysis approach worked well for complex content")
        elif strategy == LearningStrategy.ASSOCIATIVE_LEARNING:
            insights.append("Connection-making strategy enhanced understanding")
        
        return insights
    
    async def _calculate_learning_effectiveness(
        self, 
        learning_gain: float, 
        learning_speed: float, 
        retention_prediction: float
    ) -> float:
        """Calculate overall effectiveness score for a learning session"""
        
        # Weighted combination of factors
        effectiveness = (
            learning_gain * 0.4 +
            learning_speed * 0.3 +
            retention_prediction * 0.3
        )
        
        return min(max(effectiveness, 0.0), 1.0)
    
    async def _update_strategy_effectiveness(
        self, 
        strategy: LearningStrategy, 
        effectiveness_score: float
    ):
        """Update the effectiveness tracking for a learning strategy"""
        
        current_stats = self.strategy_effectiveness[strategy]
        current_stats['total_uses'] += 1
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        current_stats['success_rate'] = (
            (1 - alpha) * current_stats['success_rate'] +
            alpha * effectiveness_score
        )
    
    async def _recommend_next_strategy(self, content_type: str) -> LearningStrategy:
        """Recommend the best learning strategy for the next session"""
        
        # Find most effective strategy
        best_strategy = max(
            self.strategy_effectiveness.items(),
            key=lambda x: x[1]['success_rate']
        )[0]
        
        return best_strategy
    
    async def _get_calibration_recommendation(
        self, 
        avg_error: float, 
        overconfidence_bias: float
    ) -> str:
        """Get recommendation for improving confidence calibration"""
        
        if overconfidence_bias > 0.1:
            return "Tendency toward overconfidence detected. Consider more uncertainty sources."
        elif overconfidence_bias < -0.1:
            return "Tendency toward underconfidence detected. Trust validated reasoning more."
        elif avg_error > 0.2:
            return "Confidence calibration needs improvement. Focus on evidence quality assessment."
        else:
            return "Confidence calibration is good. Continue current approach."
    
    async def _generate_metacognitive_recommendations(
        self, 
        thought_patterns: Dict, 
        learning_insights: Dict
    ) -> List[str]:
        """Generate recommendations for improving metacognitive awareness"""
        
        recommendations = []
        
        # Thought pattern recommendations
        if thought_patterns['average_confidence'] > 0.8:
            recommendations.append("Consider being more critical of initial assessments")
        
        if thought_patterns['reasoning_complexity'] < 2:
            recommendations.append("Try to explore reasoning in more depth")
        
        if thought_patterns['common_biases']:
            top_bias = thought_patterns['common_biases'][0][0]
            recommendations.append(f"Watch for {top_bias.replace('_', ' ')} in future reasoning")
        
        # Learning recommendations
        if learning_insights['average_effectiveness'] < 0.6:
            recommendations.append("Experiment with different learning strategies")
        
        return recommendations
    
    async def _assess_metacognitive_maturity(self) -> str:
        """Assess the overall maturity of metacognitive abilities"""
        
        # This would be based on various metrics
        total_processes = await self.thought_processes_collection.count_documents({})
        
        if total_processes < 10:
            return "developing"
        elif total_processes < 50:
            return "intermediate"
        elif total_processes < 200:
            return "advanced"
        else:
            return "expert"
    
    async def _calculate_self_awareness_level(self) -> float:
        """Calculate overall self-awareness level"""
        
        # This would consider various factors
        recent_processes = await self.thought_processes_collection.count_documents({
            "timestamp": {"$gte": (datetime.utcnow() - timedelta(days=7)).isoformat()}
        })
        
        base_awareness = min(recent_processes / 20, 1.0)  # Max 20 processes per week
        return base_awareness