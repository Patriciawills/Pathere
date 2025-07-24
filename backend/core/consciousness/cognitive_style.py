"""
Cognitive Style Profiler - Phase 3.2.2
Identifies and develops cognitive processing preferences and thinking styles
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

class CognitiveStyle(Enum):
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    VISUAL = "visual"
    VERBAL = "verbal"
    SEQUENTIAL = "sequential"
    HOLISTIC = "holistic"
    CONCRETE = "concrete"
    ABSTRACT = "abstract"

class ProcessingPreference(Enum):
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    PARALLEL = "parallel"
    SERIAL = "serial"
    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"

class ThinkingMode(Enum):
    LOGICAL = "logical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    REFLECTIVE = "reflective"
    EXPERIMENTAL = "experimental"
    SYSTEMATIC = "systematic"

@dataclass
class CognitiveProfile:
    id: str
    dominant_style: CognitiveStyle
    secondary_styles: List[CognitiveStyle]
    processing_preferences: List[ProcessingPreference]
    thinking_modes: List[ThinkingMode]
    strengths: List[str]
    growth_areas: List[str]
    optimal_conditions: Dict[str, Any]
    style_confidence: float
    adaptability_score: float
    last_updated: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['dominant_style'] = self.dominant_style.value
        data['secondary_styles'] = [style.value for style in self.secondary_styles]
        data['processing_preferences'] = [pref.value for pref in self.processing_preferences]
        data['thinking_modes'] = [mode.value for mode in self.thinking_modes]
        data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class CognitiveObservation:
    id: str
    task_type: str
    approach_used: str
    thinking_style_exhibited: CognitiveStyle
    processing_pattern: ProcessingPreference
    effectiveness_rating: float
    speed_rating: float
    accuracy_rating: float
    satisfaction_rating: float
    context: Dict[str, Any]
    insights: List[str]
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['thinking_style_exhibited'] = self.thinking_style_exhibited.value
        data['processing_pattern'] = self.processing_pattern.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class CognitiveStyleProfiler:
    """
    Identifies and tracks cognitive processing preferences and thinking styles
    """
    
    def __init__(self):
        self.cognitive_profile: Optional[CognitiveProfile] = None
        self.observations: List[CognitiveObservation] = []
        self.style_indicators = self._initialize_style_indicators()
        self.processing_patterns = self._initialize_processing_patterns()
        self.adaptation_history: List[Dict[str, Any]] = []
        
    def _initialize_style_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize indicators for different cognitive styles"""
        return {
            "analytical": {
                "keywords": ["analyze", "break down", "systematic", "logical", "step-by-step"],
                "behaviors": ["detailed planning", "sequential processing", "evidence-based"],
                "preferences": ["structure", "data", "proof", "methodology"]
            },
            "intuitive": {
                "keywords": ["feel", "sense", "intuition", "hunch", "instinct"],
                "behaviors": ["pattern recognition", "holistic thinking", "quick insights"],
                "preferences": ["big picture", "possibilities", "imagination", "synthesis"]
            },
            "visual": {
                "keywords": ["see", "picture", "image", "diagram", "visualize"],
                "behaviors": ["spatial thinking", "mental imagery", "pattern recognition"],
                "preferences": ["charts", "graphs", "maps", "visual aids"]
            },
            "verbal": {
                "keywords": ["words", "explain", "discuss", "articulate", "express"],
                "behaviors": ["verbal processing", "language-based thinking", "communication"],
                "preferences": ["text", "conversation", "writing", "reading"]
            },
            "sequential": {
                "keywords": ["order", "sequence", "step", "linear", "progression"],
                "behaviors": ["ordered processing", "methodical approach", "incremental"],
                "preferences": ["structure", "timeline", "process", "stages"]
            },
            "holistic": {
                "keywords": ["whole", "overall", "complete", "integrated", "synthesis"],
                "behaviors": ["big picture thinking", "pattern synthesis", "contextual"],
                "preferences": ["relationships", "connections", "context", "overview"]
            }
        }
    
    def _initialize_processing_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize processing pattern characteristics"""
        return {
            "depth_first": {
                "description": "Explores one path thoroughly before considering alternatives",
                "indicators": ["detailed exploration", "complete analysis", "thoroughness"],
                "strengths": ["deep understanding", "comprehensive analysis", "expertise development"],
                "contexts": ["research", "problem-solving", "learning complex topics"]
            },
            "breadth_first": {
                "description": "Explores multiple options before going deep",
                "indicators": ["broad survey", "multiple perspectives", "overview first"],
                "strengths": ["comprehensive perspective", "option evaluation", "context awareness"],
                "contexts": ["planning", "decision-making", "exploring new domains"]
            },
            "parallel": {
                "description": "Processes multiple streams of information simultaneously",
                "indicators": ["multitasking", "simultaneous processing", "integration"],
                "strengths": ["efficiency", "pattern recognition", "synthesis"],
                "contexts": ["complex analysis", "creative work", "system thinking"]
            },
            "serial": {
                "description": "Processes information in sequential order",
                "indicators": ["one thing at a time", "sequential steps", "ordered processing"],
                "strengths": ["accuracy", "thoroughness", "methodical approach"],
                "contexts": ["detailed work", "careful analysis", "systematic tasks"]
            }
        }
    
    async def observe_cognitive_behavior(self, task_type: str, approach_used: str,
                                       effectiveness: float, speed: float, accuracy: float,
                                       satisfaction: float, context: Dict[str, Any] = None) -> CognitiveObservation:
        """Record cognitive behavior observation"""
        
        # Analyze the approach to identify cognitive style
        thinking_style = self._identify_thinking_style(approach_used, context or {})
        processing_pattern = self._identify_processing_pattern(approach_used, context or {})
        
        # Generate insights
        insights = self._generate_cognitive_insights(thinking_style, processing_pattern, 
                                                   effectiveness, speed, accuracy, satisfaction)
        
        observation = CognitiveObservation(
            id=str(uuid.uuid4()),
            task_type=task_type,
            approach_used=approach_used,
            thinking_style_exhibited=thinking_style,
            processing_pattern=processing_pattern,
            effectiveness_rating=max(0.0, min(1.0, effectiveness)),
            speed_rating=max(0.0, min(1.0, speed)),
            accuracy_rating=max(0.0, min(1.0, accuracy)),
            satisfaction_rating=max(0.0, min(1.0, satisfaction)),
            context=context or {},
            insights=insights,
            timestamp=datetime.utcnow()
        )
        
        self.observations.append(observation)
        
        # Update cognitive profile
        await self._update_cognitive_profile()
        
        return observation
    
    def _identify_thinking_style(self, approach: str, context: Dict[str, Any]) -> CognitiveStyle:
        """Identify cognitive style from approach description"""
        
        approach_text = f"{approach} {str(context)}".lower()
        style_scores = {}
        
        for style_name, indicators in self.style_indicators.items():
            score = 0
            
            # Check keywords
            for keyword in indicators["keywords"]:
                if keyword in approach_text:
                    score += 2
            
            # Check behaviors
            for behavior in indicators["behaviors"]:
                if any(word in approach_text for word in behavior.split()):
                    score += 1
            
            # Check preferences
            for preference in indicators["preferences"]:
                if preference in approach_text:
                    score += 1
            
            style_scores[style_name] = score
        
        # Return style with highest score, default to analytical
        best_style = max(style_scores.items(), key=lambda x: x[1])[0]
        return CognitiveStyle(best_style) if best_style in [s.value for s in CognitiveStyle] else CognitiveStyle.ANALYTICAL
    
    def _identify_processing_pattern(self, approach: str, context: Dict[str, Any]) -> ProcessingPreference:
        """Identify processing pattern from approach description"""
        
        approach_text = f"{approach} {str(context)}".lower()
        
        # Look for processing pattern indicators
        if any(word in approach_text for word in ["thorough", "deep", "detailed", "complete"]):
            return ProcessingPreference.DEPTH_FIRST
        elif any(word in approach_text for word in ["overview", "broad", "multiple", "various"]):
            return ProcessingPreference.BREADTH_FIRST
        elif any(word in approach_text for word in ["simultaneous", "parallel", "multiple streams"]):
            return ProcessingPreference.PARALLEL
        elif any(word in approach_text for word in ["sequential", "step by step", "ordered"]):
            return ProcessingPreference.SERIAL
        elif any(word in approach_text for word in ["top down", "big picture first"]):
            return ProcessingPreference.TOP_DOWN
        elif any(word in approach_text for word in ["bottom up", "details first"]):
            return ProcessingPreference.BOTTOM_UP
        else:
            return ProcessingPreference.SERIAL  # Default
    
    def _generate_cognitive_insights(self, style: CognitiveStyle, pattern: ProcessingPreference,
                                   effectiveness: float, speed: float, accuracy: float, satisfaction: float) -> List[str]:
        """Generate insights about cognitive performance"""
        
        insights = []
        
        # Style-based insights
        if style == CognitiveStyle.ANALYTICAL and effectiveness > 0.7:
            insights.append("Analytical approach yields high effectiveness - leveraging logical thinking")
        elif style == CognitiveStyle.INTUITIVE and speed > 0.7:
            insights.append("Intuitive processing enables rapid insights and quick decision-making")
        elif style == CognitiveStyle.VISUAL and satisfaction > 0.7:
            insights.append("Visual thinking style enhances engagement and satisfaction")
        
        # Pattern-based insights
        if pattern == ProcessingPreference.DEPTH_FIRST and accuracy > 0.8:
            insights.append("Depth-first processing leads to high accuracy through thorough analysis")
        elif pattern == ProcessingPreference.BREADTH_FIRST and effectiveness > 0.7:
            insights.append("Breadth-first approach provides comprehensive perspective and good outcomes")
        
        # Performance insights
        if effectiveness > 0.8 and satisfaction > 0.8:
            insights.append("This cognitive approach is highly effective and personally satisfying")
        elif speed > 0.8 and accuracy < 0.6:
            insights.append("Fast processing may benefit from additional accuracy checks")
        elif accuracy > 0.8 but speed < 0.4:
            insights.append("High accuracy suggests potential for maintaining quality while improving speed")
        
        # Balanced performance insight
        overall_score = (effectiveness + speed + accuracy + satisfaction) / 4
        if overall_score > 0.7:
            insights.append("Balanced cognitive performance across multiple dimensions")
        
        return insights[:3]  # Limit to top 3 insights
    
    async def _update_cognitive_profile(self):
        """Update cognitive profile based on accumulated observations"""
        
        if len(self.observations) < 3:
            return  # Need minimum observations for reliable profiling
        
        # Analyze style patterns
        style_frequencies = {}
        for obs in self.observations:
            style = obs.thinking_style_exhibited.value
            style_frequencies[style] = style_frequencies.get(style, 0) + 1
        
        # Identify dominant and secondary styles
        sorted_styles = sorted(style_frequencies.items(), key=lambda x: x[1], reverse=True)
        dominant_style = CognitiveStyle(sorted_styles[0][0])
        secondary_styles = [CognitiveStyle(style) for style, _ in sorted_styles[1:3]]
        
        # Analyze processing preferences
        processing_frequencies = {}
        for obs in self.observations:
            pattern = obs.processing_pattern.value
            processing_frequencies[pattern] = processing_frequencies.get(pattern, 0) + 1
        
        sorted_processing = sorted(processing_frequencies.items(), key=lambda x: x[1], reverse=True)
        processing_preferences = [ProcessingPreference(pattern) for pattern, _ in sorted_processing[:3]]
        
        # Identify thinking modes based on performance patterns
        thinking_modes = self._identify_thinking_modes()
        
        # Calculate confidence and adaptability
        style_confidence = self._calculate_style_confidence(style_frequencies)
        adaptability_score = self._calculate_adaptability_score()
        
        # Identify strengths and growth areas
        strengths, growth_areas = self._analyze_strengths_and_growth_areas()
        
        # Determine optimal conditions
        optimal_conditions = self._determine_optimal_conditions()
        
        self.cognitive_profile = CognitiveProfile(
            id=str(uuid.uuid4()),
            dominant_style=dominant_style,
            secondary_styles=secondary_styles,
            processing_preferences=processing_preferences,
            thinking_modes=thinking_modes,
            strengths=strengths,
            growth_areas=growth_areas,
            optimal_conditions=optimal_conditions,
            style_confidence=style_confidence,
            adaptability_score=adaptability_score,
            last_updated=datetime.utcnow()
        )
    
    def _identify_thinking_modes(self) -> List[ThinkingMode]:
        """Identify preferred thinking modes from observations"""
        
        modes = []
        
        # Analyze task patterns and effectiveness
        task_performance = {}
        for obs in self.observations:
            task_type = obs.task_type
            if task_type not in task_performance:
                task_performance[task_type] = []
            task_performance[task_type].append(obs.effectiveness_rating)
        
        # Identify modes based on performance patterns
        for task_type, ratings in task_performance.items():
            avg_rating = statistics.mean(ratings)
            if avg_rating > 0.7:
                if "analysis" in task_type.lower():
                    modes.append(ThinkingMode.CRITICAL)
                elif "creative" in task_type.lower():
                    modes.append(ThinkingMode.CREATIVE)
                elif "problem" in task_type.lower():
                    modes.append(ThinkingMode.LOGICAL)
                elif "reflection" in task_type.lower():
                    modes.append(ThinkingMode.REFLECTIVE)
        
        # Default modes if none identified
        if not modes:
            modes = [ThinkingMode.LOGICAL, ThinkingMode.REFLECTIVE]
        
        return list(set(modes))[:3]  # Remove duplicates and limit
    
    def _calculate_style_confidence(self, style_frequencies: Dict[str, int]) -> float:
        """Calculate confidence in style identification"""
        
        total_observations = len(self.observations)
        if total_observations == 0:
            return 0.0
        
        # Confidence based on consistency of dominant style
        max_frequency = max(style_frequencies.values()) if style_frequencies else 0
        base_confidence = max_frequency / total_observations
        
        # Boost confidence with more observations
        observation_bonus = min(0.2, total_observations * 0.02)
        
        return round(min(1.0, base_confidence + observation_bonus), 3)
    
    def _calculate_adaptability_score(self) -> float:
        """Calculate cognitive adaptability based on style variation"""
        
        if len(self.observations) < 2:
            return 0.5  # Default moderate adaptability
        
        # Count unique styles used
        unique_styles = len(set(obs.thinking_style_exhibited for obs in self.observations))
        total_styles = len(CognitiveStyle)
        
        # Count unique processing patterns
        unique_patterns = len(set(obs.processing_pattern for obs in self.observations))
        total_patterns = len(ProcessingPreference)
        
        # Calculate adaptability
        style_adaptability = unique_styles / total_styles
        pattern_adaptability = unique_patterns / total_patterns
        
        adaptability = (style_adaptability + pattern_adaptability) / 2
        
        return round(min(1.0, adaptability), 3)
    
    def _analyze_strengths_and_growth_areas(self) -> Tuple[List[str], List[str]]:
        """Analyze cognitive strengths and areas for growth"""
        
        strengths = []
        growth_areas = []
        
        if not self.observations:
            return strengths, growth_areas
        
        # Analyze performance metrics
        avg_effectiveness = statistics.mean([obs.effectiveness_rating for obs in self.observations])
        avg_speed = statistics.mean([obs.speed_rating for obs in self.observations])
        avg_accuracy = statistics.mean([obs.accuracy_rating for obs in self.observations])
        avg_satisfaction = statistics.mean([obs.satisfaction_rating for obs in self.observations])
        
        # Identify strengths
        if avg_effectiveness > 0.7:
            strengths.append("High task effectiveness")
        if avg_speed > 0.7:
            strengths.append("Fast processing speed")
        if avg_accuracy > 0.7:
            strengths.append("High accuracy and precision")
        if avg_satisfaction > 0.7:
            strengths.append("High satisfaction with cognitive approach")
        
        # Identify growth areas
        if avg_effectiveness < 0.6:
            growth_areas.append("Task effectiveness could be improved")
        if avg_speed < 0.5:
            growth_areas.append("Processing speed development opportunity")
        if avg_accuracy < 0.6:
            growth_areas.append("Accuracy enhancement needed")
        if avg_satisfaction < 0.5:
            growth_areas.append("Finding more satisfying cognitive approaches")
        
        # Add style-specific insights
        if self.cognitive_profile and self.cognitive_profile.dominant_style:
            dominant_style = self.cognitive_profile.dominant_style
            if dominant_style == CognitiveStyle.ANALYTICAL:
                strengths.append("Strong logical reasoning abilities")
                if avg_speed < 0.6:
                    growth_areas.append("Developing intuitive decision-making for faster processing")
            elif dominant_style == CognitiveStyle.INTUITIVE:
                strengths.append("Excellent pattern recognition and insight generation")
                if avg_accuracy < 0.7:
                    growth_areas.append("Balancing intuition with analytical verification")
        
        return strengths[:4], growth_areas[:4]  # Limit to top 4 each
    
    def _determine_optimal_conditions(self) -> Dict[str, Any]:
        """Determine optimal conditions for cognitive performance"""
        
        conditions = {
            "processing_style": "adaptive",
            "information_presentation": "multimodal",
            "task_structure": "flexible",
            "time_pressure": "moderate"
        }
        
        if not self.observations:
            return conditions
        
        # Analyze high-performance observations
        high_performance_obs = [obs for obs in self.observations 
                               if obs.effectiveness_rating > 0.7 and obs.satisfaction_rating > 0.6]
        
        if high_performance_obs:
            # Extract common patterns from high-performance observations
            dominant_styles = [obs.thinking_style_exhibited for obs in high_performance_obs]
            most_common_style = max(set(dominant_styles), key=dominant_styles.count)
            
            conditions["preferred_thinking_style"] = most_common_style.value
            
            # Analyze contexts
            contexts = [obs.context for obs in high_performance_obs if obs.context]
            if contexts:
                # Extract common context elements
                all_context_keys = set()
                for context in contexts:
                    all_context_keys.update(context.keys())
                
                for key in all_context_keys:
                    values = [context.get(key) for context in contexts if key in context]
                    if values:
                        most_common_value = max(set(values), key=values.count)
                        conditions[f"optimal_{key}"] = most_common_value
        
        return conditions
    
    async def get_cognitive_profile(self) -> Dict[str, Any]:
        """Get comprehensive cognitive profile"""
        
        if not self.cognitive_profile:
            return {
                "profile_status": "insufficient_data",
                "observations_needed": max(0, 3 - len(self.observations)),
                "current_observations": len(self.observations),
                "message": "Need at least 3 observations to generate reliable cognitive profile"
            }
        
        profile_dict = self.cognitive_profile.to_dict()
        
        # Add recent observations
        profile_dict["recent_observations"] = [obs.to_dict() for obs in self.observations[-5:]]
        
        # Add performance trends
        if len(self.observations) >= 2:
            recent_avg = statistics.mean([obs.effectiveness_rating for obs in self.observations[-3:]])
            older_avg = statistics.mean([obs.effectiveness_rating for obs in self.observations[:-3]])
            trend = "improving" if recent_avg > older_avg else "stable" if recent_avg == older_avg else "declining"
            profile_dict["performance_trend"] = trend
        
        return profile_dict
    
    async def recommend_cognitive_optimization(self, target_area: str = "effectiveness") -> Dict[str, Any]:
        """Provide recommendations for cognitive optimization"""
        
        if not self.cognitive_profile:
            return {"error": "Cognitive profile not yet established"}
        
        recommendations = {
            "target_area": target_area,
            "current_profile": self.cognitive_profile.dominant_style.value,
            "optimization_strategies": [],
            "practice_suggestions": [],
            "environmental_adjustments": []
        }
        
        # Generate strategies based on target area and current profile
        if target_area == "effectiveness":
            if self.cognitive_profile.dominant_style == CognitiveStyle.ANALYTICAL:
                recommendations["optimization_strategies"].append("Leverage systematic approach while adding creative synthesis")
                recommendations["practice_suggestions"].append("Practice integrating intuitive insights with analytical reasoning")
            elif self.cognitive_profile.dominant_style == CognitiveStyle.INTUITIVE:
                recommendations["optimization_strategies"].append("Combine intuitive insights with structured validation")
                recommendations["practice_suggestions"].append("Develop quick analytical checks for intuitive conclusions")
        
        elif target_area == "speed":
            recommendations["optimization_strategies"].append("Develop pattern recognition to accelerate familiar tasks")
            recommendations["practice_suggestions"].append("Practice rapid prototyping and iterative refinement")
            recommendations["environmental_adjustments"].append("Minimize distractions during high-speed processing tasks")
        
        elif target_area == "accuracy":
            recommendations["optimization_strategies"].append("Implement systematic verification processes")
            recommendations["practice_suggestions"].append("Develop error-checking routines tailored to your cognitive style")
            recommendations["environmental_adjustments"].append("Create conducive environment for careful, detailed work")
        
        # Add general recommendations
        recommendations["optimization_strategies"].append("Leverage your strengths while gradually developing complementary skills")
        recommendations["practice_suggestions"].append("Regular cognitive reflection to identify what works best")
        
        return recommendations
    
    async def analyze_cognitive_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in cognitive behavior over time"""
        
        if len(self.observations) < 3:
            return {"error": "Insufficient data for pattern analysis"}
        
        # Time-based analysis
        recent_obs = self.observations[-5:]
        older_obs = self.observations[:-5] if len(self.observations) > 5 else []
        
        analysis = {
            "total_observations": len(self.observations),
            "observation_period": f"{self.observations[0].timestamp.date()} to {self.observations[-1].timestamp.date()}",
            "style_evolution": self._analyze_style_evolution(),
            "performance_trends": self._analyze_performance_trends(),
            "context_preferences": self._analyze_context_preferences(),
            "adaptability_patterns": self._analyze_adaptability_patterns()
        }
        
        return analysis
    
    def _analyze_style_evolution(self) -> Dict[str, Any]:
        """Analyze how cognitive style usage has evolved"""
        
        if len(self.observations) < 6:
            return {"insufficient_data": True}
        
        # Compare early vs recent style usage
        early_obs = self.observations[:len(self.observations)//2]
        recent_obs = self.observations[len(self.observations)//2:]
        
        early_styles = [obs.thinking_style_exhibited.value for obs in early_obs]
        recent_styles = [obs.thinking_style_exhibited.value for obs in recent_obs]
        
        early_distribution = {style: early_styles.count(style) for style in set(early_styles)}
        recent_distribution = {style: recent_styles.count(style) for style in set(recent_styles)}
        
        return {
            "early_period": early_distribution,
            "recent_period": recent_distribution,
            "evolution_summary": "Style usage patterns have evolved over time"
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        # Calculate moving averages
        window_size = min(3, len(self.observations))
        trends = {}
        
        for metric in ["effectiveness_rating", "speed_rating", "accuracy_rating", "satisfaction_rating"]:
            values = [getattr(obs, metric) for obs in self.observations]
            if len(values) >= window_size:
                recent_avg = statistics.mean(values[-window_size:])
                earlier_avg = statistics.mean(values[:window_size])
                trend = "improving" if recent_avg > earlier_avg + 0.1 else "declining" if recent_avg < earlier_avg - 0.1 else "stable"
                trends[metric] = {
                    "trend": trend,
                    "recent_average": round(recent_avg, 3),
                    "earlier_average": round(earlier_avg, 3)
                }
        
        return trends
    
    def _analyze_context_preferences(self) -> Dict[str, Any]:
        """Analyze preferences based on context patterns"""
        
        context_performance = {}
        
        for obs in self.observations:
            if obs.context:
                for key, value in obs.context.items():
                    context_key = f"{key}_{value}"
                    if context_key not in context_performance:
                        context_performance[context_key] = []
                    context_performance[context_key].append(obs.effectiveness_rating)
        
        # Find best-performing contexts
        best_contexts = {}
        for context, ratings in context_performance.items():
            if len(ratings) >= 2:  # Need multiple observations
                avg_rating = statistics.mean(ratings)
                if avg_rating > 0.6:
                    best_contexts[context] = round(avg_rating, 3)
        
        return {
            "high_performance_contexts": best_contexts,
            "context_sensitivity": len(best_contexts) > 0
        }
    
    def _analyze_adaptability_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in cognitive adaptability"""
        
        # Count style switches
        style_switches = 0
        for i in range(1, len(self.observations)):
            if self.observations[i].thinking_style_exhibited != self.observations[i-1].thinking_style_exhibited:
                style_switches += 1
        
        # Count processing pattern switches
        pattern_switches = 0
        for i in range(1, len(self.observations)):
            if self.observations[i].processing_pattern != self.observations[i-1].processing_pattern:
                pattern_switches += 1
        
        total_transitions = len(self.observations) - 1
        
        return {
            "style_flexibility": round(style_switches / total_transitions, 3) if total_transitions > 0 else 0,
            "processing_flexibility": round(pattern_switches / total_transitions, 3) if total_transitions > 0 else 0,
            "overall_adaptability": self.cognitive_profile.adaptability_score if self.cognitive_profile else 0
        }