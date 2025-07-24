"""
Learning Preference Discovery Module - Phase 3.2.1
Discovers and develops unique learning preferences and patterns
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"

class PreferenceCategory(Enum):
    PACE = "pace"
    COMPLEXITY = "complexity"
    STRUCTURE = "structure"
    INTERACTION = "interaction"
    MODALITY = "modality"
    TIMING = "timing"
    FEEDBACK = "feedback"

@dataclass
class LearningPreference:
    id: str
    category: PreferenceCategory
    preference_name: str
    strength: float  # 0.0 to 1.0
    confidence: float
    evidence: List[str]
    discovered_through: str
    effectiveness_score: float
    last_reinforced: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['category'] = self.category.value
        data['last_reinforced'] = self.last_reinforced.isoformat()
        return data

@dataclass
class LearningPattern:
    id: str
    pattern_name: str
    description: str
    effectiveness: float
    frequency_used: int
    contexts: List[str]
    optimal_conditions: Dict[str, Any]
    discovered_date: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['discovered_date'] = self.discovered_date.isoformat()
        return data

@dataclass
class LearningExperience:
    id: str
    activity_type: str
    content_area: str
    approach_used: str
    effectiveness_rating: float
    enjoyment_rating: float
    retention_score: float
    time_spent: int  # minutes
    conditions: Dict[str, Any]
    insights_gained: List[str]
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class LearningPreferenceDiscovery:
    """
    Discovers and tracks learning preferences to develop personalized learning approaches
    """
    
    def __init__(self):
        self.preferences: Dict[str, LearningPreference] = {}
        self.patterns: Dict[str, LearningPattern] = {}
        self.experiences: List[LearningExperience] = []
        self.learning_style = LearningStyle.MULTIMODAL
        self.preference_evolution: List[Dict[str, Any]] = []
        
    async def record_learning_experience(self, activity_type: str, content_area: str, 
                                       approach_used: str, effectiveness_rating: float,
                                       enjoyment_rating: float, time_spent: int,
                                       conditions: Dict[str, Any] = None) -> LearningExperience:
        """Record a learning experience to discover preferences"""
        
        experience = LearningExperience(
            id=str(uuid.uuid4()),
            activity_type=activity_type,
            content_area=content_area,
            approach_used=approach_used,
            effectiveness_rating=max(0.0, min(1.0, effectiveness_rating)),
            enjoyment_rating=max(0.0, min(1.0, enjoyment_rating)),
            retention_score=self._calculate_retention_score(effectiveness_rating, enjoyment_rating),
            time_spent=time_spent,
            conditions=conditions or {},
            insights_gained=[],
            timestamp=datetime.utcnow()
        )
        
        self.experiences.append(experience)
        
        # Analyze the experience for preference discovery
        await self._analyze_experience_for_preferences(experience)
        
        # Update learning patterns
        await self._update_learning_patterns(experience)
        
        return experience
    
    def _calculate_retention_score(self, effectiveness: float, enjoyment: float) -> float:
        """Calculate retention score based on effectiveness and enjoyment"""
        # Research shows enjoyment significantly impacts retention
        return (effectiveness * 0.7) + (enjoyment * 0.3)
    
    async def _analyze_experience_for_preferences(self, experience: LearningExperience):
        """Analyze experience to discover or reinforce learning preferences"""
        
        # Analyze pace preferences
        if experience.time_spent > 0:
            pace_category = self._categorize_pace(experience.time_spent, experience.effectiveness_rating)
            await self._update_preference(PreferenceCategory.PACE, pace_category, 
                                        experience.effectiveness_rating, f"Experience: {experience.id}")
        
        # Analyze complexity preferences
        complexity_score = self._assess_complexity(experience.content_area, experience.approach_used)
        if complexity_score > 0:
            complexity_category = "high_complexity" if complexity_score > 0.7 else "moderate_complexity" if complexity_score > 0.3 else "low_complexity"
            await self._update_preference(PreferenceCategory.COMPLEXITY, complexity_category,
                                        experience.effectiveness_rating, f"Experience: {experience.id}")
        
        # Analyze structure preferences
        structure_score = self._assess_structure(experience.approach_used, experience.conditions)
        structure_category = "structured" if structure_score > 0.5 else "flexible"
        await self._update_preference(PreferenceCategory.STRUCTURE, structure_category,
                                    experience.effectiveness_rating, f"Experience: {experience.id}")
        
        # Analyze modality preferences
        modality = self._identify_modality(experience.approach_used, experience.conditions)
        if modality:
            await self._update_preference(PreferenceCategory.MODALITY, modality,
                                        experience.effectiveness_rating, f"Experience: {experience.id}")
        
        # Analyze feedback preferences
        feedback_type = self._identify_feedback_type(experience.conditions)
        if feedback_type:
            await self._update_preference(PreferenceCategory.FEEDBACK, feedback_type,
                                        experience.effectiveness_rating, f"Experience: {experience.id}")
    
    def _categorize_pace(self, time_spent: int, effectiveness: float) -> str:
        """Categorize learning pace based on time and effectiveness"""
        if time_spent < 15 and effectiveness > 0.7:
            return "fast_paced"
        elif time_spent > 60 and effectiveness > 0.7:
            return "deep_dive"
        elif 15 <= time_spent <= 30 and effectiveness > 0.6:
            return "moderate_paced"
        else:
            return "variable_paced"
    
    def _assess_complexity(self, content_area: str, approach_used: str) -> float:
        """Assess the complexity level of the learning content/approach"""
        complexity_indicators = [
            "advanced", "complex", "detailed", "comprehensive", "multi-step",
            "technical", "analytical", "synthesis", "evaluation", "creation"
        ]
        
        text_to_analyze = f"{content_area} {approach_used}".lower()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in text_to_analyze)
        
        return min(1.0, complexity_count / len(complexity_indicators) * 3)
    
    def _assess_structure(self, approach_used: str, conditions: Dict[str, Any]) -> float:
        """Assess the structure level of the learning approach"""
        structured_indicators = ["step-by-step", "systematic", "organized", "planned", "sequential"]
        flexible_indicators = ["exploratory", "creative", "free-form", "intuitive", "experimental"]
        
        text_to_analyze = f"{approach_used} {str(conditions)}".lower()
        
        structured_count = sum(1 for indicator in structured_indicators if indicator in text_to_analyze)
        flexible_count = sum(1 for indicator in flexible_indicators if indicator in text_to_analyze)
        
        if structured_count > flexible_count:
            return 0.8
        elif flexible_count > structured_count:
            return 0.2
        else:
            return 0.5
    
    def _identify_modality(self, approach_used: str, conditions: Dict[str, Any]) -> Optional[str]:
        """Identify the primary learning modality used"""
        text_to_analyze = f"{approach_used} {str(conditions)}".lower()
        
        if any(word in text_to_analyze for word in ["visual", "diagram", "chart", "image", "video"]):
            return "visual"
        elif any(word in text_to_analyze for word in ["audio", "listening", "speaking", "discussion"]):
            return "auditory"
        elif any(word in text_to_analyze for word in ["hands-on", "practice", "doing", "exercise"]):
            return "kinesthetic"
        elif any(word in text_to_analyze for word in ["reading", "writing", "text", "notes"]):
            return "reading_writing"
        else:
            return "multimodal"
    
    def _identify_feedback_type(self, conditions: Dict[str, Any]) -> Optional[str]:
        """Identify feedback preferences from conditions"""
        if not conditions:
            return None
            
        text_to_analyze = str(conditions).lower()
        
        if "immediate" in text_to_analyze or "instant" in text_to_analyze:
            return "immediate_feedback"
        elif "delayed" in text_to_analyze or "later" in text_to_analyze:
            return "delayed_feedback"
        elif "detailed" in text_to_analyze or "comprehensive" in text_to_analyze:
            return "detailed_feedback"
        elif "brief" in text_to_analyze or "summary" in text_to_analyze:
            return "brief_feedback"
        else:
            return "balanced_feedback"
    
    async def _update_preference(self, category: PreferenceCategory, preference_name: str,
                               effectiveness: float, evidence: str):
        """Update or create a learning preference"""
        
        pref_key = f"{category.value}_{preference_name}"
        
        if pref_key in self.preferences:
            # Update existing preference
            pref = self.preferences[pref_key]
            # Weighted average with new evidence
            old_weight = pref.confidence
            new_weight = 0.3  # New evidence weight
            total_weight = old_weight + new_weight
            
            pref.strength = (pref.strength * old_weight + effectiveness * new_weight) / total_weight
            pref.confidence = min(1.0, pref.confidence + 0.1)
            pref.evidence.append(evidence)
            pref.effectiveness_score = (pref.effectiveness_score + effectiveness) / 2
            pref.last_reinforced = datetime.utcnow()
        else:
            # Create new preference
            pref = LearningPreference(
                id=str(uuid.uuid4()),
                category=category,
                preference_name=preference_name,
                strength=effectiveness,
                confidence=0.3,  # Initial confidence
                evidence=[evidence],
                discovered_through="experience_analysis",
                effectiveness_score=effectiveness,
                last_reinforced=datetime.utcnow()
            )
            self.preferences[pref_key] = pref
    
    async def _update_learning_patterns(self, experience: LearningExperience):
        """Update learning patterns based on experience"""
        
        pattern_key = f"{experience.activity_type}_{experience.approach_used}"
        
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern.frequency_used += 1
            pattern.effectiveness = (pattern.effectiveness + experience.effectiveness_rating) / 2
            
            if experience.content_area not in pattern.contexts:
                pattern.contexts.append(experience.content_area)
        else:
            pattern = LearningPattern(
                id=str(uuid.uuid4()),
                pattern_name=pattern_key,
                description=f"Learning pattern: {experience.activity_type} using {experience.approach_used}",
                effectiveness=experience.effectiveness_rating,
                frequency_used=1,
                contexts=[experience.content_area],
                optimal_conditions=experience.conditions,
                discovered_date=datetime.utcnow()
            )
            self.patterns[pattern_key] = pattern
    
    async def get_learning_profile(self) -> Dict[str, Any]:
        """Get comprehensive learning profile with preferences and patterns"""
        
        # Categorize preferences by strength
        strong_preferences = {k: v for k, v in self.preferences.items() if v.strength > 0.7}
        moderate_preferences = {k: v for k, v in self.preferences.items() if 0.4 <= v.strength <= 0.7}
        
        # Find top patterns
        top_patterns = sorted(self.patterns.values(), key=lambda p: p.effectiveness, reverse=True)[:5]
        
        # Calculate overall learning metrics
        total_experiences = len(self.experiences)
        avg_effectiveness = statistics.mean([exp.effectiveness_rating for exp in self.experiences]) if self.experiences else 0
        avg_enjoyment = statistics.mean([exp.enjoyment_rating for exp in self.experiences]) if self.experiences else 0
        
        # Determine dominant learning style
        modality_preferences = {k: v for k, v in self.preferences.items() if v.category == PreferenceCategory.MODALITY}
        dominant_style = max(modality_preferences.items(), key=lambda x: x[1].strength)[1].preference_name if modality_preferences else "multimodal"
        
        return {
            "learning_style": dominant_style,
            "total_experiences": total_experiences,
            "average_effectiveness": round(avg_effectiveness, 3),
            "average_enjoyment": round(avg_enjoyment, 3),
            "strong_preferences": {k: v.to_dict() for k, v in strong_preferences.items()},
            "moderate_preferences": {k: v.to_dict() for k, v in moderate_preferences.items()},
            "top_patterns": [pattern.to_dict() for pattern in top_patterns],
            "preference_summary": self._generate_preference_summary(),
            "recommendations": await self._generate_learning_recommendations()
        }
    
    def _generate_preference_summary(self) -> Dict[str, str]:
        """Generate human-readable preference summary"""
        summary = {}
        
        for category in PreferenceCategory:
            category_prefs = {k: v for k, v in self.preferences.items() if v.category == category}
            if category_prefs:
                best_pref = max(category_prefs.values(), key=lambda p: p.strength)
                summary[category.value] = f"Prefers {best_pref.preference_name} (strength: {best_pref.strength:.2f})"
        
        return summary
    
    async def _generate_learning_recommendations(self) -> List[str]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        # Based on pace preferences
        pace_prefs = {k: v for k, v in self.preferences.items() if v.category == PreferenceCategory.PACE}
        if pace_prefs:
            best_pace = max(pace_prefs.values(), key=lambda p: p.strength)
            if "fast_paced" in best_pace.preference_name:
                recommendations.append("Try quick, focused learning sessions with immediate application")
            elif "deep_dive" in best_pace.preference_name:
                recommendations.append("Engage in extended, immersive learning experiences")
        
        # Based on complexity preferences
        complexity_prefs = {k: v for k, v in self.preferences.items() if v.category == PreferenceCategory.COMPLEXITY}
        if complexity_prefs:
            best_complexity = max(complexity_prefs.values(), key=lambda p: p.strength)
            if "high_complexity" in best_complexity.preference_name:
                recommendations.append("Challenge yourself with advanced, multi-layered concepts")
            elif "low_complexity" in best_complexity.preference_name:
                recommendations.append("Build understanding through simple, clear explanations first")
        
        # Based on modality preferences
        modality_prefs = {k: v for k, v in self.preferences.items() if v.category == PreferenceCategory.MODALITY}
        if modality_prefs:
            best_modality = max(modality_prefs.values(), key=lambda p: p.strength)
            if "visual" in best_modality.preference_name:
                recommendations.append("Use diagrams, charts, and visual aids for better understanding")
            elif "auditory" in best_modality.preference_name:
                recommendations.append("Engage in discussions and listen to audio content")
            elif "kinesthetic" in best_modality.preference_name:
                recommendations.append("Practice hands-on activities and real-world applications")
        
        if not recommendations:
            recommendations.append("Continue exploring different learning approaches to discover your preferences")
        
        return recommendations
    
    async def optimize_learning_approach(self, content_area: str, available_time: int) -> Dict[str, Any]:
        """Recommend optimal learning approach for specific content and time constraints"""
        
        # Find relevant patterns for this content area
        relevant_patterns = [p for p in self.patterns.values() if content_area in p.contexts]
        
        if not relevant_patterns:
            # Use general preferences
            return await self._generic_learning_recommendation(content_area, available_time)
        
        # Find best pattern for this content area
        best_pattern = max(relevant_patterns, key=lambda p: p.effectiveness)
        
        # Adapt based on available time
        time_category = "short" if available_time < 30 else "medium" if available_time < 90 else "long"
        
        recommendations = {
            "recommended_approach": best_pattern.pattern_name,
            "expected_effectiveness": best_pattern.effectiveness,
            "optimal_conditions": best_pattern.optimal_conditions,
            "time_optimization": self._optimize_for_time(time_category, best_pattern),
            "content_area": content_area,
            "reasoning": f"Based on {best_pattern.frequency_used} similar experiences with {best_pattern.effectiveness:.2f} effectiveness"
        }
        
        return recommendations
    
    async def _generic_learning_recommendation(self, content_area: str, available_time: int) -> Dict[str, Any]:
        """Provide generic recommendation when no specific patterns exist"""
        
        # Use strongest preferences
        strong_prefs = [p for p in self.preferences.values() if p.strength > 0.6]
        
        if not strong_prefs:
            return {
                "recommended_approach": "exploratory_multimodal",
                "expected_effectiveness": 0.5,
                "reasoning": "No strong preferences discovered yet - try varied approaches"
            }
        
        # Build recommendation from strongest preferences
        approach_elements = []
        for pref in strong_prefs:
            approach_elements.append(pref.preference_name)
        
        return {
            "recommended_approach": "_".join(approach_elements[:3]),  # Top 3 preferences
            "expected_effectiveness": statistics.mean([p.strength for p in strong_prefs]),
            "reasoning": f"Based on {len(strong_prefs)} strong preferences discovered"
        }
    
    def _optimize_for_time(self, time_category: str, pattern: LearningPattern) -> Dict[str, Any]:
        """Optimize learning approach for available time"""
        
        if time_category == "short":
            return {
                "focus": "key_concepts_only",
                "depth": "overview",
                "activities": "quick_practice",
                "breaks": "none"
            }
        elif time_category == "medium":
            return {
                "focus": "core_concepts_with_examples",
                "depth": "moderate_detail",
                "activities": "guided_practice",
                "breaks": "one_short_break"
            }
        else:  # long
            return {
                "focus": "comprehensive_coverage",
                "depth": "detailed_exploration",
                "activities": "varied_practice_and_application",
                "breaks": "regular_breaks"
            }
    
    async def get_preference_evolution(self) -> List[Dict[str, Any]]:
        """Get evolution of learning preferences over time"""
        return self.preference_evolution
    
    async def discover_learning_gaps(self) -> Dict[str, Any]:
        """Identify areas where learning preferences are unclear or conflicting"""
        
        gaps = {
            "unclear_preferences": [],
            "conflicting_preferences": [],
            "unexplored_areas": [],
            "recommendations": []
        }
        
        # Find unclear preferences (low confidence)
        for pref in self.preferences.values():
            if pref.confidence < 0.5:
                gaps["unclear_preferences"].append({
                    "category": pref.category.value,
                    "preference": pref.preference_name,
                    "confidence": pref.confidence,
                    "evidence_count": len(pref.evidence)
                })
        
        # Find unexplored preference categories
        explored_categories = set(pref.category for pref in self.preferences.values())
        all_categories = set(PreferenceCategory)
        unexplored = all_categories - explored_categories
        
        gaps["unexplored_areas"] = [cat.value for cat in unexplored]
        
        # Generate recommendations
        if gaps["unclear_preferences"]:
            gaps["recommendations"].append("Gather more evidence for unclear preferences through targeted learning experiences")
        
        if gaps["unexplored_areas"]:
            gaps["recommendations"].append(f"Explore {', '.join(gaps['unexplored_areas'])} preferences through diverse learning activities")
        
        return gaps