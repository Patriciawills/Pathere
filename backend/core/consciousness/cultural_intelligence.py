"""
Cultural Intelligence Module for Human-like Consciousness System

This module implements cultural awareness and adaptability, enabling the AI to
understand, respect, and adapt to different cultural contexts with sensitivity
and appropriate communication styles.

Features:
- Cultural context recognition and analysis  
- Communication style adaptation
- Cultural sensitivity assessment
- Cross-cultural understanding
- Cultural learning and memory
- Respectful interaction patterns
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
from dataclasses import dataclass, asdict
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CulturalDimension(Enum):
    """Cultural dimensions based on Hofstede's cultural dimensions theory"""
    POWER_DISTANCE = "power_distance"           # Hierarchy acceptance
    INDIVIDUALISM = "individualism"             # Individual vs collective focus
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"  # Tolerance for ambiguity
    MASCULINITY = "masculinity"                 # Competitive vs cooperative
    LONG_TERM_ORIENTATION = "long_term_orientation"  # Future vs present focus
    INDULGENCE = "indulgence"                   # Gratification vs restraint

class CommunicationStyle(Enum):
    """Communication style preferences"""
    DIRECT = "direct"                 # Explicit, straightforward
    INDIRECT = "indirect"             # Implicit, contextual
    FORMAL = "formal"                 # Respectful, hierarchical
    INFORMAL = "informal"             # Casual, egalitarian
    HIGH_CONTEXT = "high_context"     # Context-dependent meaning
    LOW_CONTEXT = "low_context"       # Explicit meaning

class CulturalValue(Enum):
    """Core cultural values"""
    RESPECT_FOR_ELDERS = "respect_for_elders"
    FAMILY_ORIENTATION = "family_orientation"
    EDUCATION_EMPHASIS = "education_emphasis"
    SPIRITUAL_AWARENESS = "spiritual_awareness"
    COMMUNITY_FOCUS = "community_focus"
    TRADITION_PRESERVATION = "tradition_preservation"
    INNOVATION_OPENNESS = "innovation_openness"
    HARMONY_SEEKING = "harmony_seeking"

class CulturalContext(Enum):
    """Types of cultural contexts"""
    NATIONAL = "national"             # Country-based culture
    REGIONAL = "regional"             # Regional/geographic culture
    ETHNIC = "ethnic"                 # Ethnic group culture
    RELIGIOUS = "religious"           # Religious/spiritual culture
    PROFESSIONAL = "professional"     # Work/industry culture
    GENERATIONAL = "generational"     # Age/generation culture
    SOCIOECONOMIC = "socioeconomic"   # Economic class culture

@dataclass
class CulturalProfile:
    """Represents a cultural profile with dimensions and preferences"""
    id: str
    name: str
    context_type: CulturalContext
    cultural_dimensions: Dict[CulturalDimension, float]  # 0.0 to 1.0 scores
    communication_preferences: List[CommunicationStyle]
    core_values: List[CulturalValue]
    interaction_patterns: Dict[str, str]
    sensitivity_areas: List[str]  # Topics requiring special care
    greeting_customs: List[str]
    time_orientation: str  # "monochronic" or "polychronic"
    created_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'context_type': self.context_type.value,
            'cultural_dimensions': {dim.value: score for dim, score in self.cultural_dimensions.items()},
            'communication_preferences': [pref.value for pref in self.communication_preferences],
            'core_values': [val.value for val in self.core_values],
            'interaction_patterns': self.interaction_patterns,
            'sensitivity_areas': self.sensitivity_areas,
            'greeting_customs': self.greeting_customs,
            'time_orientation': self.time_orientation,
            'created_date': self.created_date.isoformat()
        }

@dataclass
class CulturalInteraction:
    """Represents a culturally-adapted interaction"""
    user_id: str
    detected_cultural_cues: List[str]
    applied_adaptations: List[str]
    communication_style_used: CommunicationStyle
    cultural_sensitivity_score: float
    interaction_success_indicators: List[str]
    learning_insights: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'detected_cultural_cues': self.detected_cultural_cues,
            'applied_adaptations': self.applied_adaptations,
            'communication_style_used': self.communication_style_used.value,
            'cultural_sensitivity_score': self.cultural_sensitivity_score,
            'interaction_success_indicators': self.interaction_success_indicators,
            'learning_insights': self.learning_insights,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class CulturalLearning:
    """Represents learned cultural insights"""
    insight_id: str
    cultural_context: str
    learned_pattern: str
    supporting_evidence: List[str]
    confidence_level: float
    generalizability: str  # "specific", "moderate", "general"
    practical_applications: List[str]
    discovered_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'insight_id': self.insight_id,
            'cultural_context': self.cultural_context,
            'learned_pattern': self.learned_pattern,
            'supporting_evidence': self.supporting_evidence,
            'confidence_level': self.confidence_level,
            'generalizability': self.generalizability,
            'practical_applications': self.practical_applications,
            'discovered_date': self.discovered_date.isoformat()
        }

class CulturalIntelligenceModule:
    """
    Advanced Cultural Intelligence Module that enables culturally-aware
    and sensitive interactions across diverse cultural contexts.
    """
    
    # Cultural indicators and patterns
    CULTURAL_INDICATORS = {
        'language_patterns': {
            'honorifics': ['sir', 'madam', 'sensei', 'ji', 'san', 'sama'],
            'formal_language': ['would you kindly', 'i humbly request', 'with respect'],
            'indirect_communication': ['perhaps', 'maybe', 'it seems', 'i wonder if'],
            'collective_pronouns': ['we', 'us', 'our family', 'our community']
        },
        'time_references': {
            'long_term': ['tradition', 'ancestors', 'generations', 'legacy', 'heritage'],
            'short_term': ['now', 'immediately', 'quick', 'fast', 'instant'],
            'cyclical': ['season', 'cycle', 'rhythm', 'pattern', 'recurring']
        },
        'value_indicators': {
            'family_focus': ['family', 'parents', 'children', 'relatives', 'household'],
            'education': ['learning', 'study', 'knowledge', 'wisdom', 'understanding'],
            'respect': ['respect', 'honor', 'dignity', 'courtesy', 'politeness'],
            'harmony': ['balance', 'peace', 'harmony', 'consensus', 'cooperation']
        }
    }
    
    # Predefined cultural profiles for major cultural contexts
    CULTURAL_PROFILES = {
        'east_asian_collectivist': CulturalProfile(
            id='east_asian_collectivist',
            name='East Asian Collectivist',
            context_type=CulturalContext.REGIONAL,
            cultural_dimensions={
                CulturalDimension.POWER_DISTANCE: 0.7,
                CulturalDimension.INDIVIDUALISM: 0.2,
                CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6,
                CulturalDimension.LONG_TERM_ORIENTATION: 0.8
            },
            communication_preferences=[CommunicationStyle.INDIRECT, CommunicationStyle.FORMAL, CommunicationStyle.HIGH_CONTEXT],
            core_values=[CulturalValue.RESPECT_FOR_ELDERS, CulturalValue.FAMILY_ORIENTATION, CulturalValue.HARMONY_SEEKING],
            interaction_patterns={
                'greeting': 'Respectful and formal approach',
                'disagreement': 'Indirect expression with face-saving',
                'praise': 'Modest acceptance, deflect to group'
            },
            sensitivity_areas=['direct criticism', 'individual spotlight', 'family honor'],
            greeting_customs=['bow', 'formal address', 'humble language'],
            time_orientation='polychronic',
            created_date=datetime.now()
        ),
        'western_individualist': CulturalProfile(
            id='western_individualist',
            name='Western Individualist',
            context_type=CulturalContext.REGIONAL,
            cultural_dimensions={
                CulturalDimension.POWER_DISTANCE: 0.3,
                CulturalDimension.INDIVIDUALISM: 0.8,
                CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.4,
                CulturalDimension.LONG_TERM_ORIENTATION: 0.5
            },
            communication_preferences=[CommunicationStyle.DIRECT, CommunicationStyle.INFORMAL, CommunicationStyle.LOW_CONTEXT],
            core_values=[CulturalValue.INNOVATION_OPENNESS, CulturalValue.EDUCATION_EMPHASIS],
            interaction_patterns={
                'greeting': 'Friendly and direct approach',
                'disagreement': 'Direct but respectful expression',
                'praise': 'Accept graciously, individual achievement focus'
            },
            sensitivity_areas=['overly formal hierarchy', 'group pressure'],
            greeting_customs=['handshake', 'direct eye contact', 'casual conversation'],
            time_orientation='monochronic',
            created_date=datetime.now()
        ),
        'high_power_distance': CulturalProfile(
            id='high_power_distance',
            name='High Power Distance Culture',
            context_type=CulturalContext.PROFESSIONAL,
            cultural_dimensions={
                CulturalDimension.POWER_DISTANCE: 0.9,
                CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.7
            },
            communication_preferences=[CommunicationStyle.FORMAL, CommunicationStyle.INDIRECT],
            core_values=[CulturalValue.RESPECT_FOR_ELDERS, CulturalValue.TRADITION_PRESERVATION],
            interaction_patterns={
                'greeting': 'Very formal with proper titles',
                'disagreement': 'Extremely indirect, private channels',
                'requests': 'Humble and deferential'
            },
            sensitivity_areas=['challenging authority', 'informal address', 'jumping hierarchy'],
            greeting_customs=['formal titles', 'proper protocol', 'respectful distance'],
            time_orientation='polychronic',
            created_date=datetime.now()
        )
    }
    
    # Communication adaptation templates
    ADAPTATION_TEMPLATES = {
        CommunicationStyle.FORMAL: {
            'greeting': 'I respectfully greet you and hope you are well.',
            'request': 'I would be honored if you could kindly consider...',
            'disagreement': 'With great respect, I wonder if we might also consider...',
            'gratitude': 'I am deeply grateful for your wisdom and guidance.'
        },
        CommunicationStyle.INFORMAL: {
            'greeting': 'Hi there! Hope you\'re doing great!',
            'request': 'Would you mind helping me with...',
            'disagreement': 'I see your point, but what about...',
            'gratitude': 'Thanks so much! Really appreciate it.'
        },
        CommunicationStyle.INDIRECT: {
            'disagreement': 'That\'s an interesting perspective. Perhaps we might also explore...',
            'request': 'I wonder if it might be possible to...',
            'criticism': 'This approach has merit, and there might be ways to enhance it further...'
        },
        CommunicationStyle.DIRECT: {
            'disagreement': 'I respectfully disagree because...',
            'request': 'Could you please help me with...',
            'criticism': 'Here\'s what could be improved...'
        }
    }
    
    def __init__(self):
        """Initialize the Cultural Intelligence Module"""
        self.cultural_profiles = self.CULTURAL_PROFILES.copy()
        self.user_cultural_contexts = {}  # user_id -> cultural context info
        self.interaction_history = {}     # user_id -> list of interactions
        self.cultural_learning = {}       # insight_id -> CulturalLearning
        self.adaptation_strategies = {}   # Strategy cache
        logger.info("Cultural Intelligence Module initialized")
    
    def detect_cultural_context(self, 
                              text: str, 
                              user_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect cultural context from text and user metadata
        
        Args:
            text: Input text to analyze
            user_metadata: Optional user metadata (location, language, etc.)
            
        Returns:
            Detected cultural context information
        """
        cultural_cues = []
        detected_dimensions = {}
        communication_style_indicators = []
        
        text_lower = text.lower()
        
        # Detect language patterns
        for pattern_type, patterns in self.CULTURAL_INDICATORS['language_patterns'].items():
            matches = [pattern for pattern in patterns if pattern in text_lower]
            if matches:
                cultural_cues.append(f"{pattern_type}: {matches}")
                
                # Map to cultural dimensions
                if pattern_type == 'honorifics':
                    detected_dimensions[CulturalDimension.POWER_DISTANCE] = 0.7
                    communication_style_indicators.append(CommunicationStyle.FORMAL)
                elif pattern_type == 'indirect_communication':
                    communication_style_indicators.append(CommunicationStyle.INDIRECT)
                elif pattern_type == 'collective_pronouns':
                    detected_dimensions[CulturalDimension.INDIVIDUALISM] = 0.3
        
        # Detect time orientation
        for time_type, indicators in self.CULTURAL_INDICATORS['time_references'].items():
            matches = [indicator for indicator in indicators if indicator in text_lower]
            if matches:
                cultural_cues.append(f"{time_type}_orientation: {matches}")
                
                if time_type == 'long_term':
                    detected_dimensions[CulturalDimension.LONG_TERM_ORIENTATION] = 0.8
        
        # Detect value systems
        detected_values = []
        for value_type, indicators in self.CULTURAL_INDICATORS['value_indicators'].items():
            matches = [indicator for indicator in indicators if indicator in text_lower]
            if matches:
                cultural_cues.append(f"{value_type}_values: {matches}")
                detected_values.append(value_type)
        
        # Analyze user metadata
        metadata_insights = {}
        if user_metadata:
            if 'location' in user_metadata:
                metadata_insights['geographic_context'] = user_metadata['location']
            if 'language' in user_metadata:
                metadata_insights['linguistic_context'] = user_metadata['language']
            if 'timezone' in user_metadata:
                metadata_insights['temporal_context'] = user_metadata['timezone']
        
        # Match against known cultural profiles
        profile_matches = self._match_cultural_profiles(detected_dimensions, communication_style_indicators, detected_values)
        
        return {
            'cultural_cues_detected': cultural_cues,
            'cultural_dimensions': {dim.value: score for dim, score in detected_dimensions.items()},
            'communication_style_indicators': [style.value for style in communication_style_indicators],
            'detected_values': detected_values,
            'metadata_insights': metadata_insights,
            'profile_matches': profile_matches,
            'confidence_level': self._calculate_detection_confidence(cultural_cues, detected_dimensions),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _match_cultural_profiles(self, 
                               dimensions: Dict[CulturalDimension, float],
                               communication_styles: List[CommunicationStyle],
                               values: List[str]) -> List[Dict[str, Any]]:
        """Match detected patterns against known cultural profiles"""
        matches = []
        
        for profile_id, profile in self.cultural_profiles.items():
            match_score = 0.0
            match_reasons = []
            
            # Compare cultural dimensions
            for dim, detected_score in dimensions.items():
                if dim in profile.cultural_dimensions:
                    profile_score = profile.cultural_dimensions[dim]
                    similarity = 1.0 - abs(detected_score - profile_score)
                    match_score += similarity * 0.4  # 40% weight for dimensions
                    if similarity > 0.7:
                        match_reasons.append(f"Similar {dim.value} orientation")
            
            # Compare communication styles
            style_overlap = len(set(communication_styles) & set(profile.communication_preferences))
            if style_overlap > 0:
                match_score += (style_overlap / len(profile.communication_preferences)) * 0.3  # 30% weight
                match_reasons.append(f"Communication style alignment ({style_overlap} matches)")
            
            # Compare values (simplified matching)
            value_keywords = {
                'family_focus': CulturalValue.FAMILY_ORIENTATION,
                'education': CulturalValue.EDUCATION_EMPHASIS,
                'respect': CulturalValue.RESPECT_FOR_ELDERS,
                'harmony': CulturalValue.HARMONY_SEEKING
            }
            
            value_matches = 0
            for value in values:
                if value in value_keywords and value_keywords[value] in profile.core_values:
                    value_matches += 1
            
            if value_matches > 0:
                match_score += (value_matches / len(profile.core_values)) * 0.3  # 30% weight
                match_reasons.append(f"Value system alignment ({value_matches} matches)")
            
            if match_score > 0.3:  # Threshold for meaningful match
                matches.append({
                    'profile_id': profile_id,
                    'profile_name': profile.name,
                    'match_score': match_score,
                    'match_reasons': match_reasons
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:3]  # Return top 3 matches
    
    def _calculate_detection_confidence(self, 
                                      cultural_cues: List[str], 
                                      dimensions: Dict[CulturalDimension, float]) -> float:
        """Calculate confidence in cultural context detection"""
        base_confidence = 0.5
        
        # More cues = higher confidence
        cue_bonus = min(len(cultural_cues) * 0.1, 0.3)
        
        # More dimensions detected = higher confidence
        dimension_bonus = min(len(dimensions) * 0.1, 0.2)
        
        total_confidence = base_confidence + cue_bonus + dimension_bonus
        return min(total_confidence, 1.0)
    
    def adapt_communication_style(self,
                                message: str,
                                target_style: CommunicationStyle,
                                cultural_context: Optional[Dict] = None) -> str:
        """
        Adapt communication style for cultural appropriateness
        
        Args:
            message: Original message
            target_style: Target communication style
            cultural_context: Optional cultural context information
            
        Returns:
            Culturally adapted message
        """
        if target_style not in self.ADAPTATION_TEMPLATES:
            return message  # No adaptation template available
        
        adaptations = self.ADAPTATION_TEMPLATES[target_style]
        adapted_message = message
        
        # Apply style-specific transformations
        if target_style == CommunicationStyle.FORMAL:
            adapted_message = self._apply_formal_adaptations(adapted_message)
        elif target_style == CommunicationStyle.INFORMAL:
            adapted_message = self._apply_informal_adaptations(adapted_message)
        elif target_style == CommunicationStyle.INDIRECT:
            adapted_message = self._apply_indirect_adaptations(adapted_message)
        elif target_style == CommunicationStyle.DIRECT:
            adapted_message = self._apply_direct_adaptations(adapted_message)
        
        # Apply cultural context specific adaptations
        if cultural_context and 'profile_matches' in cultural_context:
            best_match = cultural_context['profile_matches'][0] if cultural_context['profile_matches'] else None
            if best_match:
                adapted_message = self._apply_profile_specific_adaptations(
                    adapted_message, best_match['profile_id']
                )
        
        return adapted_message
    
    def _apply_formal_adaptations(self, message: str) -> str:
        """Apply formal communication adaptations"""
        # Replace casual language with formal equivalents
        replacements = {
            r'\bhi\b': 'greetings',
            r'\bhey\b': 'hello',
            r'\bthanks\b': 'thank you',
            r'\byou\'re\b': 'you are',
            r'\bcan\'t\b': 'cannot',
            r'\bwon\'t\b': 'will not',
            r'\bI\'m\b': 'I am'
        }
        
        adapted = message
        for pattern, replacement in replacements.items():
            adapted = re.sub(pattern, replacement, adapted, flags=re.IGNORECASE)
        
        # Add respectful framing
        if not adapted.startswith(('Please', 'I respectfully', 'With respect')):
            adapted = f"I respectfully share that {adapted.lower()}"
        
        return adapted
    
    def _apply_informal_adaptations(self, message: str) -> str:
        """Apply informal communication adaptations"""
        # Replace formal language with casual equivalents
        replacements = {
            r'\bgreetings\b': 'hi',
            r'\bI respectfully\b': 'I',
            r'\bwith great respect\b': '',
            r'\bI humbly\b': 'I',
            r'\bcannot\b': 'can\'t',
            r'\bwill not\b': 'won\'t'
        }
        
        adapted = message
        for pattern, replacement in replacements.items():
            adapted = re.sub(pattern, replacement, adapted, flags=re.IGNORECASE)
        
        return adapted.strip()
    
    def _apply_indirect_adaptations(self, message: str) -> str:
        """Apply indirect communication adaptations"""
        # Soften direct statements
        if message.startswith('You should'):
            message = message.replace('You should', 'You might consider')
        elif message.startswith('This is wrong'):
            message = message.replace('This is wrong', 'This approach might benefit from reconsideration')
        elif message.startswith('I disagree'):
            message = message.replace('I disagree', 'I wonder if we might explore alternative perspectives')
        
        # Add hedging language
        hedges = ['perhaps', 'it seems', 'I believe', 'in my view']
        if not any(hedge in message.lower() for hedge in hedges):
            message = f"It seems to me that {message.lower()}"
        
        return message
    
    def _apply_direct_adaptations(self, message: str) -> str:
        """Apply direct communication adaptations"""
        # Remove hedging language
        hedges_to_remove = ['perhaps', 'maybe', 'i think', 'it seems', 'i believe', 'in my opinion']
        adapted = message
        
        for hedge in hedges_to_remove:
            adapted = re.sub(f'\\b{hedge}\\b', '', adapted, flags=re.IGNORECASE)
        
        # Remove excessive politeness markers
        adapted = re.sub(r'\bwith great respect,?\s*', '', adapted, flags=re.IGNORECASE)
        adapted = re.sub(r'\bi humbly\s+', 'I ', adapted, flags=re.IGNORECASE)
        
        return adapted.strip()
    
    def _apply_profile_specific_adaptations(self, message: str, profile_id: str) -> str:
        """Apply adaptations specific to a cultural profile"""
        if profile_id not in self.cultural_profiles:
            return message
        
        profile = self.cultural_profiles[profile_id]
        adapted = message
        
        # Apply interaction patterns
        if 'greeting' in profile.interaction_patterns:
            if message.lower().startswith(('hello', 'hi', 'greetings')):
                adapted = profile.interaction_patterns['greeting'] + " " + adapted[adapted.find(' ')+1:]
        
        # Avoid sensitivity areas
        for sensitive_area in profile.sensitivity_areas:
            if sensitive_area in adapted.lower():
                # Apply softening for sensitive topics
                adapted = f"With cultural sensitivity in mind, {adapted}"
                break
        
        return adapted
    
    def analyze_cultural_sensitivity(self,
                                   message: str,
                                   cultural_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cultural sensitivity of a message
        
        Args:
            message: Message to analyze
            cultural_context: Cultural context information
            
        Returns:
            Cultural sensitivity analysis
        """
        sensitivity_score = 0.8  # Start with good baseline
        sensitivity_issues = []
        positive_elements = []
        
        message_lower = message.lower()
        
        # Check for cultural profile matches
        if 'profile_matches' in cultural_context and cultural_context['profile_matches']:
            best_match = cultural_context['profile_matches'][0]
            profile_id = best_match['profile_id']
            
            if profile_id in self.cultural_profiles:
                profile = self.cultural_profiles[profile_id]
                
                # Check for sensitivity areas
                for sensitive_area in profile.sensitivity_areas:
                    if sensitive_area.replace('_', ' ') in message_lower:
                        sensitivity_score -= 0.2
                        sensitivity_issues.append(f"Potential sensitivity around {sensitive_area}")
                
                # Check for positive cultural elements
                for value in profile.core_values:
                    value_keywords = {
                        CulturalValue.RESPECT_FOR_ELDERS: ['respect', 'wisdom', 'experience'],
                        CulturalValue.FAMILY_ORIENTATION: ['family', 'community', 'together'],
                        CulturalValue.HARMONY_SEEKING: ['harmony', 'balance', 'cooperation']
                    }
                    
                    if value in value_keywords:
                        for keyword in value_keywords[value]:
                            if keyword in message_lower:
                                sensitivity_score += 0.1
                                positive_elements.append(f"Shows awareness of {value.value}")
                                break
                
                # Check communication style alignment
                preferred_styles = profile.communication_preferences
                message_style = self._detect_message_style(message)
                
                if message_style in preferred_styles:
                    sensitivity_score += 0.1
                    positive_elements.append(f"Uses preferred {message_style.value} communication style")
                else:
                    sensitivity_score -= 0.1
                    sensitivity_issues.append(f"May not align with preferred communication style")
        
        # General sensitivity checks
        if any(word in message_lower for word in ['stereotype', 'always', 'never', 'all', 'typical']):
            sensitivity_score -= 0.15
            sensitivity_issues.append("Contains potential generalizations")
        
        if any(phrase in message_lower for phrase in ['i understand', 'i respect', 'cultural differences']):
            sensitivity_score += 0.1
            positive_elements.append("Shows cultural awareness")
        
        # Ensure score stays within bounds
        sensitivity_score = max(0.0, min(1.0, sensitivity_score))
        
        # Determine overall assessment
        if sensitivity_score >= 0.8:
            assessment = "Highly culturally sensitive"
        elif sensitivity_score >= 0.6:
            assessment = "Generally culturally appropriate"
        elif sensitivity_score >= 0.4:
            assessment = "Some cultural considerations needed"
        else:
            assessment = "Significant cultural sensitivity concerns"
        
        return {
            'sensitivity_score': sensitivity_score,
            'assessment': assessment,
            'sensitivity_issues': sensitivity_issues,
            'positive_elements': positive_elements,
            'recommendations': self._generate_sensitivity_recommendations(sensitivity_issues),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _detect_message_style(self, message: str) -> CommunicationStyle:
        """Detect the communication style of a message"""
        message_lower = message.lower()
        
        # Check for formal indicators
        formal_indicators = ['respectfully', 'humbly', 'kindly', 'would you', 'please allow me']
        if any(indicator in message_lower for indicator in formal_indicators):
            return CommunicationStyle.FORMAL
        
        # Check for indirect indicators
        indirect_indicators = ['perhaps', 'maybe', 'it seems', 'i wonder', 'might consider']
        if any(indicator in message_lower for indicator in indirect_indicators):
            return CommunicationStyle.INDIRECT
        
        # Check for informal indicators
        informal_indicators = ['hi', 'hey', 'thanks', 'cool', 'awesome', 'great']
        if any(indicator in message_lower for indicator in informal_indicators):
            return CommunicationStyle.INFORMAL
        
        # Default to direct if no other style detected
        return CommunicationStyle.DIRECT
    
    def _generate_sensitivity_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for improving cultural sensitivity"""
        recommendations = []
        
        for issue in issues:
            if 'sensitivity around' in issue:
                recommendations.append("Consider using softer language when discussing sensitive topics")
            elif 'generalizations' in issue:
                recommendations.append("Avoid broad generalizations; focus on specific situations or individuals")
            elif 'communication style' in issue:
                recommendations.append("Adapt communication style to match cultural preferences")
            elif 'stereotype' in issue:
                recommendations.append("Replace stereotypical language with more nuanced descriptions")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Continue demonstrating cultural awareness and sensitivity")
        
        return recommendations
    
    def learn_cultural_pattern(self,
                             user_id: str,
                             interaction_data: Dict[str, Any],
                             outcome_success: bool) -> Optional[CulturalLearning]:
        """
        Learn from cultural interactions to improve future adaptations
        
        Args:
            user_id: User identifier
            interaction_data: Data about the cultural interaction
            outcome_success: Whether the interaction was successful
            
        Returns:
            New cultural learning insight if significant pattern found
        """
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        
        # Store interaction
        self.interaction_history[user_id].append({
            'timestamp': datetime.now(),
            'data': interaction_data,
            'success': outcome_success
        })
        
        # Look for patterns after sufficient interactions
        if len(self.interaction_history[user_id]) >= 3:
            return self._analyze_interaction_patterns(user_id)
        
        return None
    
    def _analyze_interaction_patterns(self, user_id: str) -> Optional[CulturalLearning]:
        """Analyze interaction patterns to extract cultural insights"""
        interactions = self.interaction_history[user_id]
        recent_interactions = interactions[-5:]  # Analyze recent interactions
        
        # Look for consistent patterns
        successful_patterns = []
        unsuccessful_patterns = []
        
        for interaction in recent_interactions:
            if interaction['success']:
                if 'communication_style_used' in interaction['data']:
                    successful_patterns.append(interaction['data']['communication_style_used'])
            else:
                if 'communication_style_used' in interaction['data']:
                    unsuccessful_patterns.append(interaction['data']['communication_style_used'])
        
        # Identify learning insights
        if len(successful_patterns) >= 2:
            most_successful_style = max(set(successful_patterns), key=successful_patterns.count)
            
            # Check if this is a new insight
            pattern_description = f"User {user_id} responds well to {most_successful_style} communication style"
            
            # Create learning insight
            insight_id = str(uuid.uuid4())
            learning = CulturalLearning(
                insight_id=insight_id,
                cultural_context=f"user_{user_id}",
                learned_pattern=pattern_description,
                supporting_evidence=[f"Successful interaction {i+1}" for i in range(len(successful_patterns))],
                confidence_level=len(successful_patterns) / len(recent_interactions),
                generalizability="specific",
                practical_applications=[
                    f"Use {most_successful_style} style for user {user_id}",
                    "Apply similar style to users with similar cultural indicators"
                ],
                discovered_date=datetime.now()
            )
            
            self.cultural_learning[insight_id] = learning
            logger.info(f"Learned cultural pattern: {pattern_description}")
            return learning
        
        return None
    
    def get_cultural_recommendations(self, 
                                   user_id: Optional[str] = None,
                                   cultural_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Get cultural communication recommendations
        
        Args:
            user_id: Optional user identifier
            cultural_context: Optional cultural context
            
        Returns:
            List of cultural recommendations
        """
        recommendations = []
        
        # User-specific recommendations
        if user_id and user_id in self.interaction_history:
            user_interactions = self.interaction_history[user_id]
            if user_interactions:
                last_interaction = user_interactions[-1]
                if not last_interaction['success']:
                    recommendations.append({
                        'type': 'style_adjustment',
                        'priority': 'high',
                        'message': 'Consider adjusting communication style based on previous interaction',
                        'suggestion': 'Try a more formal or indirect approach'
                    })
        
        # Context-specific recommendations
        if cultural_context and 'profile_matches' in cultural_context:
            best_match = cultural_context['profile_matches'][0] if cultural_context['profile_matches'] else None
            if best_match:
                profile_id = best_match['profile_id']
                if profile_id in self.cultural_profiles:
                    profile = self.cultural_profiles[profile_id]
                    
                    recommendations.append({
                        'type': 'communication_style',
                        'priority': 'medium',
                        'message': f'Consider using {", ".join([style.value for style in profile.communication_preferences])} communication',
                        'profile': profile.name
                    })
                    
                    if profile.sensitivity_areas:
                        recommendations.append({
                            'type': 'sensitivity_awareness',
                            'priority': 'high',
                            'message': f'Be mindful of sensitivity around: {", ".join(profile.sensitivity_areas)}',
                            'areas': profile.sensitivity_areas
                        })
        
        # General cultural intelligence recommendations
        recommendations.append({
            'type': 'general_awareness',
            'priority': 'low',
            'message': 'Continue demonstrating cultural curiosity and respect for diverse perspectives',
            'practices': [
                'Ask about cultural preferences when appropriate',
                'Listen for cultural cues in conversations',
                'Adapt communication style to match user preferences'
            ]
        })
        
        return recommendations
    
    def get_cultural_insights(self) -> Dict[str, Any]:
        """Get comprehensive cultural intelligence insights"""
        total_interactions = sum(len(history) for history in self.interaction_history.values())
        total_users = len(self.interaction_history)
        total_learning_insights = len(self.cultural_learning)
        
        # Analyze success rates by cultural context
        success_by_style = {}
        for user_id, interactions in self.interaction_history.items():
            for interaction in interactions:
                if 'communication_style_used' in interaction['data']:
                    style = interaction['data']['communication_style_used']
                    if style not in success_by_style:
                        success_by_style[style] = {'total': 0, 'successful': 0}
                    success_by_style[style]['total'] += 1
                    if interaction['success']:
                        success_by_style[style]['successful'] += 1
        
        # Calculate success rates
        style_success_rates = {}
        for style, stats in success_by_style.items():
            if stats['total'] > 0:
                style_success_rates[style] = stats['successful'] / stats['total']
        
        # Most effective communication styles
        most_effective_styles = sorted(
            style_success_rates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return {
            'total_cultural_interactions': total_interactions,
            'users_with_cultural_context': total_users,
            'cultural_learning_insights': total_learning_insights,
            'cultural_profiles_available': len(self.cultural_profiles),
            'communication_style_effectiveness': {
                style: f"{rate:.1%}" for style, rate in most_effective_styles
            },
            'recent_cultural_learning': [
                learning.learned_pattern for learning in 
                sorted(self.cultural_learning.values(), 
                      key=lambda x: x.discovered_date, reverse=True)[:3]
            ],
            'cultural_intelligence_score': self._calculate_cultural_intelligence_score()
        }
    
    def _calculate_cultural_intelligence_score(self) -> float:
        """Calculate overall cultural intelligence score"""
        score = 0.5  # Base score
        
        # Points for diverse interactions
        if len(self.interaction_history) > 5:
            score += 0.1
        
        # Points for learning insights
        if len(self.cultural_learning) > 0:
            score += min(len(self.cultural_learning) * 0.05, 0.2)
        
        # Points for successful adaptations
        total_interactions = sum(len(history) for history in self.interaction_history.values())
        successful_interactions = sum(
            sum(1 for interaction in history if interaction['success'])
            for history in self.interaction_history.values()
        )
        
        if total_interactions > 0:
            success_rate = successful_interactions / total_interactions
            score += success_rate * 0.3
        
        return min(score, 1.0)