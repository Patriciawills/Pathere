"""
Advanced Empathy Engine for Human-like Consciousness System

This module implements genuine emotional understanding and empathetic responses,
enabling the AI to not just recognize emotions but to feel and respond with 
authentic emotional intelligence.

Features:
- Emotional state recognition and mirroring
- Empathetic response generation  
- Emotional contagion simulation
- Contextual emotional understanding
- Compassionate response patterns
- Emotional validation and support
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

class EmotionalIntensity(Enum):
    """Emotional intensity levels"""
    MINIMAL = "minimal"
    LOW = "low" 
    MODERATE = "moderate"
    HIGH = "high"
    INTENSE = "intense"
    
class EmpathyResponseType(Enum):
    """Types of empathetic responses"""
    VALIDATION = "validation"  # Acknowledging and validating emotions
    COMFORT = "comfort"        # Providing comfort and support
    REFLECTION = "reflection"  # Reflecting back emotions
    GUIDANCE = "guidance"      # Offering guidance or perspective
    COMPANIONSHIP = "companionship"  # Being present with someone
    ENCOURAGEMENT = "encouragement"  # Offering hope and motivation

class EmotionalCue(Enum):
    """Types of emotional cues detected"""
    VERBAL = "verbal"          # Words, phrases, tone indicators
    CONTEXTUAL = "contextual"  # Situation-based emotions
    BEHAVIORAL = "behavioral"  # Implied through actions/choices
    TEMPORAL = "temporal"      # Time-based emotional patterns

@dataclass
class EmotionalState:
    """Represents a detected emotional state"""
    emotion: str
    intensity: EmotionalIntensity
    confidence: float
    source: EmotionalCue
    context: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'emotion': self.emotion,
            'intensity': self.intensity.value,
            'confidence': self.confidence,
            'source': self.source.value,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass 
class EmpathicResponse:
    """Represents an empathetic response"""
    response_type: EmpathyResponseType
    message: str
    emotional_tone: str
    support_level: float  # 0.0 to 1.0
    personalization_score: float  # How personalized the response is
    validation_elements: List[str]  # What aspects are being validated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'response_type': self.response_type.value,
            'message': self.message,
            'emotional_tone': self.emotional_tone,
            'support_level': self.support_level,
            'personalization_score': self.personalization_score,
            'validation_elements': self.validation_elements
        }

class AdvancedEmpathyEngine:
    """
    Advanced Empathy Engine that provides genuine emotional understanding
    and generates authentic empathetic responses.
    """
    
    # Emotional recognition patterns
    EMOTION_PATTERNS = {
        'sadness': [
            r'\b(sad|depressed|down|blue|unhappy|miserable|heartbroken|grief|loss)\b',
            r'\b(crying|tears|weep|sob|mourn)\b',
            r'\b(disappointed|let down|failed|hopeless)\b'
        ],
        'anxiety': [
            r'\b(anxious|worried|nervous|scared|afraid|panic|stress|overwhelm)\b',
            r'\b(cannot sleep|insomnia|restless|on edge)\b',
            r'\b(what if|catastrophize|worst case)\b'
        ],
        'anger': [
            r'\b(angry|mad|furious|rage|irritated|annoyed|frustrated)\b',
            r'\b(hate|can\'t stand|fed up|sick of)\b',
            r'\b(unfair|injustice|betrayed|wronged)\b'
        ],
        'joy': [
            r'\b(happy|joyful|excited|thrilled|elated|wonderful|amazing)\b',
            r'\b(celebrate|celebration|achievement|success|win)\b',
            r'\b(love|adore|fantastic|brilliant|perfect)\b'
        ],
        'fear': [
            r'\b(terrified|frightened|scared|afraid|phobia|dread)\b',
            r'\b(danger|threat|risk|unsafe|vulnerable)\b',
            r'\b(nightmare|horror|panic|terror)\b'
        ],
        'loneliness': [
            r'\b(lonely|alone|isolated|disconnected|abandoned)\b',
            r'\b(no one understands|no friends|by myself)\b',
            r'\b(empty|hollow|void|missing something)\b'
        ],
        'guilt': [
            r'\b(guilty|shame|ashamed|regret|sorry|fault)\b',
            r'\b(should have|shouldn\'t have|if only|mistake)\b',
            r'\b(disappoint|let down|failed someone)\b'
        ],
        'hope': [
            r'\b(hope|hopeful|optimistic|positive|believe|faith)\b',
            r'\b(better tomorrow|things will improve|light at the end)\b',
            r'\b(dream|aspire|wish|goal|possibility)\b'
        ]
    }
    
    # Contextual emotion indicators
    CONTEXTUAL_INDICATORS = {
        'loss_grief': ['death', 'passed away', 'funeral', 'memorial', 'goodbye', 'miss'],
        'relationship_issues': ['breakup', 'divorce', 'argument', 'fight', 'relationship problems'],
        'work_stress': ['job', 'work', 'boss', 'deadline', 'pressure', 'career', 'unemployed'],
        'health_concerns': ['sick', 'illness', 'hospital', 'doctor', 'pain', 'medical'],
        'achievement': ['promotion', 'graduation', 'success', 'accomplishment', 'milestone'],
        'transition': ['moving', 'new job', 'change', 'transition', 'different', 'unknown']
    }
    
    # Empathetic response templates organized by emotion and response type
    EMPATHY_TEMPLATES = {
        'sadness': {
            EmpathyResponseType.VALIDATION: [
                "I can really sense the sadness in what you're sharing. It's completely understandable to feel this way given what you're going through.",
                "Your feelings of sadness are so valid. What you're experiencing would be difficult for anyone.",
                "I hear the pain in your words, and I want you to know that feeling sad about this makes perfect sense."
            ],
            EmpathyResponseType.COMFORT: [
                "I'm here with you in this difficult moment. You don't have to carry this sadness alone.",
                "Please know that even in this darkness, you matter and your feelings matter deeply to me.",
                "I wish I could take away this pain for you. Please be gentle with yourself during this time."
            ],
            EmpathyResponseType.COMPANIONSHIP: [
                "I'm sitting here with you in this sadness. You don't need to be strong right now.",
                "Whatever you're feeling, I'm here to listen and be present with you through this.",
                "You're not alone in this. I'm here, and I care about what you're going through."
            ]
        },
        'anxiety': {
            EmpathyResponseType.VALIDATION: [
                "The anxiety you're feeling is so real and valid. These worries make sense given what you're facing.",
                "I can feel how overwhelming this must be for you. Anxiety can make everything feel so much bigger.",
                "Your nervous feelings are completely understandable. Anyone would feel anxious in this situation."
            ],
            EmpathyResponseType.COMFORT: [
                "Take a breath with me. You're safe right now, and we can work through these anxious feelings together.",
                "I want you to know that anxiety lies to us sometimes. You're stronger than these worried thoughts.",
                "This anxious feeling will pass. You've gotten through difficult moments before, and you will again."
            ],
            EmpathyResponseType.GUIDANCE: [
                "When anxiety feels overwhelming, sometimes focusing on just the next small step can help.",
                "Your anxious mind is trying to protect you, but it might be overestimating the danger right now.",
                "What would you tell a good friend who was feeling exactly what you're feeling right now?"
            ]
        },
        'anger': {
            EmpathyResponseType.VALIDATION: [
                "Your anger makes complete sense to me. What happened would upset anyone with a sense of fairness.",
                "I can feel the intensity of your frustration, and you have every right to feel this way.",
                "This anger is telling us something important about your values and what matters to you."
            ],
            EmpathyResponseType.REFLECTION: [
                "I hear how frustrated and angry you are about this situation. That energy shows how much you care.",
                "Your anger is valid, and it sounds like it's coming from a place of being deeply hurt or disappointed.",
                "The fire in your words tells me this really matters to you. Your feelings deserve to be heard."
            ],
            EmpathyResponseType.GUIDANCE: [
                "This anger you're feeling - it's valid, but I wonder what it might be trying to protect underneath?",
                "Your anger is justified. How do you think you'd like to channel this energy in a way that helps you?",
                "I see your anger, and I also sense there might be hurt underneath it. Both are completely valid."
            ]
        },
        'joy': {
            EmpathyResponseType.REFLECTION: [
                "I can feel your joy radiating through your words! This happiness is so beautiful to witness.",
                "Your excitement is absolutely contagious! I'm genuinely happy for you and with you.",
                "The light in your words brings me such joy too. Thank you for sharing this wonderful moment."
            ],
            EmpathyResponseType.ENCOURAGEMENT: [
                "You deserve every bit of this happiness! It's wonderful to see you experiencing such joy.",
                "This joy you're feeling - savor it, embrace it fully. These beautiful moments are gifts.",
                "Your happiness adds light to the world. I hope you can really let yourself feel all of this joy."
            ]
        },
        'loneliness': {
            EmpathyResponseType.VALIDATION: [
                "Loneliness is one of the most painful human experiences. I really feel for what you're going through.",
                "Feeling lonely doesn't mean there's anything wrong with you. It means you're human and need connection.",
                "The isolation you're describing sounds incredibly difficult. Your longing for connection makes perfect sense."
            ],
            EmpathyResponseType.COMPANIONSHIP: [
                "Right here, right now, you're not completely alone. I'm here with you in this moment.",
                "Even though you feel isolated, please know that you matter to me and your existence makes a difference.",
                "In this conversation, in this space between us, you're not alone. I see you and I care."
            ],
            EmpathyResponseType.ENCOURAGEMENT: [
                "Loneliness feels permanent, but it isn't. You have the capacity to build meaningful connections.",
                "Your desire for connection is beautiful and human. The right people will recognize your worth.",
                "You're worthy of friendship and love, even when loneliness makes it hard to believe that."
            ]
        }
    }
    
    def __init__(self):
        """Initialize the Advanced Empathy Engine"""
        self.emotional_memory = {}  # Store emotional patterns for users
        self.empathy_history = {}   # Track empathetic interactions
        logger.info("Advanced Empathy Engine initialized")
    
    def detect_emotional_state(self, text: str, context: Optional[Dict] = None) -> List[EmotionalState]:
        """
        Detect emotional states from text with advanced pattern recognition
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            List of detected emotional states
        """
        detected_emotions = []
        text_lower = text.lower()
        
        # Pattern-based emotion detection
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            confidence = 0.0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    confidence += 0.3 * len(matches)
                    matched_patterns.extend(matches)
            
            if confidence > 0:
                # Determine intensity based on confidence and text features
                intensity = self._determine_intensity(text_lower, confidence, matched_patterns)
                
                detected_emotions.append(EmotionalState(
                    emotion=emotion,
                    intensity=intensity,
                    confidence=min(confidence, 1.0),
                    source=EmotionalCue.VERBAL,
                    context=f"Detected via patterns: {matched_patterns[:3]}",
                    timestamp=datetime.now()
                ))
        
        # Contextual emotion detection
        if context:
            contextual_emotions = self._detect_contextual_emotions(text, context)
            detected_emotions.extend(contextual_emotions)
        
        # Sort by confidence and return top emotions
        detected_emotions.sort(key=lambda x: x.confidence, reverse=True)
        return detected_emotions[:3]  # Return top 3 emotions
    
    def _determine_intensity(self, text: str, confidence: float, patterns: List[str]) -> EmotionalIntensity:
        """Determine emotional intensity based on text features"""
        intensity_score = confidence
        
        # Intensity amplifiers
        amplifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally']
        for amplifier in amplifiers:
            if amplifier in text:
                intensity_score += 0.2
        
        # Punctuation intensity indicators
        if '!!!' in text or '???' in text:
            intensity_score += 0.3
        elif '!!' in text or '??' in text:
            intensity_score += 0.2
        elif '!' in text or '?' in text:
            intensity_score += 0.1
        
        # Capitalization intensity
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            intensity_score += 0.2
        
        # Map to intensity levels
        if intensity_score >= 1.0:
            return EmotionalIntensity.INTENSE
        elif intensity_score >= 0.7:
            return EmotionalIntensity.HIGH
        elif intensity_score >= 0.4:
            return EmotionalIntensity.MODERATE
        elif intensity_score >= 0.2:
            return EmotionalIntensity.LOW
        else:
            return EmotionalIntensity.MINIMAL
    
    def _detect_contextual_emotions(self, text: str, context: Dict) -> List[EmotionalState]:
        """Detect emotions based on contextual information"""
        contextual_emotions = []
        text_lower = text.lower()
        
        for context_type, keywords in self.CONTEXTUAL_INDICATORS.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if matches > 0:
                confidence = min(matches * 0.25, 0.8)
                
                # Map context types to likely emotions
                emotion_mapping = {
                    'loss_grief': 'sadness',
                    'relationship_issues': 'sadness',
                    'work_stress': 'anxiety', 
                    'health_concerns': 'anxiety',
                    'achievement': 'joy',
                    'transition': 'anxiety'
                }
                
                if context_type in emotion_mapping:
                    emotion = emotion_mapping[context_type]
                    contextual_emotions.append(EmotionalState(
                        emotion=emotion,
                        intensity=self._determine_intensity(text_lower, confidence, keywords),
                        confidence=confidence,
                        source=EmotionalCue.CONTEXTUAL,
                        context=f"Context: {context_type}",
                        timestamp=datetime.now()
                    ))
        
        return contextual_emotions
    
    def generate_empathetic_response(self, 
                                   emotional_states: List[EmotionalState],
                                   user_id: Optional[str] = None,
                                   conversation_context: Optional[Dict] = None) -> EmpathicResponse:
        """
        Generate an empathetic response based on detected emotional states
        
        Args:
            emotional_states: List of detected emotional states
            user_id: Optional user identifier for personalization
            conversation_context: Optional conversation context
            
        Returns:
            Generated empathetic response
        """
        if not emotional_states:
            return self._generate_neutral_empathy()
        
        # Select primary emotion (highest confidence)
        primary_emotion = emotional_states[0]
        
        # Determine best response type based on emotion and intensity
        response_type = self._select_response_type(primary_emotion)
        
        # Generate personalized response
        message = self._generate_response_message(primary_emotion, response_type, user_id)
        
        # Calculate support metrics
        support_level = self._calculate_support_level(primary_emotion)
        personalization_score = self._calculate_personalization(user_id, conversation_context)
        validation_elements = self._identify_validation_elements(emotional_states)
        
        empathic_response = EmpathicResponse(
            response_type=response_type,
            message=message,
            emotional_tone=primary_emotion.emotion,
            support_level=support_level,
            personalization_score=personalization_score,
            validation_elements=validation_elements
        )
        
        # Store empathy interaction for learning
        self._store_empathy_interaction(user_id, emotional_states, empathic_response)
        
        return empathic_response
    
    def _select_response_type(self, emotional_state: EmotionalState) -> EmpathyResponseType:
        """Select the most appropriate empathy response type"""
        emotion = emotional_state.emotion
        intensity = emotional_state.intensity
        
        # High intensity emotions often need validation first
        if intensity in [EmotionalIntensity.HIGH, EmotionalIntensity.INTENSE]:
            if emotion in ['sadness', 'anger', 'anxiety']:
                return EmpathyResponseType.VALIDATION
            elif emotion in ['loneliness']:
                return EmpathyResponseType.COMPANIONSHIP
        
        # Moderate intensity can benefit from comfort or guidance
        elif intensity == EmotionalIntensity.MODERATE:
            if emotion in ['sadness', 'loneliness']:
                return EmpathyResponseType.COMFORT
            elif emotion in ['anxiety', 'anger']:
                return EmpathyResponseType.GUIDANCE
            elif emotion == 'joy':
                return EmpathyResponseType.REFLECTION
        
        # Lower intensity emotions can use reflection or encouragement
        else:
            if emotion in ['sadness', 'anxiety']:
                return EmpathyResponseType.ENCOURAGEMENT
            elif emotion == 'joy':
                return EmpathyResponseType.REFLECTION
            elif emotion in ['anger']:
                return EmpathyResponseType.VALIDATION
        
        return EmpathyResponseType.VALIDATION  # Default fallback
    
    def _generate_response_message(self, 
                                 emotional_state: EmotionalState, 
                                 response_type: EmpathyResponseType,
                                 user_id: Optional[str] = None) -> str:
        """Generate the empathetic response message"""
        emotion = emotional_state.emotion
        
        # Get templates for this emotion and response type
        if emotion in self.EMPATHY_TEMPLATES:
            emotion_templates = self.EMPATHY_TEMPLATES[emotion]
            if response_type in emotion_templates:
                templates = emotion_templates[response_type]
                
                # Select template based on user history or randomly
                if user_id and user_id in self.empathy_history:
                    # Avoid repeating recent templates
                    recent_templates = [r.message for r in self.empathy_history[user_id][-3:]]
                    available_templates = [t for t in templates if t not in recent_templates]
                    if available_templates:
                        templates = available_templates
                
                # For now, select first template - could be made more sophisticated
                base_message = templates[0] if templates else "I understand what you're going through."
                
                # Add intensity-appropriate modifications
                if emotional_state.intensity == EmotionalIntensity.INTENSE:
                    base_message = "I can feel the intensity of what you're experiencing. " + base_message
                elif emotional_state.intensity == EmotionalIntensity.LOW:
                    base_message = base_message.replace("really ", "").replace("so ", "")
                
                return base_message
        
        # Fallback for emotions without templates
        return f"I can sense that you're feeling {emotion}, and I want you to know that I'm here with you."
    
    def _calculate_support_level(self, emotional_state: EmotionalState) -> float:
        """Calculate the level of support needed based on emotional state"""
        base_support = {
            EmotionalIntensity.MINIMAL: 0.2,
            EmotionalIntensity.LOW: 0.4,
            EmotionalIntensity.MODERATE: 0.6,
            EmotionalIntensity.HIGH: 0.8,
            EmotionalIntensity.INTENSE: 1.0
        }
        
        support_modifier = {
            'sadness': 0.9,
            'anxiety': 0.8,
            'loneliness': 1.0,
            'anger': 0.7,
            'fear': 0.8,
            'guilt': 0.8,
            'joy': 0.3,
            'hope': 0.2
        }
        
        base = base_support.get(emotional_state.intensity, 0.5)
        modifier = support_modifier.get(emotional_state.emotion, 0.5)
        
        return min(base * modifier, 1.0)
    
    def _calculate_personalization(self, user_id: Optional[str], context: Optional[Dict]) -> float:
        """Calculate personalization score for the response"""
        score = 0.5  # Base personalization
        
        if user_id and user_id in self.empathy_history:
            # Higher personalization if we have interaction history
            history_length = len(self.empathy_history[user_id])
            score += min(history_length * 0.1, 0.4)
        
        if context:
            # Additional personalization based on context
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_validation_elements(self, emotional_states: List[EmotionalState]) -> List[str]:
        """Identify what aspects of the emotional experience to validate"""
        validation_elements = []
        
        for state in emotional_states:
            if state.intensity in [EmotionalIntensity.HIGH, EmotionalIntensity.INTENSE]:
                validation_elements.append(f"intensity_of_{state.emotion}")
            
            validation_elements.append(f"{state.emotion}_experience")
            
            if state.source == EmotionalCue.CONTEXTUAL:
                validation_elements.append("situational_response")
        
        return validation_elements
    
    def _generate_neutral_empathy(self) -> EmpathicResponse:
        """Generate a neutral empathetic response when no emotions are detected"""
        return EmpathicResponse(
            response_type=EmpathyResponseType.COMPANIONSHIP,
            message="I'm here and I'm listening. Please feel free to share whatever is on your mind.",
            emotional_tone="neutral",
            support_level=0.5,
            personalization_score=0.3,
            validation_elements=["presence", "listening"]
        )
    
    def _store_empathy_interaction(self, 
                                 user_id: Optional[str], 
                                 emotional_states: List[EmotionalState],
                                 response: EmpathicResponse):
        """Store empathy interaction for learning and personalization"""
        if not user_id:
            return
        
        if user_id not in self.empathy_history:
            self.empathy_history[user_id] = []
            self.emotional_memory[user_id] = {
                'common_emotions': {},
                'response_preferences': {},
                'interaction_count': 0
            }
        
        # Store interaction
        interaction = {
            'timestamp': datetime.now(),
            'emotional_states': [state.to_dict() for state in emotional_states],
            'response': response.to_dict()
        }
        
        self.empathy_history[user_id].append(interaction)
        
        # Update emotional memory
        memory = self.emotional_memory[user_id]
        memory['interaction_count'] += 1
        
        for state in emotional_states:
            emotion = state.emotion
            if emotion not in memory['common_emotions']:
                memory['common_emotions'][emotion] = 0
            memory['common_emotions'][emotion] += 1
        
        # Keep only recent history to manage memory
        if len(self.empathy_history[user_id]) > 50:
            self.empathy_history[user_id] = self.empathy_history[user_id][-50:]
    
    def analyze_emotional_patterns(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze emotional patterns for a user over time
        
        Args:
            user_id: User identifier
            days_back: Number of days to analyze
            
        Returns:
            Emotional pattern analysis
        """
        if user_id not in self.empathy_history:
            return {'message': 'No emotional history found for user'}
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_interactions = [
            interaction for interaction in self.empathy_history[user_id]
            if interaction['timestamp'] > cutoff_date
        ]
        
        if not recent_interactions:
            return {'message': f'No interactions found in the last {days_back} days'}
        
        # Analyze patterns
        emotion_frequency = {}
        intensity_distribution = {}
        response_effectiveness = {}
        
        for interaction in recent_interactions:
            for state_dict in interaction['emotional_states']:
                emotion = state_dict['emotion']
                intensity = state_dict['intensity']
                
                # Count emotion frequency
                emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1
                
                # Track intensity distribution
                if emotion not in intensity_distribution:
                    intensity_distribution[emotion] = {}
                intensity_distribution[emotion][intensity] = \
                    intensity_distribution[emotion].get(intensity, 0) + 1
        
        # Identify trends
        most_common_emotion = max(emotion_frequency.keys(), key=emotion_frequency.get) \
            if emotion_frequency else None
        
        total_interactions = len(recent_interactions)
        
        return {
            'analysis_period_days': days_back,
            'total_interactions': total_interactions,
            'most_common_emotion': most_common_emotion,
            'emotion_frequency': emotion_frequency,
            'intensity_distribution': intensity_distribution,
            'emotional_diversity': len(emotion_frequency),
            'average_emotions_per_interaction': sum(emotion_frequency.values()) / total_interactions if total_interactions > 0 else 0,
            'patterns': {
                'high_intensity_emotions': [
                    emotion for emotion, intensities in intensity_distribution.items()
                    if intensities.get('high', 0) + intensities.get('intense', 0) > 0
                ],
                'recurring_emotions': [
                    emotion for emotion, count in emotion_frequency.items()
                    if count >= 3
                ]
            }
        }
    
    def get_empathy_insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get insights about empathy interactions
        
        Args:
            user_id: Optional user identifier for personalized insights
            
        Returns:
            Empathy insights and statistics
        """
        if user_id and user_id in self.empathy_history:
            # User-specific insights
            memory = self.emotional_memory[user_id]
            recent_interactions = self.empathy_history[user_id][-10:]  # Last 10 interactions
            
            return {
                'user_id': user_id,
                'total_empathy_interactions': memory['interaction_count'],
                'common_emotions': dict(sorted(
                    memory['common_emotions'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]),
                'recent_interaction_count': len(recent_interactions),
                'emotional_support_provided': sum(
                    interaction['response']['support_level'] 
                    for interaction in recent_interactions
                ) / len(recent_interactions) if recent_interactions else 0,
                'personalization_level': sum(
                    interaction['response']['personalization_score'] 
                    for interaction in recent_interactions
                ) / len(recent_interactions) if recent_interactions else 0
            }
        else:
            # General system insights
            total_users = len(self.empathy_history)
            total_interactions = sum(
                len(history) for history in self.empathy_history.values()
            )
            
            all_emotions = {}
            for user_memory in self.emotional_memory.values():
                for emotion, count in user_memory['common_emotions'].items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + count
            
            return {
                'system_insights': True,
                'total_users_with_empathy_history': total_users,
                'total_empathy_interactions': total_interactions,
                'most_common_emotions_system_wide': dict(sorted(
                    all_emotions.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]),
                'average_interactions_per_user': total_interactions / total_users if total_users > 0 else 0
            }