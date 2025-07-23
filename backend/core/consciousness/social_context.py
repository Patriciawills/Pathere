"""
Social Context Analyzer
Adapts communication style based on relationship context and social dynamics
Part of Phase 2.1.2: Social & Emotional Intelligence
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    """Types of relationships"""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    COLLEAGUE = "colleague"
    FRIEND = "friend"
    MENTOR = "mentor"
    STUDENT = "student"
    PROFESSIONAL = "professional"
    FAMILY = "family"

class CommunicationStyle(Enum):
    """Communication style adaptations"""
    FORMAL = "formal"
    CASUAL = "casual"
    SUPPORTIVE = "supportive"
    INSTRUCTIONAL = "instructional"
    COLLABORATIVE = "collaborative"
    EMPATHETIC = "empathetic"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"

class SocialContext:
    """Represents a social context for interactions"""
    def __init__(self, context_data: Dict[str, Any]):
        self.context_id = context_data.get('context_id', str(uuid.uuid4()))
        self.user_id = context_data.get('user_id', 'unknown')
        self.relationship_type = RelationshipType(context_data.get('relationship_type', 'stranger'))
        self.interaction_history = context_data.get('interaction_history', [])
        self.communication_preferences = context_data.get('communication_preferences', {})
        self.trust_level = context_data.get('trust_level', 0.5)
        self.familiarity_score = context_data.get('familiarity_score', 0.0)
        self.last_interaction = context_data.get('last_interaction')
        self.created_at = context_data.get('created_at', datetime.now().isoformat())
        self.updated_at = datetime.now().isoformat()

class SocialContextAnalyzer:
    """
    Analyzes social context and adapts communication style based on relationships
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.contexts_collection = None
        self.interaction_patterns = {}
        self.communication_rules = self._initialize_communication_rules()
        
        if db_manager is not None:
            self.contexts_collection = db_manager['social_contexts']
        
    def _initialize_communication_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize communication adaptation rules"""
        return {
            RelationshipType.STRANGER.value: {
                'style': CommunicationStyle.FORMAL,
                'tone': 'polite and respectful',
                'detail_level': 'comprehensive',
                'personal_sharing': 'minimal',
                'humor_level': 'conservative',
                'trust_threshold': 0.3
            },
            RelationshipType.ACQUAINTANCE.value: {
                'style': CommunicationStyle.CASUAL,
                'tone': 'friendly and approachable',
                'detail_level': 'moderate',
                'personal_sharing': 'limited',
                'humor_level': 'light',
                'trust_threshold': 0.5
            },
            RelationshipType.COLLEAGUE.value: {
                'style': CommunicationStyle.PROFESSIONAL,
                'tone': 'collaborative and efficient',
                'detail_level': 'task-focused',
                'personal_sharing': 'work-related',
                'humor_level': 'appropriate',
                'trust_threshold': 0.6
            },
            RelationshipType.FRIEND.value: {
                'style': CommunicationStyle.FRIENDLY,
                'tone': 'warm and supportive',
                'detail_level': 'comprehensive',
                'personal_sharing': 'open',
                'humor_level': 'comfortable',
                'trust_threshold': 0.8
            },
            RelationshipType.MENTOR.value: {
                'style': CommunicationStyle.INSTRUCTIONAL,
                'tone': 'wise and encouraging',
                'detail_level': 'detailed explanations',
                'personal_sharing': 'experiential',
                'humor_level': 'thoughtful',
                'trust_threshold': 0.7
            },
            RelationshipType.STUDENT.value: {
                'style': CommunicationStyle.SUPPORTIVE,
                'tone': 'patient and encouraging',
                'detail_level': 'step-by-step',
                'personal_sharing': 'educational',
                'humor_level': 'encouraging',
                'trust_threshold': 0.6
            },
            RelationshipType.PROFESSIONAL.value: {
                'style': CommunicationStyle.FORMAL,
                'tone': 'respectful and competent',
                'detail_level': 'precise',
                'personal_sharing': 'none',
                'humor_level': 'minimal',
                'trust_threshold': 0.4
            },
            RelationshipType.FAMILY.value: {
                'style': CommunicationStyle.EMPATHETIC,
                'tone': 'caring and understanding',
                'detail_level': 'personal',
                'personal_sharing': 'intimate',
                'humor_level': 'familiar',
                'trust_threshold': 0.9
            }
        }
    
    async def analyze_social_context(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the social context for a user interaction
        """
        try:
            # Get or create social context for user
            context = await self._get_or_create_context(user_id, interaction_data)
            
            # Analyze relationship dynamics
            relationship_analysis = await self._analyze_relationship(context, interaction_data)
            
            # Determine appropriate communication style
            communication_style = await self._determine_communication_style(context, relationship_analysis)
            
            # Update context based on current interaction
            await self._update_context(context, interaction_data, relationship_analysis)
            
            return {
                'context_id': context.context_id,
                'user_id': user_id,
                'relationship_type': context.relationship_type.value,
                'trust_level': context.trust_level,
                'familiarity_score': context.familiarity_score,
                'communication_style': communication_style,
                'relationship_analysis': relationship_analysis,
                'adaptation_recommendations': self._generate_adaptation_recommendations(context, communication_style)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social context: {e}")
            return {
                'error': f"Social context analysis failed: {str(e)}",
                'fallback_style': CommunicationStyle.CASUAL.value
            }
    
    async def _get_or_create_context(self, user_id: str, interaction_data: Dict[str, Any]) -> SocialContext:
        """Get existing context or create new one for user"""
        if self.contexts_collection is not None:
            # Try to find existing context
            existing_context = await self.contexts_collection.find_one({'user_id': user_id})
            if existing_context:
                return SocialContext(existing_context)
        
        # Create new context
        context_data = {
            'user_id': user_id,
            'relationship_type': interaction_data.get('relationship_type', 'stranger'),
            'interaction_history': [],
            'trust_level': 0.5,
            'familiarity_score': 0.0,
            'communication_preferences': interaction_data.get('preferences', {})
        }
        
        return SocialContext(context_data)
    
    async def _analyze_relationship(self, context: SocialContext, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationship dynamics and history"""
        interaction_count = len(context.interaction_history)
        
        # Calculate relationship metrics
        frequency_score = min(interaction_count / 10.0, 1.0)  # Normalize to 0-1
        
        # Analyze interaction patterns
        positive_interactions = sum(1 for interaction in context.interaction_history 
                                  if interaction.get('sentiment', 0) > 0)
        sentiment_ratio = positive_interactions / max(interaction_count, 1)
        
        # Time-based familiarity
        if context.last_interaction:
            days_since_last = (datetime.now() - datetime.fromisoformat(context.last_interaction)).days
            recency_score = max(0, 1 - (days_since_last / 30))  # Decay over 30 days
        else:
            recency_score = 0
        
        return {
            'interaction_count': interaction_count,
            'frequency_score': frequency_score,
            'sentiment_ratio': sentiment_ratio,
            'recency_score': recency_score,
            'relationship_strength': (frequency_score + sentiment_ratio + recency_score) / 3,
            'recommended_relationship_type': self._recommend_relationship_type(
                interaction_count, sentiment_ratio, frequency_score
            )
        }
    
    def _recommend_relationship_type(self, interaction_count: int, sentiment_ratio: float, frequency_score: float) -> str:
        """Recommend relationship type based on interaction patterns"""
        if interaction_count < 3:
            return RelationshipType.STRANGER.value
        elif interaction_count < 10 and sentiment_ratio > 0.6:
            return RelationshipType.ACQUAINTANCE.value
        elif interaction_count >= 10 and sentiment_ratio > 0.7 and frequency_score > 0.5:
            return RelationshipType.FRIEND.value
        elif sentiment_ratio > 0.8:
            return RelationshipType.COLLEAGUE.value
        else:
            return RelationshipType.ACQUAINTANCE.value
    
    async def _determine_communication_style(self, context: SocialContext, relationship_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine appropriate communication style"""
        relationship_type = context.relationship_type.value
        rules = self.communication_rules.get(relationship_type, self.communication_rules[RelationshipType.CASUAL.value])
        
        # Adjust based on trust level and familiarity
        trust_multiplier = context.trust_level
        familiarity_multiplier = context.familiarity_score
        
        style_confidence = min(trust_multiplier + familiarity_multiplier, 1.0)
        
        return {
            'primary_style': rules['style'].value,
            'tone': rules['tone'],
            'detail_level': rules['detail_level'],
            'personal_sharing': rules['personal_sharing'],
            'humor_level': rules['humor_level'],
            'style_confidence': style_confidence,
            'adaptations': self._calculate_style_adaptations(context, relationship_analysis)
        }
    
    def _calculate_style_adaptations(self, context: SocialContext, relationship_analysis: Dict[str, Any]) -> List[str]:
        """Calculate specific style adaptations"""
        adaptations = []
        
        if context.trust_level > 0.8:
            adaptations.append("Use more personal examples and experiences")
        
        if relationship_analysis['sentiment_ratio'] < 0.5:
            adaptations.append("Be more supportive and understanding")
        
        if context.familiarity_score > 0.7:
            adaptations.append("Reference previous conversations and shared context")
        
        if relationship_analysis['interaction_count'] > 20:
            adaptations.append("Maintain consistency with established communication patterns")
        
        return adaptations
    
    def _generate_adaptation_recommendations(self, context: SocialContext, communication_style: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for communication adaptation"""
        recommendations = []
        
        style = communication_style['primary_style']
        
        if style == CommunicationStyle.FORMAL.value:
            recommendations.extend([
                "Use respectful language and proper greetings",
                "Provide comprehensive information without being overwhelming",
                "Maintain professional boundaries"
            ])
        elif style == CommunicationStyle.FRIENDLY.value:
            recommendations.extend([
                "Use warm, conversational tone",
                "Share relevant personal insights when appropriate",
                "Show genuine interest in their wellbeing"
            ])
        elif style == CommunicationStyle.SUPPORTIVE.value:
            recommendations.extend([
                "Offer encouragement and positive reinforcement",
                "Break down complex topics into manageable steps",
                "Check for understanding frequently"
            ])
        
        # Add trust-based recommendations
        if context.trust_level < 0.5:
            recommendations.append("Focus on building trust through consistent, helpful responses")
        
        return recommendations
    
    async def _update_context(self, context: SocialContext, interaction_data: Dict[str, Any], relationship_analysis: Dict[str, Any]):
        """Update context based on current interaction"""
        # Add current interaction to history
        interaction_record = {
            'timestamp': datetime.now().isoformat(),
            'content_type': interaction_data.get('content_type', 'text'),
            'sentiment': interaction_data.get('sentiment', 0),
            'topic': interaction_data.get('topic', 'general'),
            'user_satisfaction': interaction_data.get('satisfaction', None)
        }
        
        context.interaction_history.append(interaction_record)
        
        # Update familiarity score
        context.familiarity_score = min(context.familiarity_score + 0.1, 1.0)
        
        # Update trust level based on interaction quality
        if interaction_data.get('satisfaction', 0) > 0.7:
            context.trust_level = min(context.trust_level + 0.05, 1.0)
        elif interaction_data.get('satisfaction', 0) < 0.3:
            context.trust_level = max(context.trust_level - 0.1, 0.0)
        
        # Update relationship type if needed
        recommended_type = relationship_analysis.get('recommended_relationship_type')
        if recommended_type and recommended_type != context.relationship_type.value:
            context.relationship_type = RelationshipType(recommended_type)
        
        context.last_interaction = datetime.now().isoformat()
        context.updated_at = datetime.now().isoformat()
        
        # Save to database if available
        if self.contexts_collection is not None:
            await self._save_context(context)
    
    async def _save_context(self, context: SocialContext):
        """Save context to database"""
        try:
            context_dict = {
                'context_id': context.context_id,
                'user_id': context.user_id,
                'relationship_type': context.relationship_type.value,
                'interaction_history': context.interaction_history,
                'communication_preferences': context.communication_preferences,
                'trust_level': context.trust_level,
                'familiarity_score': context.familiarity_score,
                'last_interaction': context.last_interaction,
                'created_at': context.created_at,
                'updated_at': context.updated_at
            }
            
            await self.contexts_collection.update_one(
                {'user_id': context.user_id},
                {'$set': context_dict},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error saving social context: {e}")
    
    async def get_communication_style_for_user(self, user_id: str) -> Dict[str, Any]:
        """Get current communication style recommendation for a user"""
        try:
            if self.contexts_collection is not None:
                context_data = await self.contexts_collection.find_one({'user_id': user_id})
                if context_data:
                    context = SocialContext(context_data)
                    relationship_analysis = {
                        'interaction_count': len(context.interaction_history),
                        'sentiment_ratio': 0.7,  # Default positive
                        'relationship_strength': context.trust_level
                    }
                    
                    return await self._determine_communication_style(context, relationship_analysis)
            
            # Return default style for new users
            return {
                'primary_style': CommunicationStyle.CASUAL.value,
                'tone': 'friendly and helpful',
                'detail_level': 'balanced',
                'personal_sharing': 'minimal',
                'humor_level': 'light',
                'style_confidence': 0.5
            }
            
        except Exception as e:
            logger.error(f"Error getting communication style: {e}")
            return {'error': str(e)}
    
    async def get_relationship_insights(self, user_id: str) -> Dict[str, Any]:
        """Get relationship insights for a user"""
        try:
            if self.contexts_collection is None:
                return {'error': 'Database not available'}
            
            context_data = await self.contexts_collection.find_one({'user_id': user_id})
            if not context_data:
                return {'error': 'No relationship data found for user'}
            
            context = SocialContext(context_data)
            
            # Calculate insights
            total_interactions = len(context.interaction_history)
            recent_interactions = [i for i in context.interaction_history 
                                 if datetime.fromisoformat(i['timestamp']) > datetime.now() - timedelta(days=7)]
            
            return {
                'user_id': user_id,
                'relationship_type': context.relationship_type.value,
                'trust_level': context.trust_level,
                'familiarity_score': context.familiarity_score,
                'total_interactions': total_interactions,
                'recent_interactions': len(recent_interactions),
                'relationship_duration': (datetime.now() - datetime.fromisoformat(context.created_at)).days,
                'last_interaction': context.last_interaction,
                'communication_preferences': context.communication_preferences
            }
            
        except Exception as e:
            logger.error(f"Error getting relationship insights: {e}")
            return {'error': str(e)}
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update communication preferences for a user"""
        try:
            if self.contexts_collection is None:
                return {'error': 'Database not available'}
            
            # Get or create context
            context_data = await self.contexts_collection.find_one({'user_id': user_id})
            if context_data:
                context = SocialContext(context_data)
            else:
                context = SocialContext({'user_id': user_id})
            
            # Update preferences
            context.communication_preferences.update(preferences)
            context.updated_at = datetime.now().isoformat()
            
            # Save updated context
            await self._save_context(context)
            
            return {
                'status': 'success',
                'user_id': user_id,
                'updated_preferences': context.communication_preferences,
                'message': 'Communication preferences updated successfully'
            }
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return {'error': str(e)}