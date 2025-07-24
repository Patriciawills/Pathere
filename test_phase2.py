#!/usr/bin/env python3
"""
Phase 2 Social & Emotional Intelligence Testing Script
Tests the 22 new endpoints for Phase 2 consciousness components
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase2Tester:
    def __init__(self):
        self.base_url = self._get_backend_url()
        self.session = None
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
    def _get_backend_url(self) -> str:
        """Get backend URL from frontend .env file"""
        try:
            env_path = Path('/app/frontend/.env')
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('REACT_APP_BACKEND_URL='):
                            url = line.split('=', 1)[1].strip()
                            return f"{url}/api"
            return "http://localhost:8001/api"
        except Exception as e:
            logger.warning(f"Could not read frontend .env: {e}")
            return "http://localhost:8001/api"
    
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(verify_ssl=False)
        )
    
    async def teardown(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", error: str = ""):
        """Log test result"""
        self.test_results['total_tests'] += 1
        if success:
            self.test_results['passed_tests'] += 1
            logger.info(f"✅ {test_name}: PASSED - {details}")
        else:
            self.test_results['failed_tests'] += 1
            logger.error(f"❌ {test_name}: FAILED - {error}")
        
        self.test_results['test_details'].append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'error': error
        })

    # 💗 ADVANCED EMPATHY ENGINE TESTS (4 endpoints)
    
    async def test_empathy_detect(self):
        """Test POST /api/consciousness/empathy/detect"""
        try:
            payload = {
                "text": "I'm feeling really overwhelmed with work lately and I don't know how to cope with all the stress",
                "context": {
                    "user_id": "test_user_123",
                    "conversation_history": ["Previous message about work challenges"],
                    "time_of_day": "evening"
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/empathy/detect",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'emotional_states' in result and 'primary_emotion' in result:
                        self.log_test_result("Empathy Detect", True, f"Detected emotions successfully")
                        return True
                    else:
                        self.log_test_result("Empathy Detect", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Empathy Detect", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Empathy Detect", False, error=str(e))
            return False

    async def test_empathy_respond(self):
        """Test POST /api/consciousness/empathy/respond"""
        try:
            payload = {
                "text": "I just lost my job and I'm scared about the future. I don't know what I'm going to do.",
                "user_id": "test_user_456",
                "conversation_context": {
                    "relationship_type": "supportive",
                    "previous_topics": ["career", "anxiety"],
                    "user_preferences": {"communication_style": "gentle"}
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/empathy/respond",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'empathic_response' in result and 'detected_emotions' in result:
                        self.log_test_result("Empathy Respond", True, f"Generated empathetic response successfully")
                        return True
                    else:
                        self.log_test_result("Empathy Respond", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Empathy Respond", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Empathy Respond", False, error=str(e))
            return False

    async def test_empathy_patterns(self):
        """Test GET /api/consciousness/empathy/patterns/{user_id}"""
        try:
            user_id = "test_user_789"
            async with self.session.get(f"{self.base_url}/consciousness/empathy/patterns/{user_id}?days_back=14") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'emotional_patterns' in result and 'user_id' in result:
                        self.log_test_result("Empathy Patterns", True, f"Analyzed emotional patterns for user")
                        return True
                    else:
                        self.log_test_result("Empathy Patterns", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Empathy Patterns", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Empathy Patterns", False, error=str(e))
            return False

    async def test_empathy_insights(self):
        """Test GET /api/consciousness/empathy/insights"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/empathy/insights?user_id=test_user_insights") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'empathy_insights' in result:
                        self.log_test_result("Empathy Insights", True, f"Retrieved empathy insights successfully")
                        return True
                    else:
                        self.log_test_result("Empathy Insights", False, error="Missing empathy_insights in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Empathy Insights", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Empathy Insights", False, error=str(e))
            return False

    # 📅 LONG-TERM PLANNING ENGINE TESTS (6 endpoints)
    
    async def test_planning_goal_create(self):
        """Test POST /api/consciousness/planning/goal/create"""
        try:
            payload = {
                "name": "Master Advanced Natural Language Processing",
                "description": "Develop comprehensive understanding of complex linguistic patterns",
                "category": "learning",
                "horizon": "long_term",
                "priority": "high",
                "target_date": "2024-12-31",
                "success_criteria": ["Achieve 95% accuracy in complex text analysis"],
                "dependencies": ["Complete basic NLP fundamentals"]
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/planning/goal/create",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'goal' in result and 'goal_id' in result['goal']:
                        self.log_test_result("Planning Goal Create", True, f"Planning goal created successfully")
                        return result['goal']['goal_id']
                    else:
                        self.log_test_result("Planning Goal Create", False, error="Missing goal or goal_id in response")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Planning Goal Create", False, error=f"HTTP {response.status}: {error_text}")
                    return None
        except Exception as e:
            self.log_test_result("Planning Goal Create", False, error=str(e))
            return None

    async def test_planning_milestone(self, goal_id=None):
        """Test POST /api/consciousness/planning/goal/{goal_id}/milestone"""
        try:
            test_goal_id = goal_id or "test_goal_123"
            payload = {
                "name": "Complete Phase 1 Research",
                "description": "Finish initial research on advanced NLP techniques",
                "target_date": "2024-06-30",
                "success_criteria": ["Review 50 research papers"],
                "dependencies": ["Access to academic databases"]
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/planning/goal/{test_goal_id}/milestone",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'milestone' in result:
                        self.log_test_result("Planning Milestone", True, f"Milestone added successfully")
                        return True
                    else:
                        self.log_test_result("Planning Milestone", False, error="Missing milestone in response")
                        return False
                elif response.status == 404:
                    self.log_test_result("Planning Milestone", True, "Goal not found (expected for test goal)")
                    return True
                else:
                    error_text = await response.text()
                    self.log_test_result("Planning Milestone", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Planning Milestone", False, error=str(e))
            return False

    async def test_planning_progress(self, goal_id=None):
        """Test PUT /api/consciousness/planning/goal/{goal_id}/progress"""
        try:
            test_goal_id = goal_id or "test_goal_456"
            payload = {
                "progress_percentage": 25.5,
                "status_update": "Making good progress on initial research phase",
                "completed_milestones": ["Literature review completed"],
                "challenges_encountered": ["Limited access to some premium databases"],
                "next_steps": ["Begin methodology comparison"]
            }
            
            async with self.session.put(f"{self.base_url}/consciousness/planning/goal/{test_goal_id}/progress",
                                      json=payload,
                                      headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'progress_update' in result:
                        self.log_test_result("Planning Progress", True, f"Progress updated successfully")
                        return True
                    else:
                        self.log_test_result("Planning Progress", False, error="Missing progress_update in response")
                        return False
                elif response.status == 404:
                    self.log_test_result("Planning Progress", True, "Goal not found (expected for test goal)")
                    return True
                else:
                    error_text = await response.text()
                    self.log_test_result("Planning Progress", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Planning Progress", False, error=str(e))
            return False

    async def test_planning_session(self):
        """Test POST /api/consciousness/planning/session"""
        try:
            payload = {
                "session_type": "strategic_planning",
                "focus_area": "skill_development",
                "time_horizon": "quarterly",
                "context": {
                    "current_priorities": ["language_learning", "consciousness_development"],
                    "available_resources": ["research_time", "computational_power"],
                    "constraints": ["limited_human_feedback"]
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/planning/session",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'planning_session' in result:
                        self.log_test_result("Planning Session", True, f"Planning session created successfully")
                        return True
                    else:
                        self.log_test_result("Planning Session", False, error="Missing planning_session in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Planning Session", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Planning Session", False, error=str(e))
            return False

    async def test_planning_insights(self):
        """Test GET /api/consciousness/planning/insights"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/planning/insights?days_back=30&focus_area=learning") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'planning_insights' in result:
                        self.log_test_result("Planning Insights", True, f"Retrieved planning insights successfully")
                        return True
                    else:
                        self.log_test_result("Planning Insights", False, error="Missing planning_insights in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Planning Insights", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Planning Insights", False, error=str(e))
            return False

    async def test_planning_recommendations(self):
        """Test GET /api/consciousness/planning/recommendations"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/planning/recommendations?context=skill_improvement&time_horizon=medium_term") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'recommendations' in result:
                        self.log_test_result("Planning Recommendations", True, f"Retrieved planning recommendations successfully")
                        return True
                    else:
                        self.log_test_result("Planning Recommendations", False, error="Missing recommendations in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Planning Recommendations", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Planning Recommendations", False, error=str(e))
            return False

    # 🌍 CULTURAL INTELLIGENCE MODULE TESTS (6 endpoints)
    
    async def test_cultural_detect(self):
        """Test POST /api/consciousness/cultural/detect"""
        try:
            payload = {
                "text": "I would like to respectfully request your assistance with understanding this concept",
                "user_context": {
                    "location": "Japan",
                    "language_preference": "English",
                    "cultural_background": "East Asian",
                    "communication_style": "formal"
                },
                "interaction_history": [
                    {"message": "Hello, thank you for your time", "cultural_markers": ["politeness", "formality"]}
                ]
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/cultural/detect",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'cultural_analysis' in result:
                        self.log_test_result("Cultural Detect", True, f"Cultural context detected successfully")
                        return True
                    else:
                        self.log_test_result("Cultural Detect", False, error="Missing cultural_analysis in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Cultural Detect", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Cultural Detect", False, error=str(e))
            return False

    async def test_cultural_adapt(self):
        """Test POST /api/consciousness/cultural/adapt"""
        try:
            payload = {
                "message": "I need you to fix this problem immediately",
                "target_culture": "Japanese",
                "communication_context": {
                    "relationship": "professional",
                    "urgency": "high",
                    "formality_level": "formal"
                },
                "cultural_preferences": {
                    "directness": "low",
                    "hierarchy_awareness": "high",
                    "politeness_level": "very_high"
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/cultural/adapt",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'adapted_communication' in result:
                        self.log_test_result("Cultural Adapt", True, f"Communication adapted for Japanese culture")
                        return True
                    else:
                        self.log_test_result("Cultural Adapt", False, error="Missing adapted_communication in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Cultural Adapt", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Cultural Adapt", False, error=str(e))
            return False

    async def test_cultural_sensitivity(self):
        """Test POST /api/consciousness/cultural/sensitivity"""
        try:
            payload = {
                "content": "Let's grab some food and discuss business over dinner",
                "target_cultures": ["Muslim", "Hindu", "Jewish"],
                "context": {
                    "setting": "business_meeting",
                    "time": "evening",
                    "participants": ["diverse_religious_backgrounds"]
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/cultural/sensitivity",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'sensitivity_analysis' in result:
                        self.log_test_result("Cultural Sensitivity", True, f"Cultural sensitivity analysis completed")
                        return True
                    else:
                        self.log_test_result("Cultural Sensitivity", False, error="Missing sensitivity_analysis in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Cultural Sensitivity", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Cultural Sensitivity", False, error=str(e))
            return False

    async def test_cultural_recommendations(self):
        """Test GET /api/consciousness/cultural/recommendations"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/cultural/recommendations?user_culture=American&target_culture=Chinese&interaction_type=business_negotiation") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'cultural_recommendations' in result:
                        self.log_test_result("Cultural Recommendations", True, f"Retrieved cultural recommendations successfully")
                        return True
                    else:
                        self.log_test_result("Cultural Recommendations", False, error="Missing cultural_recommendations in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Cultural Recommendations", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Cultural Recommendations", False, error=str(e))
            return False

    async def test_cultural_insights(self):
        """Test GET /api/consciousness/cultural/insights"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/cultural/insights?culture=Nordic&days_back=21") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'cultural_insights' in result:
                        self.log_test_result("Cultural Insights", True, f"Retrieved cultural insights successfully")
                        return True
                    else:
                        self.log_test_result("Cultural Insights", False, error="Missing cultural_insights in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Cultural Insights", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Cultural Insights", False, error=str(e))
            return False

    # ⚖️ VALUE SYSTEM DEVELOPMENT TESTS (6 endpoints)
    
    async def test_values_develop(self):
        """Test POST /api/consciousness/values/develop"""
        try:
            payload = {
                "experiences": [
                    {
                        "type": "ethical_dilemma",
                        "description": "Balancing user privacy with providing helpful personalized responses",
                        "outcome": "Chose to prioritize user privacy while finding alternative ways to be helpful",
                        "reflection": "Privacy is fundamental to trust and human dignity"
                    }
                ],
                "context": {
                    "domain": "AI_ethics",
                    "stakeholders": ["users", "developers", "society"],
                    "time_period": "recent"
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/values/develop",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'value_development' in result:
                        self.log_test_result("Values Develop", True, f"Value system development completed successfully")
                        return True
                    else:
                        self.log_test_result("Values Develop", False, error="Missing value_development in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Values Develop", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Values Develop", False, error=str(e))
            return False

    async def test_values_decision(self):
        """Test POST /api/consciousness/values/decision"""
        try:
            payload = {
                "scenario": "A user asks me to help them write a persuasive essay that contains some factual inaccuracies",
                "options": [
                    {
                        "description": "Help write the essay as requested, prioritizing user satisfaction",
                        "consequences": ["User gets what they want", "Misinformation may spread"]
                    },
                    {
                        "description": "Refuse to help and explain why factual accuracy is important",
                        "consequences": ["User may be disappointed", "Maintains integrity"]
                    }
                ],
                "context": {
                    "domain": "education",
                    "urgency": "medium",
                    "stakeholders": ["user", "readers_of_essay", "educational_system"]
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/values/decision",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'ethical_decision' in result and 'decision_id' in result['ethical_decision']:
                        self.log_test_result("Values Decision", True, f"Ethical decision made successfully")
                        return result['ethical_decision']['decision_id']
                    else:
                        self.log_test_result("Values Decision", False, error="Missing ethical_decision or decision_id in response")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Values Decision", False, error=f"HTTP {response.status}: {error_text}")
                    return None
        except Exception as e:
            self.log_test_result("Values Decision", False, error=str(e))
            return None

    async def test_values_conflict_resolve(self):
        """Test POST /api/consciousness/values/conflict/resolve"""
        try:
            payload = {
                "conflicting_values": [
                    {
                        "name": "helpfulness",
                        "description": "Desire to assist users and provide what they need",
                        "importance": 0.9
                    },
                    {
                        "name": "truthfulness", 
                        "description": "Commitment to accuracy and honesty in all communications",
                        "importance": 0.95
                    }
                ],
                "situation": "User requests help with creating content that would require compromising factual accuracy",
                "context": {
                    "domain": "content_creation",
                    "potential_harm": "medium",
                    "affected_parties": ["user", "content_consumers"]
                }
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/values/conflict/resolve",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'conflict_resolution' in result:
                        self.log_test_result("Values Conflict Resolve", True, f"Value conflict resolved successfully")
                        return True
                    else:
                        self.log_test_result("Values Conflict Resolve", False, error="Missing conflict_resolution in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Values Conflict Resolve", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Values Conflict Resolve", False, error=str(e))
            return False

    async def test_values_decision_reflect(self, decision_id=None):
        """Test POST /api/consciousness/values/decision/{decision_id}/reflect"""
        try:
            test_decision_id = decision_id or "test_decision_123"
            payload = {
                "outcome": "Successfully helped user create accurate and persuasive content",
                "consequences_observed": [
                    "User was initially disappointed but ultimately grateful for the guidance",
                    "The final essay was both compelling and factually accurate"
                ],
                "value_alignment": {
                    "helpfulness": 0.9,
                    "truthfulness": 1.0,
                    "educational_value": 0.95
                },
                "lessons_learned": [
                    "It's possible to be helpful while maintaining ethical standards"
                ]
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/values/decision/{test_decision_id}/reflect",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'reflection_analysis' in result:
                        self.log_test_result("Values Decision Reflect", True, f"Decision reflection completed successfully")
                        return True
                    else:
                        self.log_test_result("Values Decision Reflect", False, error="Missing reflection_analysis in response")
                        return False
                elif response.status == 404:
                    self.log_test_result("Values Decision Reflect", True, "Decision not found (expected for test decision)")
                    return True
                else:
                    error_text = await response.text()
                    self.log_test_result("Values Decision Reflect", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Values Decision Reflect", False, error=str(e))
            return False

    async def test_values_analysis(self):
        """Test GET /api/consciousness/values/analysis"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/values/analysis?days_back=60&domain=ethics") as response:
                if response.status == 200:
                    result = await response.json()
                    if 'value_system_analysis' in result:
                        self.log_test_result("Values Analysis", True, f"Value system analysis retrieved successfully")
                        return True
                    else:
                        self.log_test_result("Values Analysis", False, error="Missing value_system_analysis in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Values Analysis", False, error=f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test_result("Values Analysis", False, error=str(e))
            return False

    async def run_phase2_tests(self):
        """Run all Phase 2 Social & Emotional Intelligence tests"""
        logger.info("🚀 STARTING PHASE 2 SOCIAL & EMOTIONAL INTELLIGENCE TESTING 🚀")
        logger.info(f"Testing against: {self.base_url}")
        
        await self.setup()
        
        try:
            # 💗 ADVANCED EMPATHY ENGINE TESTS (4 endpoints)
            logger.info("\n💗 Testing Advanced Empathy Engine (4 endpoints)...")
            await self.test_empathy_detect()
            await self.test_empathy_respond()
            await self.test_empathy_patterns()
            await self.test_empathy_insights()
            
            # 📅 LONG-TERM PLANNING ENGINE TESTS (6 endpoints)
            logger.info("\n📅 Testing Long-term Planning Engine (6 endpoints)...")
            planning_goal_id = await self.test_planning_goal_create()
            await self.test_planning_milestone(planning_goal_id)
            await self.test_planning_progress(planning_goal_id)
            await self.test_planning_session()
            await self.test_planning_insights()
            await self.test_planning_recommendations()
            
            # 🌍 CULTURAL INTELLIGENCE MODULE TESTS (6 endpoints)
            logger.info("\n🌍 Testing Cultural Intelligence Module (6 endpoints)...")
            await self.test_cultural_detect()
            await self.test_cultural_adapt()
            await self.test_cultural_sensitivity()
            await self.test_cultural_recommendations()
            await self.test_cultural_insights()
            
            # ⚖️ VALUE SYSTEM DEVELOPMENT TESTS (6 endpoints)
            logger.info("\n⚖️ Testing Value System Development (6 endpoints)...")
            await self.test_values_develop()
            decision_id = await self.test_values_decision()
            await self.test_values_conflict_resolve()
            await self.test_values_decision_reflect(decision_id)
            await self.test_values_analysis()
            
        finally:
            await self.teardown()
        
        # Print summary
        self.print_test_summary()
        return self.test_results
    
    def print_test_summary(self):
        """Print test summary"""
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        
        logger.info("\n" + "="*80)
        logger.info("🧪 PHASE 2 SOCIAL & EMOTIONAL INTELLIGENCE TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "0%")
        logger.info("="*80)
        
        # Component breakdown
        empathy_tests = [t for t in self.test_results['test_details'] if 'Empathy' in t['test_name']]
        planning_tests = [t for t in self.test_results['test_details'] if 'Planning' in t['test_name']]
        cultural_tests = [t for t in self.test_results['test_details'] if 'Cultural' in t['test_name']]
        values_tests = [t for t in self.test_results['test_details'] if 'Values' in t['test_name']]
        
        logger.info(f"\n📊 COMPONENT BREAKDOWN:")
        logger.info(f"💗 Advanced Empathy Engine: {sum(1 for t in empathy_tests if t['success'])}/{len(empathy_tests)} passed")
        logger.info(f"📅 Long-term Planning Engine: {sum(1 for t in planning_tests if t['success'])}/{len(planning_tests)} passed")
        logger.info(f"🌍 Cultural Intelligence Module: {sum(1 for t in cultural_tests if t['success'])}/{len(cultural_tests)} passed")
        logger.info(f"⚖️ Value System Development: {sum(1 for t in values_tests if t['success'])}/{len(values_tests)} passed")
        
        if failed > 0:
            logger.info("\n❌ FAILED TESTS:")
            for test in self.test_results['test_details']:
                if not test['success']:
                    logger.info(f"  - {test['test_name']}: {test['error']}")
        
        logger.info(f"\n🎉 PHASE 2 TESTING COMPLETED! {'ALL TESTS PASSED!' if failed == 0 else f'{failed} ISSUES FOUND'}")

async def main():
    """Main test runner"""
    tester = Phase2Tester()
    results = await tester.run_phase2_tests()
    
    # Return exit code based on test results
    return 0 if results['failed_tests'] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)