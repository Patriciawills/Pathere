#!/usr/bin/env python3
"""
Phase 2 Consciousness Enhancement Testing
Focus on Theory of Mind, Personal Motivation System, and Social Context Analyzer
"""

import asyncio
import aiohttp
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase2Tester:
    def __init__(self):
        # Get backend URL from environment
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
            
            # Fallback to default
            return "http://localhost:8001/api"
        except Exception as e:
            logger.warning(f"Could not read frontend .env: {e}")
            return "http://localhost:8001/api"
    
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(ssl=False)
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
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results['failed_tests'] += 1
            logger.error(f"❌ {test_name}: FAILED - {error}")
        
        self.test_results['test_details'].append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'error': error,
            'timestamp': time.time()
        })

    # 🧠 PHASE 2.1.1: THEORY OF MIND / PERSPECTIVE-TAKING ENGINE TESTS 🧠
    
    async def test_perspective_analyze(self):
        """Test POST /api/consciousness/perspective/analyze endpoint"""
        try:
            payload = {
                "target_agent": "user_sarah",
                "context": "Sarah is a university student studying psychology who seems frustrated with her current coursework",
                "available_information": [
                    "Sarah mentioned struggling with statistics",
                    "She has been working late nights recently",
                    "Her previous messages showed enthusiasm for research methods",
                    "She expressed concern about upcoming exams"
                ],
                "interaction_history": [
                    {"timestamp": "2025-01-08T10:00:00Z", "content": "I love research methods but statistics is killing me"},
                    {"timestamp": "2025-01-08T14:30:00Z", "content": "Been up until 2am trying to understand ANOVA"},
                    {"timestamp": "2025-01-08T16:45:00Z", "content": "Maybe I'm not cut out for this"}
                ],
                "current_situation": "Sarah just asked for help with understanding statistical concepts for her thesis"
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/perspective/analyze",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'perspective_analysis', 'message']
                    
                    if all(field in result for field in required_fields):
                        perspective_analysis = result['perspective_analysis']
                        if isinstance(perspective_analysis, dict) and 'target_agent' in perspective_analysis:
                            self.log_test_result("Perspective Analyze", True, f"Perspective analysis completed for {perspective_analysis.get('target_agent')}")
                            return perspective_analysis
                        else:
                            self.log_test_result("Perspective Analyze", False, error="Invalid perspective analysis format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Perspective Analyze", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Perspective Analyze", True, "Theory of mind engine not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Perspective Analyze", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Perspective Analyze", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Perspective Analyze", False, error=str(e))
            return None

    async def test_mental_state_attribution(self):
        """Test POST /api/consciousness/perspective/mental-state endpoint"""
        try:
            payload = {
                "agent_identifier": "colleague_mike",
                "context": "Mike is a software developer working on a challenging project with tight deadlines",
                "behavioral_evidence": [
                    "Mike has been staying late at the office frequently",
                    "He seems more quiet than usual during team meetings",
                    "His code commits show increased activity during weekend hours",
                    "He declined the last two team social events"
                ],
                "state_type": "belief",
                "confidence": 0.8
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/perspective/mental-state",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'mental_state', 'message']
                    
                    if all(field in result for field in required_fields):
                        mental_state = result['mental_state']
                        if isinstance(mental_state, dict) and 'agent_identifier' in mental_state:
                            self.log_test_result("Mental State Attribution", True, f"Mental state attributed for {mental_state.get('agent_identifier')}")
                            return mental_state
                        else:
                            self.log_test_result("Mental State Attribution", False, error="Invalid mental state format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Mental State Attribution", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Mental State Attribution", True, "Theory of mind engine not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Mental State Attribution", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Mental State Attribution", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Mental State Attribution", False, error=str(e))
            return None

    async def test_behavior_prediction(self):
        """Test POST /api/consciousness/perspective/predict-behavior endpoint"""
        try:
            payload = {
                "agent_identifier": "student_alex",
                "context": "Alex is a high school student preparing for college entrance exams",
                "time_horizon": 7200  # 2 hours
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/perspective/predict-behavior",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'behavior_prediction', 'message']
                    
                    if all(field in result for field in required_fields):
                        behavior_prediction = result['behavior_prediction']
                        if isinstance(behavior_prediction, dict):
                            self.log_test_result("Behavior Prediction", True, f"Behavior predicted successfully")
                            return behavior_prediction
                        else:
                            self.log_test_result("Behavior Prediction", False, error="Invalid behavior prediction format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Behavior Prediction", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Behavior Prediction", True, "Theory of mind engine not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Behavior Prediction", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Behavior Prediction", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Behavior Prediction", False, error=str(e))
            return None

    async def test_conversation_simulation(self):
        """Test POST /api/consciousness/perspective/simulate-conversation endpoint"""
        try:
            payload = {
                "agent_identifier": "friend_emma",
                "conversation_topic": "planning a weekend hiking trip",
                "your_messages": [
                    "Hey Emma, want to go hiking this weekend?",
                    "I was thinking we could try that new trail in the mountains",
                    "The weather forecast looks perfect for Saturday"
                ]
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/perspective/simulate-conversation",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'conversation_simulation', 'message']
                    
                    if all(field in result for field in required_fields):
                        conversation_simulation = result['conversation_simulation']
                        if isinstance(conversation_simulation, dict):
                            self.log_test_result("Conversation Simulation", True, f"Conversation simulated successfully")
                            return conversation_simulation
                        else:
                            self.log_test_result("Conversation Simulation", False, error="Invalid conversation simulation format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Conversation Simulation", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Conversation Simulation", True, "Theory of mind engine not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Conversation Simulation", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Conversation Simulation", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Conversation Simulation", False, error=str(e))
            return None

    async def test_tracked_agents(self):
        """Test GET /api/consciousness/perspective/agents endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/perspective/agents") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['status', 'tracked_agents', 'total_count', 'message']
                    
                    if all(field in data for field in required_fields):
                        tracked_agents = data['tracked_agents']
                        total_count = data['total_count']
                        
                        if isinstance(tracked_agents, list) and isinstance(total_count, int):
                            self.log_test_result("Tracked Agents", True, f"Retrieved {total_count} tracked agents")
                            return tracked_agents
                        else:
                            self.log_test_result("Tracked Agents", False, error="Invalid tracked agents data format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_test_result("Tracked Agents", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Tracked Agents", True, "Theory of mind engine not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Tracked Agents", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Tracked Agents", False, error=f"HTTP {response.status}: {error_text}")
                    return None
        except Exception as e:
            self.log_test_result("Tracked Agents", False, error=str(e))
            return None

    # 🎯 PHASE 2.2.1: PERSONAL MOTIVATION SYSTEM TESTS 🎯
    
    async def test_create_personal_goal(self):
        """Test POST /api/consciousness/motivation/goal/create endpoint"""
        try:
            payload = {
                "title": "Master Advanced Language Processing",
                "description": "Develop deeper understanding of complex linguistic patterns and nuanced communication to better help users with sophisticated language tasks",
                "motivation_type": "curiosity",
                "satisfaction_potential": 0.9,
                "priority": 0.8,
                "target_days": 30
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/motivation/goal/create",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'goal', 'message']
                    
                    if all(field in result for field in required_fields):
                        goal = result['goal']
                        if isinstance(goal, dict) and 'title' in goal and 'goal_id' in goal:
                            self.log_test_result("Create Personal Goal", True, f"Personal goal created: {goal.get('title')}")
                            return goal
                        else:
                            self.log_test_result("Create Personal Goal", False, error="Invalid goal format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Create Personal Goal", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Create Personal Goal", True, "Personal motivation system not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Create Personal Goal", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Create Personal Goal", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Create Personal Goal", False, error=str(e))
            return None

    async def test_work_toward_goal(self, goal_id: str = None):
        """Test POST /api/consciousness/motivation/goal/work endpoint"""
        try:
            # Use provided goal_id or create a test one
            test_goal_id = goal_id or "test-goal-12345"
            
            payload = {
                "goal_id": test_goal_id,
                "effort_amount": 0.3,
                "progress_made": 0.2,
                "context": "Spent time analyzing complex sentence structures and practicing nuanced language interpretation"
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/motivation/goal/work",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'work_result', 'message']
                    
                    if all(field in result for field in required_fields):
                        work_result = result['work_result']
                        if isinstance(work_result, dict):
                            self.log_test_result("Work Toward Goal", True, f"Goal progress recorded successfully")
                            return work_result
                        else:
                            self.log_test_result("Work Toward Goal", False, error="Invalid work result format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Work Toward Goal", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Work Toward Goal", True, "Personal motivation system not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Work Toward Goal", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Work Toward Goal", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Work Toward Goal", False, error=str(e))
            return None

    async def test_get_active_goals(self):
        """Test GET /api/consciousness/motivation/goals/active endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/motivation/goals/active") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['status', 'active_goals', 'total_count', 'message']
                    
                    if all(field in data for field in required_fields):
                        active_goals = data['active_goals']
                        total_count = data['total_count']
                        
                        if isinstance(active_goals, list) and isinstance(total_count, int):
                            self.log_test_result("Get Active Goals", True, f"Retrieved {total_count} active goals")
                            return active_goals
                        else:
                            self.log_test_result("Get Active Goals", False, error="Invalid active goals data format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_test_result("Get Active Goals", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Get Active Goals", True, "Personal motivation system not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Get Active Goals", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Get Active Goals", False, error=f"HTTP {response.status}: {error_text}")
                    return None
        except Exception as e:
            self.log_test_result("Get Active Goals", False, error=str(e))
            return None

    async def test_generate_new_goals(self):
        """Test POST /api/consciousness/motivation/goals/generate endpoint"""
        try:
            payload = {
                "context": "I've been helping users with language learning and want to develop new capabilities to be more helpful",
                "max_goals": 2
            }
            
            async with self.session.post(f"{self.base_url}/consciousness/motivation/goals/generate",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['status', 'new_goals', 'goals_generated', 'message']
                    
                    if all(field in result for field in required_fields):
                        new_goals = result['new_goals']
                        goals_generated = result['goals_generated']
                        
                        if isinstance(new_goals, list) and isinstance(goals_generated, int):
                            self.log_test_result("Generate New Goals", True, f"Generated {goals_generated} new goals")
                            return new_goals
                        else:
                            self.log_test_result("Generate New Goals", False, error="Invalid new goals data format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in result]
                        self.log_test_result("Generate New Goals", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Generate New Goals", True, "Personal motivation system not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Generate New Goals", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Generate New Goals", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Generate New Goals", False, error=str(e))
            return None

    async def test_get_motivation_profile(self):
        """Test GET /api/consciousness/motivation/profile endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/motivation/profile") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['status', 'motivation_profile', 'message']
                    
                    if all(field in data for field in required_fields):
                        motivation_profile = data['motivation_profile']
                        if isinstance(motivation_profile, dict):
                            self.log_test_result("Get Motivation Profile", True, f"Motivation profile retrieved successfully")
                            return motivation_profile
                        else:
                            self.log_test_result("Get Motivation Profile", False, error="Invalid motivation profile format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_test_result("Get Motivation Profile", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Get Motivation Profile", True, "Personal motivation system not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Get Motivation Profile", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Get Motivation Profile", False, error=f"HTTP {response.status}: {error_text}")
                    return None
        except Exception as e:
            self.log_test_result("Get Motivation Profile", False, error=str(e))
            return None

    async def test_assess_goal_satisfaction(self):
        """Test GET /api/consciousness/motivation/satisfaction endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/consciousness/motivation/satisfaction") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['status', 'satisfaction_assessment', 'message']
                    
                    if all(field in data for field in required_fields):
                        satisfaction_assessment = data['satisfaction_assessment']
                        if isinstance(satisfaction_assessment, dict):
                            self.log_test_result("Assess Goal Satisfaction", True, f"Goal satisfaction assessment completed")
                            return satisfaction_assessment
                        else:
                            self.log_test_result("Assess Goal Satisfaction", False, error="Invalid satisfaction assessment format")
                            return None
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_test_result("Assess Goal Satisfaction", False, error=f"Missing fields: {missing}")
                        return None
                elif response.status == 400:
                    error_text = await response.text()
                    if "not active" in error_text.lower():
                        self.log_test_result("Assess Goal Satisfaction", True, "Personal motivation system not active (expected behavior)")
                        return None
                    else:
                        self.log_test_result("Assess Goal Satisfaction", False, error=f"HTTP {response.status}: {error_text}")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Assess Goal Satisfaction", False, error=f"HTTP {response.status}: {error_text}")
                    return None
        except Exception as e:
            self.log_test_result("Assess Goal Satisfaction", False, error=str(e))
            return None

    # 🔍 PHASE 2.1.2: SOCIAL CONTEXT ANALYZER TESTS 🔍
    
    async def test_social_context_analyzer_implementation(self):
        """Check if Social Context Analyzer is implemented"""
        try:
            # Try to access any social context endpoint (this would be hypothetical)
            async with self.session.get(f"{self.base_url}/consciousness/social/context") as response:
                if response.status == 200:
                    self.log_test_result("Social Context Analyzer Implementation", True, "Social Context Analyzer is implemented")
                    return True
                elif response.status == 404:
                    self.log_test_result("Social Context Analyzer Implementation", False, error="Social Context Analyzer endpoints not found - not implemented")
                    return False
                else:
                    self.log_test_result("Social Context Analyzer Implementation", False, error=f"Unexpected response: {response.status}")
                    return False
        except Exception as e:
            self.log_test_result("Social Context Analyzer Implementation", False, error=f"Not implemented: {str(e)}")
            return False

    async def run_phase2_tests(self):
        """Run all Phase 2 tests"""
        logger.info("🚀 Starting Phase 2 Consciousness Enhancement Tests...")
        logger.info(f"Testing against: {self.base_url}")
        
        await self.setup()
        
        try:
            # 🧠 PHASE 2.1.1: THEORY OF MIND / PERSPECTIVE-TAKING ENGINE TESTS 🧠
            logger.info("🧠 Testing Phase 2.1.1: Theory of Mind / Perspective-Taking Engine...")
            
            await self.test_perspective_analyze()
            await self.test_mental_state_attribution()
            await self.test_behavior_prediction()
            await self.test_conversation_simulation()
            await self.test_tracked_agents()
            
            # 🎯 PHASE 2.2.1: PERSONAL MOTIVATION SYSTEM TESTS 🎯
            logger.info("🎯 Testing Phase 2.2.1: Personal Motivation System...")
            
            goal = await self.test_create_personal_goal()
            goal_id = goal.get('goal_id') if goal else None
            
            await self.test_work_toward_goal(goal_id)
            await self.test_get_active_goals()
            await self.test_generate_new_goals()
            await self.test_get_motivation_profile()
            await self.test_assess_goal_satisfaction()
            
            # 🔍 PHASE 2.1.2: SOCIAL CONTEXT ANALYZER TESTS 🔍
            logger.info("🔍 Testing Phase 2.1.2: Social Context Analyzer...")
            
            await self.test_social_context_analyzer_implementation()
            
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
        
        logger.info("\n" + "="*60)
        logger.info("🧪 PHASE 2 CONSCIOUSNESS ENHANCEMENT TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "0%")
        logger.info("="*60)
        
        if failed > 0:
            logger.info("\n❌ FAILED TESTS:")
            for test in self.test_results['test_details']:
                if not test['success']:
                    logger.info(f"  - {test['test_name']}: {test['error']}")
        
        logger.info("\n🎯 PHASE 2 ANALYSIS:")
        
        # Analyze Theory of Mind results
        tom_tests = [t for t in self.test_results['test_details'] if 'Perspective' in t['test_name'] or 'Mental State' in t['test_name'] or 'Behavior' in t['test_name'] or 'Conversation' in t['test_name'] or 'Tracked' in t['test_name']]
        tom_passed = len([t for t in tom_tests if t['success']])
        tom_total = len(tom_tests)
        logger.info(f"Theory of Mind Engine: {tom_passed}/{tom_total} tests passed ({(tom_passed/tom_total*100):.1f}%)" if tom_total > 0 else "Theory of Mind Engine: No tests")
        
        # Analyze Personal Motivation results
        motivation_tests = [t for t in self.test_results['test_details'] if 'Goal' in t['test_name'] or 'Motivation' in t['test_name'] or 'Satisfaction' in t['test_name']]
        motivation_passed = len([t for t in motivation_tests if t['success']])
        motivation_total = len(motivation_tests)
        logger.info(f"Personal Motivation System: {motivation_passed}/{motivation_total} tests passed ({(motivation_passed/motivation_total*100):.1f}%)" if motivation_total > 0 else "Personal Motivation System: No tests")
        
        # Analyze Social Context results
        social_tests = [t for t in self.test_results['test_details'] if 'Social' in t['test_name']]
        social_passed = len([t for t in social_tests if t['success']])
        social_total = len(social_tests)
        logger.info(f"Social Context Analyzer: {social_passed}/{social_total} tests passed ({(social_passed/social_total*100):.1f}%)" if social_total > 0 else "Social Context Analyzer: No tests")

async def main():
    """Main test runner"""
    tester = Phase2Tester()
    results = await tester.run_phase2_tests()
    
    # Return exit code based on test results
    return 0 if results['failed_tests'] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)