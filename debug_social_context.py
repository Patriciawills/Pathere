#!/usr/bin/env python3
"""
Debug Social Context Analyzer Testing
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialContextDebugTester:
    def __init__(self):
        self.base_url = self._get_backend_url()
        self.session = None
        
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

    async def debug_social_context_analyze(self):
        """Debug the social context analyze endpoint"""
        try:
            payload = {
                "user_id": "debug_user_001",
                "interaction_data": {
                    "content_type": "text",
                    "topic": "general_inquiry",
                    "sentiment": 0.6,
                    "satisfaction": 0.7,
                    "relationship_type": "stranger"
                }
            }
            
            logger.info(f"Testing POST {self.base_url}/consciousness/social/analyze")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            async with self.session.post(f"{self.base_url}/consciousness/social/analyze",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                logger.info(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Response Body: {json.dumps(result, indent=2)}")
                    
                    if 'social_context_analysis' in result:
                        analysis = result['social_context_analysis']
                        logger.info(f"Analysis Keys: {list(analysis.keys())}")
                        logger.info(f"Relationship Type: {analysis.get('relationship_type')}")
                        logger.info(f"Communication Style: {analysis.get('communication_style')}")
                    else:
                        logger.error("Missing 'social_context_analysis' in response")
                        
                else:
                    error_text = await response.text()
                    logger.error(f"Error Response: {error_text}")
                    
        except Exception as e:
            logger.error(f"Exception during test: {e}")

    async def debug_relationship_insights(self):
        """Debug the relationship insights endpoint"""
        try:
            user_id = "debug_user_002"
            
            logger.info(f"Testing GET {self.base_url}/consciousness/social/relationship/{user_id}")
            
            async with self.session.get(f"{self.base_url}/consciousness/social/relationship/{user_id}") as response:
                logger.info(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Response Body: {json.dumps(result, indent=2)}")
                    
                    if 'relationship_insights' in result:
                        insights = result['relationship_insights']
                        logger.info(f"Insights Keys: {list(insights.keys())}")
                        logger.info(f"Insights Type: {type(insights)}")
                    else:
                        logger.error("Missing 'relationship_insights' in response")
                        
                else:
                    error_text = await response.text()
                    logger.error(f"Error Response: {error_text}")
                    
        except Exception as e:
            logger.error(f"Exception during test: {e}")

    async def run_debug_tests(self):
        """Run debug tests"""
        logger.info("🔍 Starting Social Context Analyzer Debug Tests...")
        logger.info(f"Testing against: {self.base_url}")
        
        await self.setup()
        
        try:
            await self.debug_social_context_analyze()
            await self.debug_relationship_insights()
            
        finally:
            await self.teardown()

async def main():
    """Main debug runner"""
    tester = SocialContextDebugTester()
    await tester.run_debug_tests()

if __name__ == "__main__":
    asyncio.run(main())