#!/usr/bin/env python3
"""
Detailed Learning Engine Test - Focus on vocabulary learning issue
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningEngineDetailedTest:
    def __init__(self):
        # Get backend URL from environment
        env_path = Path('/app/frontend/.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        url = line.split('=', 1)[1].strip()
                        self.base_url = f"{url}/api"
                        break
        else:
            self.base_url = "http://localhost:8001/api"
        
        self.session = None
    
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
    
    async def test_vocabulary_learning_detailed(self):
        """Test vocabulary learning in detail"""
        logger.info("🔍 Testing Vocabulary Learning Engine in Detail...")
        
        # Test 1: Add multiple vocabulary entries
        test_words = [
            {
                "word": "ephemeral",
                "definitions": ["Lasting for a very short time"],
                "part_of_speech": "adjective",
                "phonetic": "/ɪˈfem(ə)rəl/",
                "examples": ["The beauty of cherry blossoms is ephemeral"],
                "synonyms": ["transient", "fleeting", "temporary"],
                "antonyms": ["permanent", "lasting", "enduring"]
            },
            {
                "word": "ubiquitous",
                "definitions": ["Present, appearing, or found everywhere"],
                "part_of_speech": "adjective",
                "phonetic": "/yo͞oˈbikwədəs/",
                "examples": ["Smartphones have become ubiquitous in modern society"],
                "synonyms": ["omnipresent", "pervasive", "universal"],
                "antonyms": ["rare", "scarce", "absent"]
            },
            {
                "word": "perspicacious",
                "definitions": ["Having a ready insight into and understanding of things"],
                "part_of_speech": "adjective",
                "phonetic": "/ˌpərspɪˈkeɪʃəs/",
                "examples": ["Her perspicacious analysis of the situation impressed everyone"],
                "synonyms": ["perceptive", "astute", "shrewd"],
                "antonyms": ["obtuse", "dull", "unperceptive"]
            }
        ]
        
        added_words = []
        for word_data in test_words:
            payload = {
                "data_type": "vocabulary",
                "language": "english",
                "content": word_data
            }
            
            async with self.session.post(f"{self.base_url}/add-data",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Added word: {word_data['word']} - ID: {result.get('data_id')}")
                    added_words.append(word_data['word'])
                else:
                    logger.error(f"❌ Failed to add word: {word_data['word']}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Test 2: Query each added word
        successful_queries = 0
        failed_queries = 0
        
        for word in added_words:
            payload = {
                "query_text": word,
                "language": "english",
                "query_type": "meaning"
            }
            
            async with self.session.post(f"{self.base_url}/query",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    query_result = result.get('result', {})
                    
                    if 'definition' in query_result and query_result['definition'] != 'Word not found in vocabulary':
                        logger.info(f"✅ Successfully queried: {word}")
                        logger.info(f"   Definition: {query_result.get('definition', 'N/A')}")
                        logger.info(f"   Confidence: {query_result.get('confidence', 0)}")
                        successful_queries += 1
                    else:
                        logger.warning(f"⚠️  Word found but no definition: {word}")
                        failed_queries += 1
                else:
                    logger.error(f"❌ Failed to query word: {word}")
                    failed_queries += 1
        
        # Test 3: Check learning engine statistics
        async with self.session.get(f"{self.base_url}/stats") as response:
            if response.status == 200:
                stats = await response.json()
                learning_stats = stats.get('learning_engine', {})
                
                logger.info("\n📊 Learning Engine Statistics:")
                logger.info(f"   Total Words: {learning_stats.get('total_words', 0)}")
                logger.info(f"   Total Rules: {learning_stats.get('total_rules', 0)}")
                logger.info(f"   Successful Queries: {learning_stats.get('successful_queries', 0)}")
                logger.info(f"   Failed Queries: {learning_stats.get('failed_queries', 0)}")
                logger.info(f"   Memory Usage: {learning_stats.get('memory_usage', 'Unknown')}")
                
                vocab_by_lang = learning_stats.get('vocabulary_by_language', {})
                if 'english' in vocab_by_lang:
                    eng_stats = vocab_by_lang['english']
                    logger.info(f"   English Vocabulary:")
                    logger.info(f"     - Total: {eng_stats.get('total_words', 0)}")
                    logger.info(f"     - New: {eng_stats.get('new_words', 0)}")
                    logger.info(f"     - Learning: {eng_stats.get('learning_words', 0)}")
                    logger.info(f"     - Mastered: {eng_stats.get('mastered_words', 0)}")
        
        # Test 4: Test grammar rules (to verify they work as mentioned)
        grammar_payload = {
            "data_type": "grammar",
            "language": "english",
            "content": {
                "rule_name": "Subjunctive Mood",
                "description": "Used to express hypothetical or non-factual situations",
                "category": "mood",
                "examples": [
                    "If I were you, I would study harder",
                    "I suggest that he be more careful",
                    "It's important that she arrive on time"
                ],
                "pattern": "If + subject + were/past form, subject + would + base verb"
            }
        }
        
        async with self.session.post(f"{self.base_url}/add-data",
                                   json=grammar_payload,
                                   headers={'Content-Type': 'application/json'}) as response:
            if response.status == 200:
                logger.info("✅ Grammar rule added successfully")
                
                # Query the grammar rule
                grammar_query = {
                    "query_text": "subjunctive mood",
                    "language": "english",
                    "query_type": "grammar"
                }
                
                await asyncio.sleep(1)
                
                async with self.session.post(f"{self.base_url}/query",
                                           json=grammar_query,
                                           headers={'Content-Type': 'application/json'}) as response:
                    if response.status == 200:
                        result = await response.json()
                        grammar_result = result.get('result', {})
                        
                        if 'rule' in grammar_result and grammar_result['rule'] != 'No specific grammar rule found for this query':
                            logger.info("✅ Grammar rule query successful")
                            logger.info(f"   Rule: {grammar_result.get('rule', 'N/A')}")
                        else:
                            logger.warning("⚠️  Grammar rule not found in query")
            else:
                logger.error("❌ Failed to add grammar rule")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎯 VOCABULARY LEARNING TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Words Added: {len(added_words)}")
        logger.info(f"Successful Queries: {successful_queries}")
        logger.info(f"Failed Queries: {failed_queries}")
        
        if successful_queries == len(added_words):
            logger.info("✅ VOCABULARY LEARNING: WORKING CORRECTLY")
            return True
        elif successful_queries > 0:
            logger.info("⚠️  VOCABULARY LEARNING: PARTIALLY WORKING")
            return "partial"
        else:
            logger.info("❌ VOCABULARY LEARNING: NOT WORKING")
            return False
    
    async def run_test(self):
        """Run the detailed learning engine test"""
        await self.setup()
        try:
            result = await self.test_vocabulary_learning_detailed()
            return result
        finally:
            await self.teardown()

async def main():
    tester = LearningEngineDetailedTest()
    result = await tester.run_test()
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nTest Result: {result}")