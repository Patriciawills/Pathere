#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Grammar & Vocabulary Engine
Tests all REST endpoints and core functionality
"""

import asyncio
import aiohttp
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendTester:
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
    
    async def test_root_endpoint(self):
        """Test the root API endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    if 'message' in data and 'version' in data:
                        self.log_test_result("Root Endpoint", True, f"Response: {data}")
                        return True
                    else:
                        self.log_test_result("Root Endpoint", False, error="Missing required fields in response")
                        return False
                else:
                    self.log_test_result("Root Endpoint", False, error=f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test_result("Root Endpoint", False, error=str(e))
            return False
    
    async def test_stats_endpoint(self):
        """Test the statistics endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['database', 'learning_engine', 'knowledge_graph', 'system']
                    
                    if all(field in data for field in required_fields):
                        self.log_test_result("Stats Endpoint", True, f"Stats retrieved successfully")
                        return True
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_test_result("Stats Endpoint", False, error=f"Missing fields: {missing}")
                        return False
                else:
                    self.log_test_result("Stats Endpoint", False, error=f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test_result("Stats Endpoint", False, error=str(e))
            return False
    
    async def test_pdf_upload(self):
        """Test PDF upload functionality"""
        try:
            # Create a simple test PDF content (mock)
            test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(test_content)
                tmp_file.flush()
                
                # Upload the file
                data = aiohttp.FormData()
                data.add_field('file', 
                              open(tmp_file.name, 'rb'),
                              filename='test_dictionary.pdf',
                              content_type='application/pdf')
                
                async with self.session.post(f"{self.base_url}/upload-pdf", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'file_id' in result and 'status' in result:
                            self.log_test_result("PDF Upload", True, f"File uploaded with ID: {result['file_id']}")
                            # Clean up
                            os.unlink(tmp_file.name)
                            return result['file_id']
                        else:
                            self.log_test_result("PDF Upload", False, error="Missing required fields in response")
                            os.unlink(tmp_file.name)
                            return None
                    else:
                        error_text = await response.text()
                        self.log_test_result("PDF Upload", False, error=f"HTTP {response.status}: {error_text}")
                        os.unlink(tmp_file.name)
                        return None
                        
        except Exception as e:
            self.log_test_result("PDF Upload", False, error=str(e))
            return None
    
    async def test_pdf_processing(self, file_id: str):
        """Test PDF processing functionality"""
        if not file_id:
            self.log_test_result("PDF Processing", False, error="No file ID provided")
            return False
            
        try:
            payload = {
                "pdf_file_id": file_id,
                "page_number": 0,
                "processing_type": "dictionary"
            }
            
            async with self.session.post(f"{self.base_url}/process-pdf", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'processing_id' in result and 'status' in result:
                        self.log_test_result("PDF Processing", True, f"PDF processed with ID: {result['processing_id']}")
                        return True
                    else:
                        self.log_test_result("PDF Processing", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("PDF Processing", False, error=f"HTTP {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            self.log_test_result("PDF Processing", False, error=str(e))
            return False
    
    async def test_add_vocabulary_data(self):
        """Test adding vocabulary data"""
        try:
            payload = {
                "data_type": "vocabulary",
                "language": "english",
                "content": {
                    "word": "serendipity",
                    "definitions": ["The occurrence and development of events by chance in a happy or beneficial way"],
                    "part_of_speech": "noun",
                    "phonetic": "/ˌserənˈdipədē/",
                    "examples": ["A fortunate stroke of serendipity brought the two old friends together"],
                    "synonyms": ["chance", "fortune", "luck"],
                    "antonyms": ["misfortune", "bad luck"]
                }
            }
            
            async with self.session.post(f"{self.base_url}/add-data",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'data_id' in result and 'status' in result:
                        self.log_test_result("Add Vocabulary Data", True, f"Data added with ID: {result['data_id']}")
                        return True
                    else:
                        self.log_test_result("Add Vocabulary Data", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Add Vocabulary Data", False, error=f"HTTP {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            self.log_test_result("Add Vocabulary Data", False, error=str(e))
            return False
    
    async def test_add_grammar_data(self):
        """Test adding grammar rule data"""
        try:
            payload = {
                "data_type": "grammar",
                "language": "english",
                "content": {
                    "rule_name": "Present Perfect Tense",
                    "description": "Used to describe actions that started in the past and continue to the present, or actions completed at an unspecified time",
                    "category": "tense",
                    "examples": [
                        "I have lived here for five years",
                        "She has finished her homework",
                        "They have never been to Paris"
                    ],
                    "exceptions": ["Some irregular verbs have unique past participle forms"],
                    "pattern": "Subject + have/has + past participle"
                }
            }
            
            async with self.session.post(f"{self.base_url}/add-data",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'data_id' in result and 'status' in result:
                        self.log_test_result("Add Grammar Data", True, f"Grammar rule added with ID: {result['data_id']}")
                        return True
                    else:
                        self.log_test_result("Add Grammar Data", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Add Grammar Data", False, error=f"HTTP {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            self.log_test_result("Add Grammar Data", False, error=str(e))
            return False
    
    async def test_meaning_query(self):
        """Test meaning query functionality"""
        try:
            payload = {
                "query_text": "serendipity",
                "language": "english",
                "query_type": "meaning"
            }
            
            async with self.session.post(f"{self.base_url}/query",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'query_id' in result and 'result' in result:
                        self.log_test_result("Meaning Query", True, f"Query processed with ID: {result['query_id']}")
                        return result['query_id']
                    else:
                        self.log_test_result("Meaning Query", False, error="Missing required fields in response")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Meaning Query", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Meaning Query", False, error=str(e))
            return None
    
    async def test_grammar_query(self):
        """Test grammar query functionality"""
        try:
            payload = {
                "query_text": "present perfect tense",
                "language": "english",
                "query_type": "grammar"
            }
            
            async with self.session.post(f"{self.base_url}/query",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'query_id' in result and 'result' in result:
                        self.log_test_result("Grammar Query", True, f"Grammar query processed with ID: {result['query_id']}")
                        return result['query_id']
                    else:
                        self.log_test_result("Grammar Query", False, error="Missing required fields in response")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Grammar Query", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Grammar Query", False, error=str(e))
            return None
    
    async def test_usage_query(self):
        """Test usage query functionality"""
        try:
            payload = {
                "query_text": "serendipity",
                "language": "english",
                "query_type": "usage"
            }
            
            async with self.session.post(f"{self.base_url}/query",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'query_id' in result and 'result' in result:
                        self.log_test_result("Usage Query", True, f"Usage query processed with ID: {result['query_id']}")
                        return result['query_id']
                    else:
                        self.log_test_result("Usage Query", False, error="Missing required fields in response")
                        return None
                else:
                    error_text = await response.text()
                    self.log_test_result("Usage Query", False, error=f"HTTP {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.log_test_result("Usage Query", False, error=str(e))
            return None
    
    async def test_feedback_submission(self, query_id: str):
        """Test feedback submission functionality"""
        if not query_id:
            self.log_test_result("Feedback Submission", False, error="No query ID provided")
            return False
            
        try:
            payload = {
                "query_id": query_id,
                "correction": "A pleasant surprise or fortunate discovery made by accident",
                "feedback_type": "improvement"
            }
            
            async with self.session.post(f"{self.base_url}/feedback",
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'feedback_id' in result and 'status' in result:
                        self.log_test_result("Feedback Submission", True, f"Feedback submitted with ID: {result['feedback_id']}")
                        return True
                    else:
                        self.log_test_result("Feedback Submission", False, error="Missing required fields in response")
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result("Feedback Submission", False, error=f"HTTP {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            self.log_test_result("Feedback Submission", False, error=str(e))
            return False
    
    async def test_learning_engine_vocabulary_issue(self):
        """Test the specific vocabulary learning issue mentioned in test_result.md"""
        try:
            # First add some vocabulary data
            await self.test_add_vocabulary_data()
            
            # Wait a moment for processing
            await asyncio.sleep(1)
            
            # Try to query the added vocabulary
            query_id = await self.test_meaning_query()
            
            if query_id:
                self.log_test_result("Learning Engine Vocabulary", True, "Vocabulary learning appears to be working")
                return True
            else:
                self.log_test_result("Learning Engine Vocabulary", False, error="Vocabulary not properly learned or queryable")
                return False
                
        except Exception as e:
            self.log_test_result("Learning Engine Vocabulary", False, error=str(e))
            return False
    
    async def run_all_tests(self):
        """Run all backend tests"""
        logger.info("🚀 Starting Backend API Tests...")
        logger.info(f"Testing against: {self.base_url}")
        
        await self.setup()
        
        try:
            # Basic connectivity tests
            await self.test_root_endpoint()
            await self.test_stats_endpoint()
            
            # PDF processing tests
            file_id = await self.test_pdf_upload()
            if file_id:
                await self.test_pdf_processing(file_id)
            
            # Data addition tests
            await self.test_add_vocabulary_data()
            await self.test_add_grammar_data()
            
            # Query tests
            meaning_query_id = await self.test_meaning_query()
            grammar_query_id = await self.test_grammar_query()
            usage_query_id = await self.test_usage_query()
            
            # Feedback test
            if meaning_query_id:
                await self.test_feedback_submission(meaning_query_id)
            
            # Specific issue test
            await self.test_learning_engine_vocabulary_issue()
            
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
        logger.info("🧪 BACKEND API TEST SUMMARY")
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
        
        logger.info("\n🎯 CRITICAL ISSUES FOUND:" if failed > 0 else "\n✅ ALL TESTS PASSED!")

async def main():
    """Main test runner"""
    tester = BackendTester()
    results = await tester.run_all_tests()
    
    # Return exit code based on test results
    return 0 if results['failed_tests'] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)