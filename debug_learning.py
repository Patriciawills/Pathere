#!/usr/bin/env python3
"""
Debug test to see what's happening in the learning process
"""

import asyncio
import aiohttp
import json
from pathlib import Path

async def debug_learning_process():
    # Get backend URL
    env_path = Path('/app/frontend/.env')
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('REACT_APP_BACKEND_URL='):
                url = line.split('=', 1)[1].strip()
                base_url = f"{url}/api"
                break
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        # Test 1: Add a simple word
        print("🔍 Adding a simple word...")
        payload = {
            "data_type": "vocabulary",
            "language": "english",
            "content": {
                "word": "test",
                "definitions": ["A simple test word"],
                "part_of_speech": "noun"
            }
        }
        
        async with session.post(f"{base_url}/add-data", json=payload) as response:
            result = await response.json()
            print(f"Add data response: {json.dumps(result, indent=2)}")
        
        # Test 2: Check stats immediately
        print("\n📊 Checking stats after adding word...")
        async with session.get(f"{base_url}/stats") as response:
            stats = await response.json()
            learning_stats = stats.get('learning_engine', {})
            print(f"Learning engine stats: {json.dumps(learning_stats, indent=2)}")
        
        # Test 3: Try to query the word
        print("\n🔍 Querying the word...")
        query_payload = {
            "query_text": "test",
            "language": "english",
            "query_type": "meaning"
        }
        
        async with session.post(f"{base_url}/query", json=query_payload) as response:
            query_result = await response.json()
            print(f"Query result: {json.dumps(query_result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(debug_learning_process())