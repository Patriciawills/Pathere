#!/usr/bin/env python3
"""
Data loader script to populate the Grammar & Vocabulary Engine with sample data
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.append(str(backend_dir))

# Import our modules
from core.learning_engine import LearningEngine
from core.knowledge_graph import KnowledgeGraph
from core.dataset_manager import DatasetManager

class DataLoader:
    """Load sample data into the learning system"""
    
    def __init__(self):
        self.learning_engine = LearningEngine()
        self.knowledge_graph = KnowledgeGraph()
        self.dataset_manager = DatasetManager()
        
    async def initialize(self):
        """Initialize all components"""
        print("Initializing learning components...")
        await self.learning_engine.initialize()
        await self.knowledge_graph.initialize()
        print("Components initialized successfully!")
        
    async def load_dictionary_data(self, file_path: str):
        """Load dictionary data from JSON file"""
        print(f"Loading dictionary data from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = data.get('entries', [])
            loaded_count = 0
            
            for entry in entries:
                # Convert to our internal format
                processed_data = {
                    'data_type': 'vocabulary',
                    'language': 'english',
                    'content': entry
                }
                
                # Learn the data
                result = await self.learning_engine.learn_from_data(processed_data)
                if result.get('success'):
                    # Add to knowledge graph
                    await self.knowledge_graph.add_entity(processed_data)
                    loaded_count += 1
                    print(f"✓ Loaded word: {entry.get('word', 'unknown')}")
                else:
                    print(f"✗ Failed to load word: {entry.get('word', 'unknown')}")
            
            print(f"Dictionary loading complete: {loaded_count}/{len(entries)} entries loaded")
            return loaded_count
            
        except Exception as e:
            print(f"Error loading dictionary data: {e}")
            return 0
    
    async def load_grammar_data(self, file_path: str):
        """Load grammar data from JSON file"""
        print(f"Loading grammar data from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = data.get('entries', [])
            loaded_count = 0
            
            for entry in entries:
                # Convert to our internal format
                processed_data = {
                    'data_type': 'grammar',
                    'language': 'english',
                    'content': entry
                }
                
                # Learn the data
                result = await self.learning_engine.learn_from_data(processed_data)
                if result.get('success'):
                    # Add to knowledge graph
                    await self.knowledge_graph.add_entity(processed_data)
                    loaded_count += 1
                    print(f"✓ Loaded rule: {entry.get('rule_name', 'unknown')}")
                else:
                    print(f"✗ Failed to load rule: {entry.get('rule_name', 'unknown')}")
            
            print(f"Grammar loading complete: {loaded_count}/{len(entries)} entries loaded")
            return loaded_count
            
        except Exception as e:
            print(f"Error loading grammar data: {e}")
            return 0
    
    async def test_queries(self):
        """Test some sample queries"""
        print("\nTesting sample queries...")
        
        test_queries = [
            ("language", "meaning"),
            ("learn", "meaning"),
            ("grammar", "meaning"),
            ("Present Simple Tense", "grammar"),
            ("vocabulary", "usage")
        ]
        
        for query_text, query_type in test_queries:
            try:
                result = await self.learning_engine.process_query(
                    query_text, 
                    "english", 
                    query_type
                )
                
                print(f"Query: '{query_text}' ({query_type})")
                if 'error' in result:
                    print(f"  ✗ Error: {result['error']}")
                else:
                    print(f"  ✓ Success (confidence: {result.get('confidence', 0):.2f})")
                    if query_type == "meaning" and 'definition' in result:
                        print(f"    Definition: {result['definition'][:100]}...")
                    elif query_type == "grammar" and 'rule' in result:
                        print(f"    Rule: {result['rule'][:100]}...")
                print()
                
            except Exception as e:
                print(f"  ✗ Query failed: {e}")
    
    async def show_stats(self):
        """Show system statistics"""
        print("=== SYSTEM STATISTICS ===")
        
        # Learning engine stats
        learning_stats = await self.learning_engine.get_stats()
        print(f"Learning Engine:")
        print(f"  Memory Usage: {learning_stats.get('memory_usage', 'N/A')}")
        print(f"  Total Words: {learning_stats.get('total_words', 0)}")
        print(f"  Total Rules: {learning_stats.get('total_rules', 0)}")
        print(f"  Successful Queries: {learning_stats.get('successful_queries', 0)}")
        print(f"  Failed Queries: {learning_stats.get('failed_queries', 0)}")
        
        # Knowledge graph stats
        graph_stats = await self.knowledge_graph.get_stats()
        print(f"Knowledge Graph:")
        print(f"  Total Nodes: {graph_stats.get('nodes_count', 0)}")
        print(f"  Total Edges: {graph_stats.get('edges_count', 0)}")
        print(f"  Languages: {', '.join(graph_stats.get('languages', []))}")
        print(f"  Average Connections: {graph_stats.get('average_connections', 0):.2f}")

async def main():
    """Main function"""
    print("🧠 Grammar & Vocabulary Engine - Data Loader")
    print("=" * 50)
    
    # Check if sample data files exist
    data_dir = Path(__file__).parent / 'data'
    dictionary_file = data_dir / 'english_dictionary_sample.json'
    grammar_file = data_dir / 'english_grammar_sample.json'
    
    if not dictionary_file.exists():
        print(f"❌ Dictionary file not found: {dictionary_file}")
        return
    
    if not grammar_file.exists():
        print(f"❌ Grammar file not found: {grammar_file}")
        return
    
    # Initialize loader
    loader = DataLoader()
    await loader.initialize()
    
    # Load sample data
    dict_count = await loader.load_dictionary_data(str(dictionary_file))
    grammar_count = await loader.load_grammar_data(str(grammar_file))
    
    # Test queries
    await loader.test_queries()
    
    # Show final stats
    await loader.show_stats()
    
    print("\n✅ Data loading completed!")
    print(f"📚 Dictionary entries loaded: {dict_count}")
    print(f"📖 Grammar rules loaded: {grammar_count}")
    print("\nYou can now test the system through the web interface!")

if __name__ == "__main__":
    asyncio.run(main())