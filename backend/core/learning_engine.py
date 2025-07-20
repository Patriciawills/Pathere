"""
Core Learning Engine - Human-like language learning system
Uses rule-based symbolic AI rather than probability-based prediction
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
import psutil
import os
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class LearningRule:
    """Represents a learned linguistic rule"""
    rule_id: str
    rule_type: str  # 'grammar', 'vocabulary', 'phonetic', 'semantic'
    language: str
    pattern: str
    description: str
    examples: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    usage_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type,
            'language': self.language,
            'pattern': self.pattern,
            'description': self.description,
            'examples': self.examples,
            'exceptions': self.exceptions,
            'confidence': self.confidence,
            'usage_count': self.usage_count,
            'last_used': self.last_used,
            'created_at': self.created_at
        }

@dataclass
class VocabularyEntry:
    """Represents a vocabulary item with rich linguistic information"""
    word: str
    language: str
    definitions: List[str]
    part_of_speech: str
    phonetic: str = ""
    examples: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    frequency: float = 0.0
    learning_stage: str = "new"  # new, learning, familiar, mastered
    associations: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'language': self.language,
            'definitions': self.definitions,
            'part_of_speech': self.part_of_speech,
            'phonetic': self.phonetic,
            'examples': self.examples,
            'synonyms': self.synonyms,
            'antonyms': self.antonyms,
            'frequency': self.frequency,
            'learning_stage': self.learning_stage,
            'associations': self.associations,
            'created_at': self.created_at
        }

class LearningEngine:
    """
    Core learning engine that mimics human language acquisition
    """
    
    def __init__(self):
        # Memory structures
        self.vocabulary: Dict[str, Dict[str, VocabularyEntry]] = defaultdict(dict)  # language -> word -> entry
        self.grammar_rules: Dict[str, List[LearningRule]] = defaultdict(list)  # language -> rules
        self.learning_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.error_memory: deque = deque(maxlen=1000)  # Remember mistakes for improvement
        
        # Learning state
        self.is_initialized = False
        self.learning_stats = {
            'total_words': 0,
            'total_rules': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'memory_usage': '0 MB',
            'last_learning_session': None
        }
        
        # Rule discovery system
        self.pattern_detector = PatternDetector()
        self.rule_generator = RuleGenerator()
        self.memory_manager = MemoryManager()
    
    async def initialize(self):
        """Initialize the learning engine"""
        try:
            logger.info("Initializing Learning Engine...")
            
            # Initialize sub-components
            await self.pattern_detector.initialize()
            await self.rule_generator.initialize()
            await self.memory_manager.initialize()
            
            # Load existing knowledge if available
            await self._load_existing_knowledge()
            
            self.is_initialized = True
            logger.info("Learning Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize learning engine: {str(e)}")
            raise
    
    async def learn_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from structured language data (human-like learning process)
        """
        try:
            data_type = data.get('data_type', 'unknown')
            language = data.get('language', 'english')
            content = data.get('content', {})
            
            if data_type in ['dictionary', 'vocabulary', 'word']:
                return await self._learn_vocabulary(content, language)
            elif data_type in ['grammar', 'rule']:
                return await self._learn_grammar_rules(content, language)
            elif data_type == 'text':
                return await self._learn_from_text(content, language)
            else:
                return {'success': False, 'error': f'Unknown data type: {data_type}'}
                
        except Exception as e:
            logger.error(f"Learning error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _learn_vocabulary(self, content: Dict[str, Any], language: str) -> Dict[str, Any]:
        """
        Learn vocabulary from dictionary entries
        """
        learned_words = 0
        skipped_words = 0
        
        # Handle both single entry and multiple entries
        if 'entries' in content:
            entries = content.get('entries', [])
        elif 'word' in content:
            # Single word entry
            entries = [content]
        else:
            return {'success': False, 'error': 'No vocabulary data found'}
        
        for entry in entries:
            try:
                word = entry.get('word', '').lower().strip()
                if not word:
                    continue
                
                # Handle both 'definition' (string) and 'definitions' (list)
                definitions = entry.get('definitions', [])
                if not definitions and 'definition' in entry:
                    definitions = [entry['definition']]
                elif isinstance(definitions, str):
                    definitions = [definitions]
                
                # Create vocabulary entry
                vocab_entry = VocabularyEntry(
                    word=word,
                    language=language,
                    definitions=definitions,
                    part_of_speech=entry.get('part_of_speech', 'unknown'),
                    phonetic=entry.get('phonetic', ''),
                    examples=entry.get('examples', []),
                    synonyms=entry.get('synonyms', []),
                    antonyms=entry.get('antonyms', [])
                )
                
                # Check if word already exists
                if word in self.vocabulary[language]:
                    # Merge information
                    existing = self.vocabulary[language][word]
                    existing.definitions.extend(vocab_entry.definitions)
                    existing.examples.extend(vocab_entry.examples)
                    existing.synonyms.extend(vocab_entry.synonyms)
                    existing.antonyms.extend(vocab_entry.antonyms)
                    
                    # Remove duplicates
                    existing.definitions = list(set(existing.definitions))
                    existing.examples = list(set(existing.examples))
                    existing.synonyms = list(set(existing.synonyms))
                    existing.antonyms = list(set(existing.antonyms))
                else:
                    self.vocabulary[language][word] = vocab_entry
                    learned_words += 1
                
                # Discover patterns and rules from this word
                await self._discover_word_patterns(vocab_entry, language)
                
            except Exception as e:
                logger.warning(f"Failed to learn word: {str(e)}")
                skipped_words += 1
                continue
        
        # Update stats
        self.learning_stats['total_words'] = sum(len(words) for words in self.vocabulary.values())
        self.learning_stats['last_learning_session'] = time.time()
        
        return {
            'success': True,
            'learned_words': learned_words,
            'skipped_words': skipped_words,
            'total_vocabulary': len(self.vocabulary[language])
        }
    
    async def _learn_grammar_rules(self, content: Dict[str, Any], language: str) -> Dict[str, Any]:
        """
        Learn grammar rules from structured grammar data
        """
        learned_rules = 0
        skipped_rules = 0
        
        # Handle both single entry and multiple entries
        if 'entries' in content:
            entries = content.get('entries', [])
        elif 'rule_name' in content:
            # Single rule entry
            entries = [content]
        else:
            return {'success': False, 'error': 'No grammar rule data found'}
        
        for rule_data in entries:
            try:
                rule_name = rule_data.get('rule_name', '')
                description = rule_data.get('description', '')
                examples = rule_data.get('examples', [])
                category = rule_data.get('category', 'general')
                
                if not rule_name or not description:
                    skipped_rules += 1
                    continue
                
                # Generate rule ID
                rule_id = hashlib.md5(f"{language}_{rule_name}_{description}".encode()).hexdigest()[:12]
                
                # Create learning rule
                learning_rule = LearningRule(
                    rule_id=rule_id,
                    rule_type=category,
                    language=language,
                    pattern=self._extract_pattern(description, examples),
                    description=description,
                    examples=examples,
                    exceptions=rule_data.get('exceptions', []),
                    confidence=0.8  # High confidence for explicitly provided rules
                )
                
                # Add to grammar rules
                self.grammar_rules[language].append(learning_rule)
                learned_rules += 1
                
                # Generate related rules through pattern analysis
                await self._generate_related_rules(learning_rule, language)
                
            except Exception as e:
                logger.warning(f"Failed to learn grammar rule: {str(e)}")
                skipped_rules += 1
                continue
        
        # Update stats
        self.learning_stats['total_rules'] = sum(len(rules) for rules in self.grammar_rules.values())
        self.learning_stats['last_learning_session'] = time.time()
        
        return {
            'success': True,
            'learned_rules': learned_rules,
            'skipped_rules': skipped_rules,
            'total_rules': len(self.grammar_rules[language])
        }
    
    async def process_query(self, query_text: str, language: str, query_type: str) -> Dict[str, Any]:
        """
        Process natural language queries using learned knowledge
        """
        start_time = time.time()
        
        try:
            if query_type == 'meaning':
                result = await self._process_meaning_query(query_text, language)
            elif query_type == 'grammar':
                result = await self._process_grammar_query(query_text, language)
            elif query_type == 'usage':
                result = await self._process_usage_query(query_text, language)
            else:
                result = {'error': f'Unknown query type: {query_type}'}
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if 'error' not in result:
                self.learning_stats['successful_queries'] += 1
                result['processing_time'] = processing_time
                result['confidence'] = result.get('confidence', 0.5)
            else:
                self.learning_stats['failed_queries'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            self.learning_stats['failed_queries'] += 1
            return {'error': str(e), 'processing_time': (time.time() - start_time) * 1000}
    
    async def _process_meaning_query(self, query_text: str, language: str) -> Dict[str, Any]:
        """
        Process word meaning queries
        """
        word = query_text.lower().strip()
        
        if word in self.vocabulary[language]:
            entry = self.vocabulary[language][word]
            entry.frequency += 1  # Track usage
            
            return {
                'definition': entry.definitions[0] if entry.definitions else 'No definition available',
                'part_of_speech': entry.part_of_speech,
                'phonetic': entry.phonetic,
                'examples': entry.examples[:3],  # Limit to 3 examples
                'synonyms': entry.synonyms[:5],  # Limit to 5 synonyms
                'confidence': min(0.9, 0.3 + entry.frequency * 0.1)  # Confidence based on usage
            }
        else:
            # Try fuzzy matching or morphological analysis
            similar_words = await self._find_similar_words(word, language)
            if similar_words:
                return {
                    'definition': f'Word not found. Did you mean: {", ".join(similar_words[:3])}?',
                    'suggestions': similar_words[:5],
                    'confidence': 0.3
                }
            else:
                return {
                    'definition': 'Word not found in vocabulary',
                    'confidence': 0.0
                }
    
    async def _process_grammar_query(self, query_text: str, language: str) -> Dict[str, Any]:
        """
        Process grammar-related queries
        """
        relevant_rules = []
        
        # Search for relevant grammar rules
        for rule in self.grammar_rules[language]:
            if self._is_rule_relevant(query_text, rule):
                rule.usage_count += 1
                rule.last_used = time.time()
                relevant_rules.append(rule)
        
        if relevant_rules:
            # Sort by relevance and confidence
            relevant_rules.sort(key=lambda r: (r.confidence, r.usage_count), reverse=True)
            best_rule = relevant_rules[0]
            
            return {
                'rule': best_rule.description,
                'examples': best_rule.examples[:3],
                'pattern': best_rule.pattern,
                'confidence': best_rule.confidence,
                'related_rules': [r.description for r in relevant_rules[1:3]]
            }
        else:
            return {
                'rule': 'No specific grammar rule found for this query',
                'confidence': 0.0
            }
    
    async def _process_usage_query(self, query_text: str, language: str) -> Dict[str, Any]:
        """
        Process usage example queries
        """
        word = query_text.lower().strip()
        
        if word in self.vocabulary[language]:
            entry = self.vocabulary[language][word]
            return {
                'examples': entry.examples,
                'contexts': self._generate_usage_contexts(entry),
                'confidence': 0.8
            }
        else:
            return {
                'examples': [],
                'error': 'Word not found',
                'confidence': 0.0
            }
    
    async def process_feedback(self, query_record: Dict[str, Any], correction: str, feedback_type: str) -> Dict[str, Any]:
        """
        Process feedback to improve learning
        """
        try:
            # Store error for learning
            error_record = {
                'query': query_record['query_text'],
                'language': query_record['language'],
                'original_result': query_record['result'],
                'correction': correction,
                'feedback_type': feedback_type,
                'timestamp': time.time()
            }
            
            self.error_memory.append(error_record)
            
            # Learn from the correction
            improvements = await self._learn_from_feedback(error_record)
            
            return {
                'success': True,
                'improvements': improvements,
                'update_graph': len(improvements) > 0
            }
            
        except Exception as e:
            logger.error(f"Feedback processing error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get learning engine statistics
        """
        # Update memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        self.learning_stats['memory_usage'] = f"{memory_usage:.1f} MB"
        
        # Calculate vocabulary statistics
        vocab_stats = {}
        for lang, words in self.vocabulary.items():
            vocab_stats[lang] = {
                'total_words': len(words),
                'mastered_words': len([w for w in words.values() if w.learning_stage == 'mastered']),
                'learning_words': len([w for w in words.values() if w.learning_stage == 'learning']),
                'new_words': len([w for w in words.values() if w.learning_stage == 'new'])
            }
        
        return {
            **self.learning_stats,
            'vocabulary_by_language': vocab_stats,
            'rules_count': sum(len(rules) for rules in self.grammar_rules.values()),
            'error_patterns': len(self.error_memory),
            'initialization_status': self.is_initialized
        }
    
    # Helper methods
    def _extract_pattern(self, description: str, examples: List[str]) -> str:
        """Extract learnable pattern from rule description and examples"""
        # Simplified pattern extraction - can be enhanced
        return f"PATTERN: {description[:50]}..." if len(description) > 50 else f"PATTERN: {description}"
    
    async def _discover_word_patterns(self, vocab_entry: VocabularyEntry, language: str):
        """Discover patterns from vocabulary entries"""
        # This would implement morphological analysis, phonetic patterns, etc.
        pass
    
    async def _generate_related_rules(self, rule: LearningRule, language: str):
        """Generate related rules through pattern analysis"""
        # This would implement rule induction and generalization
        pass
    
    def _is_rule_relevant(self, query: str, rule: LearningRule) -> bool:
        """Check if a grammar rule is relevant to the query"""
        query_lower = query.lower()
        rule_lower = rule.description.lower()
        pattern_lower = rule.pattern.lower()
        
        # Simple keyword matching - can be enhanced
        common_words = set(query_lower.split()) & set(rule_lower.split() + pattern_lower.split())
        return len(common_words) > 0
    
    async def _find_similar_words(self, word: str, language: str) -> List[str]:
        """Find similar words using edit distance"""
        similar = []
        for vocab_word in self.vocabulary[language].keys():
            if self._edit_distance(word, vocab_word) <= 2:
                similar.append(vocab_word)
        return sorted(similar, key=lambda w: self._edit_distance(word, w))
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _generate_usage_contexts(self, entry: VocabularyEntry) -> List[str]:
        """Generate contextual usage examples"""
        contexts = []
        pos = entry.part_of_speech.lower()
        word = entry.word
        
        if 'noun' in pos:
            contexts.append(f"The {word} is important in this context.")
        elif 'verb' in pos:
            contexts.append(f"You can {word} this effectively.")
        elif 'adjective' in pos:
            contexts.append(f"This is very {word}.")
        
        return contexts
    
    async def _learn_from_feedback(self, error_record: Dict[str, Any]) -> List[str]:
        """Learn from user feedback and corrections"""
        improvements = []
        
        query = error_record['query']
        correction = error_record['correction']
        language = error_record['language']
        
        # If it's a vocabulary correction
        if error_record['feedback_type'] == 'error':
            word = query.lower().strip()
            if word in self.vocabulary[language]:
                # Update the definition based on correction
                entry = self.vocabulary[language][word]
                if correction not in entry.definitions:
                    entry.definitions.insert(0, correction)
                    improvements.append(f"Updated definition for '{word}'")
        
        return improvements
    
    async def _load_existing_knowledge(self):
        """Load existing knowledge from persistent storage"""
        # This would load from files or database
        pass


# Supporting classes
class PatternDetector:
    """Detects linguistic patterns in data"""
    
    async def initialize(self):
        pass

class RuleGenerator:
    """Generates new rules from patterns"""
    
    async def initialize(self):
        pass

class MemoryManager:
    """Manages memory usage and optimization"""
    
    async def initialize(self):
        pass