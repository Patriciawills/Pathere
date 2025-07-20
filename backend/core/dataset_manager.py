"""
Dataset Manager for handling structured language data
Flexible system for converting various data formats into learnable structures
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import hashlib
import time
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)

@dataclass
class DatasetSchema:
    """Schema definition for dataset validation"""
    name: str
    version: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

class DatasetManager:
    """
    Manages conversion of various data formats into structured learning datasets
    """
    
    def __init__(self):
        # Supported data schemas
        self.schemas = {
            'dictionary_entry': DatasetSchema(
                name='dictionary_entry',
                version='1.0',
                required_fields=['word', 'definition'],
                optional_fields=['phonetic', 'part_of_speech', 'examples', 'synonyms', 'antonyms', 'etymology'],
                field_types={
                    'word': str,
                    'definition': str,
                    'phonetic': str,
                    'part_of_speech': str,
                    'examples': list,
                    'synonyms': list,
                    'antonyms': list,
                    'etymology': str
                },
                validation_rules={
                    'word': {'min_length': 1, 'max_length': 100},
                    'definition': {'min_length': 5, 'max_length': 1000}
                }
            ),
            'grammar_rule': DatasetSchema(
                name='grammar_rule',
                version='1.0',
                required_fields=['rule_name', 'description'],
                optional_fields=['category', 'examples', 'exceptions', 'difficulty_level'],
                field_types={
                    'rule_name': str,
                    'description': str,
                    'category': str,
                    'examples': list,
                    'exceptions': list,
                    'difficulty_level': str
                },
                validation_rules={
                    'rule_name': {'min_length': 3, 'max_length': 200},
                    'description': {'min_length': 10, 'max_length': 2000}
                }
            ),
            'text_corpus': DatasetSchema(
                name='text_corpus',
                version='1.0',
                required_fields=['text', 'language'],
                optional_fields=['source', 'genre', 'difficulty', 'metadata'],
                field_types={
                    'text': str,
                    'language': str,
                    'source': str,
                    'genre': str,
                    'difficulty': str,
                    'metadata': dict
                }
            )
        }
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'duplicates_removed': 0,
            'validation_errors': 0
        }
        
        # Deduplication cache
        self.content_hashes = set()
        
        # Text processing patterns
        self.text_patterns = {
            'word_boundary': re.compile(r'\b\w+\b'),
            'sentence_boundary': re.compile(r'[.!?]+'),
            'phonetic_symbols': re.compile(r'\[([^\]]+)\]'),
            'pos_tags': re.compile(r'\(([^)]+)\)'),
            'example_markers': re.compile(r'^(?:\d+\.|\-|\•)\s*(.+)$'),
            'definition_markers': re.compile(r'^(?:Definition|Meaning|Def):\s*(.+)$', re.IGNORECASE)
        }
    
    async def structure_extracted_data(self, extracted_data: Dict[str, Any], processing_type: str) -> Dict[str, Any]:
        """
        Convert extracted OCR data into structured dataset format
        """
        try:
            if processing_type == "dictionary":
                return await self._structure_dictionary_data(extracted_data)
            elif processing_type == "grammar":
                return await self._structure_grammar_data(extracted_data)
            elif processing_type == "text":
                return await self._structure_text_data(extracted_data)
            else:
                return {'error': f'Unsupported processing type: {processing_type}'}
                
        except Exception as e:
            logger.error(f"Data structuring error: {str(e)}")
            return {'error': str(e)}
    
    async def _structure_dictionary_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure dictionary data according to schema"""
        try:
            structured_entries = []
            raw_entries = extracted_data.get('structured_data', {}).get('entries', [])
            
            for raw_entry in raw_entries:
                try:
                    # Clean and validate entry
                    cleaned_entry = await self._clean_dictionary_entry(raw_entry)
                    
                    if cleaned_entry and await self._validate_entry(cleaned_entry, 'dictionary_entry'):
                        # Check for duplicates
                        content_hash = self._generate_content_hash(cleaned_entry)
                        if content_hash not in self.content_hashes:
                            self.content_hashes.add(content_hash)
                            structured_entries.append(cleaned_entry)
                            self.processing_stats['successful'] += 1
                        else:
                            self.processing_stats['duplicates_removed'] += 1
                    else:
                        self.processing_stats['validation_errors'] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process dictionary entry: {str(e)}")
                    self.processing_stats['failed'] += 1
                    continue
            
            self.processing_stats['total_processed'] += len(raw_entries)
            
            return {
                'entries': structured_entries,
                'entry_count': len(structured_entries),
                'processing_stats': self.processing_stats.copy(),
                'schema': 'dictionary_entry',
                'version': self.schemas['dictionary_entry'].version
            }
            
        except Exception as e:
            logger.error(f"Dictionary data structuring error: {str(e)}")
            return {'error': str(e)}
    
    async def _structure_grammar_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure grammar data according to schema"""
        try:
            structured_rules = []
            raw_entries = extracted_data.get('structured_data', {}).get('entries', [])
            
            for raw_rule in raw_entries:
                try:
                    # Clean and validate rule
                    cleaned_rule = await self._clean_grammar_rule(raw_rule)
                    
                    if cleaned_rule and await self._validate_entry(cleaned_rule, 'grammar_rule'):
                        # Check for duplicates
                        content_hash = self._generate_content_hash(cleaned_rule)
                        if content_hash not in self.content_hashes:
                            self.content_hashes.add(content_hash)
                            structured_rules.append(cleaned_rule)
                            self.processing_stats['successful'] += 1
                        else:
                            self.processing_stats['duplicates_removed'] += 1
                    else:
                        self.processing_stats['validation_errors'] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process grammar rule: {str(e)}")
                    self.processing_stats['failed'] += 1
                    continue
            
            self.processing_stats['total_processed'] += len(raw_entries)
            
            return {
                'entries': structured_rules,
                'rule_count': len(structured_rules),
                'processing_stats': self.processing_stats.copy(),
                'schema': 'grammar_rule',
                'version': self.schemas['grammar_rule'].version
            }
            
        except Exception as e:
            logger.error(f"Grammar data structuring error: {str(e)}")
            return {'error': str(e)}
    
    async def _structure_text_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure general text data for corpus building"""
        try:
            text_content = extracted_data.get('raw_text', '')
            
            # Split into manageable chunks
            chunks = self._split_text_into_chunks(text_content)
            
            structured_chunks = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only process substantial chunks
                    cleaned_chunk = {
                        'text': chunk.strip(),
                        'language': 'english',  # Default, can be detected
                        'word_count': len(self.text_patterns['word_boundary'].findall(chunk)),
                        'sentence_count': len(self.text_patterns['sentence_boundary'].split(chunk)),
                        'source': 'ocr_extraction',
                        'processed_at': time.time()
                    }
                    
                    if await self._validate_entry(cleaned_chunk, 'text_corpus'):
                        structured_chunks.append(cleaned_chunk)
            
            return {
                'entries': structured_chunks,
                'chunk_count': len(structured_chunks),
                'total_words': sum(chunk['word_count'] for chunk in structured_chunks),
                'total_sentences': sum(chunk['sentence_count'] for chunk in structured_chunks),
                'schema': 'text_corpus',
                'version': self.schemas['text_corpus'].version
            }
            
        except Exception as e:
            logger.error(f"Text data structuring error: {str(e)}")
            return {'error': str(e)}
    
    async def validate_and_process(self, data_type: str, language: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process manually submitted data
        """
        try:
            # Determine schema based on data type
            schema_name = self._map_data_type_to_schema(data_type)
            if not schema_name:
                return {'error': f'Unsupported data type: {data_type}'}
            
            # Add metadata
            processed_content = {
                **content,
                'data_type': data_type,
                'language': language,
                'processed_at': time.time(),
                'source': 'manual_input'
            }
            
            # Validate against schema
            if await self._validate_entry(processed_content, schema_name):
                return processed_content
            else:
                return {'error': 'Validation failed'}
                
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return {'error': str(e)}
    
    async def _clean_dictionary_entry(self, raw_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and standardize dictionary entry"""
        try:
            word = raw_entry.get('word', '').strip().lower()
            definition = raw_entry.get('definition', '').strip()
            
            if not word or not definition or len(word) < 1:
                return None
            
            # Clean definition
            definition = self._clean_text(definition)
            if len(definition) < 5:
                return None
            
            # Extract and clean other fields
            phonetic = self._extract_phonetic(raw_entry.get('phonetic', ''))
            part_of_speech = self._clean_pos(raw_entry.get('part_of_speech', ''))
            examples = self._clean_examples(raw_entry.get('examples', []))
            synonyms = self._clean_word_list(raw_entry.get('synonyms', []))
            antonyms = self._clean_word_list(raw_entry.get('antonyms', []))
            
            return {
                'word': word,
                'definition': definition,
                'phonetic': phonetic,
                'part_of_speech': part_of_speech,
                'examples': examples,
                'synonyms': synonyms,
                'antonyms': antonyms,
                'processed_at': time.time()
            }
            
        except Exception as e:
            logger.warning(f"Dictionary entry cleaning error: {str(e)}")
            return None
    
    async def _clean_grammar_rule(self, raw_rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and standardize grammar rule"""
        try:
            rule_name = raw_rule.get('rule_name', '').strip()
            description = raw_rule.get('description', '').strip()
            
            if not rule_name or not description:
                return None
            
            # Clean description
            description = self._clean_text(description)
            if len(description) < 10:
                return None
            
            # Clean other fields
            category = raw_rule.get('category', 'general').lower()
            examples = self._clean_examples(raw_rule.get('examples', []))
            exceptions = self._clean_examples(raw_rule.get('exceptions', []))
            
            return {
                'rule_name': rule_name,
                'description': description,
                'category': category,
                'examples': examples,
                'exceptions': exceptions,
                'difficulty_level': self._assess_difficulty(description, examples),
                'processed_at': time.time()
            }
            
        except Exception as e:
            logger.warning(f"Grammar rule cleaning error: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        text = text.replace(' , ', ', ')
        text = text.replace(' . ', '. ')
        text = text.replace(' ; ', '; ')
        text = text.replace(' : ', ': ')
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def _extract_phonetic(self, phonetic_text: str) -> str:
        """Extract clean phonetic notation"""
        if not phonetic_text:
            return ""
        
        # Extract content within brackets or slashes
        match = self.text_patterns['phonetic_symbols'].search(phonetic_text)
        if match:
            return match.group(1).strip()
        
        return phonetic_text.strip()
    
    def _clean_pos(self, pos_text: str) -> str:
        """Clean part of speech tag"""
        if not pos_text:
            return "unknown"
        
        # Extract from parentheses if present
        match = self.text_patterns['pos_tags'].search(pos_text)
        if match:
            pos_text = match.group(1)
        
        # Normalize common POS tags
        pos_lower = pos_text.lower().strip()
        pos_mapping = {
            'n.': 'noun',
            'noun': 'noun',
            'v.': 'verb',
            'verb': 'verb',
            'adj.': 'adjective',
            'adjective': 'adjective',
            'adv.': 'adverb',
            'adverb': 'adverb',
            'prep.': 'preposition',
            'preposition': 'preposition',
            'conj.': 'conjunction',
            'conjunction': 'conjunction',
            'pron.': 'pronoun',
            'pronoun': 'pronoun'
        }
        
        return pos_mapping.get(pos_lower, pos_lower)
    
    def _clean_examples(self, examples: List[str]) -> List[str]:
        """Clean example sentences"""
        cleaned = []
        for example in examples:
            if isinstance(example, str):
                example = example.strip()
                
                # Remove example markers
                match = self.text_patterns['example_markers'].search(example)
                if match:
                    example = match.group(1)
                
                example = self._clean_text(example)
                if len(example) > 5:  # Only keep substantial examples
                    cleaned.append(example)
        
        return cleaned[:10]  # Limit to 10 examples
    
    def _clean_word_list(self, words: List[str]) -> List[str]:
        """Clean list of words (synonyms, antonyms)"""
        cleaned = []
        for word in words:
            if isinstance(word, str):
                word = word.strip().lower()
                if word and len(word) > 1 and word.isalpha():
                    cleaned.append(word)
        
        return list(set(cleaned))[:20]  # Remove duplicates and limit to 20
    
    def _assess_difficulty(self, description: str, examples: List[str]) -> str:
        """Assess difficulty level of grammar rule"""
        complexity_indicators = {
            'beginner': ['simple', 'basic', 'easy', 'present tense', 'singular', 'plural'],
            'intermediate': ['complex', 'advanced', 'conditional', 'subjunctive', 'passive'],
            'advanced': ['sophisticated', 'nuanced', 'exception', 'irregular', 'idiomatic']
        }
        
        text_to_analyze = (description + ' ' + ' '.join(examples)).lower()
        
        scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_to_analyze)
            scores[level] = score
        
        return max(scores, key=scores.get) if scores else 'intermediate'
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into manageable chunks"""
        if not text:
            return []
        
        # Split by sentences first
        sentences = self.text_patterns['sentence_boundary'].split(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _map_data_type_to_schema(self, data_type: str) -> Optional[str]:
        """Map data type to schema name"""
        mapping = {
            'word': 'dictionary_entry',
            'vocabulary': 'dictionary_entry',
            'dictionary': 'dictionary_entry',
            'rule': 'grammar_rule',
            'grammar': 'grammar_rule',
            'text': 'text_corpus',
            'corpus': 'text_corpus'
        }
        return mapping.get(data_type.lower())
    
    async def _validate_entry(self, entry: Dict[str, Any], schema_name: str) -> bool:
        """Validate entry against schema"""
        try:
            if schema_name not in self.schemas:
                return False
            
            schema = self.schemas[schema_name]
            
            # Check required fields with flexibility for definitions/definition
            for field in schema.required_fields:
                field_exists = field in entry and entry[field]
                
                # Special handling for definition/definitions
                if field == 'definition' and not field_exists:
                    # Check if 'definitions' exists instead
                    definitions = entry.get('definitions', [])
                    if definitions and len(definitions) > 0:
                        # Convert definitions list to single definition
                        entry['definition'] = definitions[0] if isinstance(definitions, list) else str(definitions)
                        field_exists = True
                
                if not field_exists:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check field types
            for field, expected_type in schema.field_types.items():
                if field in entry and entry[field] is not None:
                    if not isinstance(entry[field], expected_type):
                        logger.warning(f"Invalid type for field {field}: expected {expected_type}, got {type(entry[field])}")
                        return False
            
            # Check validation rules
            for field, rules in schema.validation_rules.items():
                if field in entry and entry[field]:
                    value = entry[field]
                    
                    if isinstance(value, str):
                        if 'min_length' in rules and len(value) < rules['min_length']:
                            return False
                        if 'max_length' in rules and len(value) > rules['max_length']:
                            return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation error: {str(e)}")
            return False
    
    def _generate_content_hash(self, content: Dict[str, Any]) -> str:
        """Generate hash for content deduplication"""
        # Create a string representation focusing on key content
        if 'word' in content and 'definition' in content:
            key_content = f"{content['word']}:{content['definition']}"
        elif 'rule_name' in content and 'description' in content:
            key_content = f"{content['rule_name']}:{content['description']}"
        elif 'text' in content:
            key_content = content['text']
        else:
            key_content = json.dumps(content, sort_keys=True)
        
        return hashlib.md5(key_content.encode()).hexdigest()
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'cache_size': len(self.content_hashes),
            'supported_schemas': list(self.schemas.keys()),
            'schema_versions': {name: schema.version for name, schema in self.schemas.items()}
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'duplicates_removed': 0,
            'validation_errors': 0
        }
        
    def clear_cache(self):
        """Clear the deduplication cache"""
        self.content_hashes.clear()