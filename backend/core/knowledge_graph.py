"""
Knowledge Graph for storing linguistic relationships and semantic connections
Uses memory-efficient graph structures for fast traversal
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import hashlib
import time
from enum import Enum

logger = logging.getLogger(__name__)

class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    HYPONYM = "hyponym"  # is-a relationship
    HYPERNYM = "hypernym"  # parent concept
    MERONYM = "meronym"  # part-of relationship
    HOLONYM = "holonym"  # whole-of relationship
    SIMILAR = "similar"
    RELATED = "related"
    GRAMMAR_RULE = "grammar_rule"
    EXAMPLE = "example"
    DEFINITION = "definition"

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: str  # 'word', 'concept', 'rule', 'example'
    language: str
    content: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'language': self.language,
            'content': self.content,
            'properties': self.properties,
            'created_at': self.created_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }

@dataclass
class GraphEdge:
    """Represents an edge (relationship) in the knowledge graph"""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_id': self.edge_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'weight': self.weight,
            'confidence': self.confidence,
            'properties': self.properties,
            'created_at': self.created_at
        }

class KnowledgeGraph:
    """
    Memory-efficient knowledge graph for linguistic relationships
    """
    
    def __init__(self):
        # Core graph structures
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        
        # Efficient lookup structures
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)  # node_id -> set of connected node_ids
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # incoming connections
        self.relation_index: Dict[RelationType, List[str]] = defaultdict(list)  # relation -> edge_ids
        self.language_index: Dict[str, Set[str]] = defaultdict(set)  # language -> node_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # node_type -> node_ids
        
        # Caching for frequent queries
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_max_size = 1000
        
        # Statistics
        self.stats = {
            'nodes_count': 0,
            'edges_count': 0,
            'languages': set(),
            'node_types': set(),
            'relation_types': set()
        }
    
    async def initialize(self):
        """Initialize the knowledge graph"""
        try:
            logger.info("Initializing Knowledge Graph...")
            
            # Load existing graph if available
            await self._load_existing_graph()
            
            logger.info(f"Knowledge Graph initialized with {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {str(e)}")
            raise
    
    async def add_entity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new entity (word, concept, rule) to the knowledge graph
        """
        try:
            entity_type = data.get('data_type', 'unknown')
            language = data.get('language', 'english')
            content = data.get('content', {})
            
            if entity_type == 'vocabulary':
                return await self._add_vocabulary_entity(content, language)
            elif entity_type == 'grammar':
                return await self._add_grammar_entity(content, language)
            elif entity_type == 'concept':
                return await self._add_concept_entity(content, language)
            else:
                return {'success': False, 'error': f'Unknown entity type: {entity_type}'}
                
        except Exception as e:
            logger.error(f"Error adding entity to graph: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _add_vocabulary_entity(self, content: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Add vocabulary word to the graph"""
        try:
            word = content.get('word', '').lower()
            if not word:
                return {'success': False, 'error': 'Word is required'}
            
            # Create node ID
            node_id = self._generate_node_id(word, language, 'word')
            
            # Create node
            node = GraphNode(
                node_id=node_id,
                node_type='word',
                language=language,
                content=word,
                properties={
                    'definitions': content.get('definitions', []),
                    'part_of_speech': content.get('part_of_speech', ''),
                    'phonetic': content.get('phonetic', ''),
                    'examples': content.get('examples', []),
                    'frequency': content.get('frequency', 0.0)
                }
            )
            
            # Add to graph
            self.nodes[node_id] = node
            self.language_index[language].add(node_id)
            self.type_index['word'].add(node_id)
            
            # Create relationships
            connections = 0
            
            # Synonym relationships
            for synonym in content.get('synonyms', []):
                synonym_id = self._generate_node_id(synonym.lower(), language, 'word')
                if synonym_id != node_id:
                    edge = await self._create_edge(node_id, synonym_id, RelationType.SYNONYM, 0.8)
                    connections += 1
            
            # Antonym relationships
            for antonym in content.get('antonyms', []):
                antonym_id = self._generate_node_id(antonym.lower(), language, 'word')
                if antonym_id != node_id:
                    edge = await self._create_edge(node_id, antonym_id, RelationType.ANTONYM, 0.8)
                    connections += 1
            
            # Create definition nodes and relationships
            for i, definition in enumerate(content.get('definitions', [])):
                if definition.strip():
                    def_id = self._generate_node_id(f"{word}_def_{i}", language, 'definition')
                    def_node = GraphNode(
                        node_id=def_id,
                        node_type='definition',
                        language=language,
                        content=definition,
                        properties={'word': word, 'definition_index': i}
                    )
                    self.nodes[def_id] = def_node
                    self.type_index['definition'].add(def_id)
                    
                    edge = await self._create_edge(node_id, def_id, RelationType.DEFINITION, 1.0)
                    connections += 1
            
            # Update statistics
            await self._update_stats()
            
            return {
                'success': True,
                'node_id': node_id,
                'connections': connections
            }
            
        except Exception as e:
            logger.error(f"Error adding vocabulary entity: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _add_grammar_entity(self, content: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Add grammar rule to the graph"""
        try:
            rule_name = content.get('rule_name', '')
            description = content.get('description', '')
            
            if not rule_name or not description:
                return {'success': False, 'error': 'Rule name and description are required'}
            
            # Create node ID
            node_id = self._generate_node_id(rule_name, language, 'grammar_rule')
            
            # Create node
            node = GraphNode(
                node_id=node_id,
                node_type='grammar_rule',
                language=language,
                content=rule_name,
                properties={
                    'description': description,
                    'category': content.get('category', 'general'),
                    'examples': content.get('examples', []),
                    'exceptions': content.get('exceptions', []),
                    'pattern': content.get('pattern', '')
                }
            )
            
            # Add to graph
            self.nodes[node_id] = node
            self.language_index[language].add(node_id)
            self.type_index['grammar_rule'].add(node_id)
            
            # Create example nodes and relationships
            connections = 0
            for i, example in enumerate(content.get('examples', [])):
                if example.strip():
                    ex_id = self._generate_node_id(f"{rule_name}_ex_{i}", language, 'example')
                    ex_node = GraphNode(
                        node_id=ex_id,
                        node_type='example',
                        language=language,
                        content=example,
                        properties={'rule_name': rule_name, 'example_index': i}
                    )
                    self.nodes[ex_id] = ex_node
                    self.type_index['example'].add(ex_id)
                    
                    edge = await self._create_edge(node_id, ex_id, RelationType.EXAMPLE, 0.9)
                    connections += 1
            
            # Update statistics
            await self._update_stats()
            
            return {
                'success': True,
                'node_id': node_id,
                'connections': connections
            }
            
        except Exception as e:
            logger.error(f"Error adding grammar entity: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _add_concept_entity(self, content: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Add concept to the graph"""
        # Implementation for concept entities
        pass
    
    async def _create_edge(self, source_id: str, target_id: str, relation_type: RelationType, confidence: float = 0.5) -> Optional[GraphEdge]:
        """Create an edge between two nodes"""
        try:
            # Generate edge ID
            edge_id = hashlib.md5(f"{source_id}_{target_id}_{relation_type.value}".encode()).hexdigest()[:16]
            
            # Check if edge already exists
            if edge_id in self.edges:
                return self.edges[edge_id]
            
            # Create target node if it doesn't exist (placeholder)
            if target_id not in self.nodes:
                # Extract word from target_id for placeholder
                parts = target_id.split('_')
                word = parts[0] if parts else target_id
                lang = parts[1] if len(parts) > 1 else 'english'
                
                placeholder_node = GraphNode(
                    node_id=target_id,
                    node_type='word',
                    language=lang,
                    content=word,
                    properties={'placeholder': True}
                )
                self.nodes[target_id] = placeholder_node
                self.language_index[lang].add(target_id)
                self.type_index['word'].add(target_id)
            
            # Create edge
            edge = GraphEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=confidence
            )
            
            # Add to graph structures
            self.edges[edge_id] = edge
            self.adjacency_list[source_id].add(target_id)
            self.reverse_adjacency[target_id].add(source_id)
            self.relation_index[relation_type].append(edge_id)
            
            return edge
            
        except Exception as e:
            logger.error(f"Error creating edge: {str(e)}")
            return None
    
    async def get_context(self, query: str, language: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get contextual information for a query using graph traversal
        """
        try:
            # Check cache first
            cache_key = f"{query}_{language}_{max_depth}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            query_lower = query.lower().strip()
            node_id = self._generate_node_id(query_lower, language, 'word')
            
            if node_id not in self.nodes:
                return {'related_words': [], 'definitions': [], 'examples': []}
            
            # Update access statistics
            self.nodes[node_id].access_count += 1
            self.nodes[node_id].last_accessed = time.time()
            
            # BFS traversal to find context
            context = await self._bfs_context_search(node_id, max_depth)
            
            # Cache the result
            if len(self.query_cache) < self.cache_max_size:
                self.query_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return {'related_words': [], 'definitions': [], 'examples': []}
    
    async def _bfs_context_search(self, start_node: str, max_depth: int) -> Dict[str, Any]:
        """
        Breadth-first search to find contextual information
        """
        visited = set()
        queue = deque([(start_node, 0)])
        
        related_words = []
        definitions = []
        examples = []
        grammar_rules = []
        
        while queue and len(visited) < 100:  # Limit search to prevent memory issues
            node_id, depth = queue.popleft()
            
            if node_id in visited or depth > max_depth:
                continue
            
            visited.add(node_id)
            node = self.nodes.get(node_id)
            
            if not node:
                continue
            
            # Collect information based on node type
            if node.node_type == 'word' and node_id != start_node:
                related_words.append(node.content)
            elif node.node_type == 'definition':
                definitions.append(node.content)
            elif node.node_type == 'example':
                examples.append(node.content)
            elif node.node_type == 'grammar_rule':
                grammar_rules.append(node.content)
            
            # Add neighbors to queue
            for neighbor_id in self.adjacency_list[node_id]:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))
        
        return {
            'related_words': related_words[:10],  # Limit results
            'definitions': definitions[:5],
            'examples': examples[:5],
            'grammar_rules': grammar_rules[:3]
        }
    
    async def update_from_feedback(self, query: str, correction: str, language: str) -> Dict[str, Any]:
        """
        Update knowledge graph based on user feedback
        """
        try:
            query_lower = query.lower().strip()
            node_id = self._generate_node_id(query_lower, language, 'word')
            
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Add correction as new definition if it's not already there
                current_definitions = node.properties.get('definitions', [])
                if correction not in current_definitions:
                    current_definitions.insert(0, correction)  # Add as primary definition
                    node.properties['definitions'] = current_definitions
                    
                    # Create new definition node
                    def_id = self._generate_node_id(f"{query_lower}_feedback_{time.time()}", language, 'definition')
                    def_node = GraphNode(
                        node_id=def_id,
                        node_type='definition',
                        language=language,
                        content=correction,
                        properties={'word': query_lower, 'from_feedback': True}
                    )
                    
                    self.nodes[def_id] = def_node
                    self.type_index['definition'].add(def_id)
                    
                    # Create relationship
                    await self._create_edge(node_id, def_id, RelationType.DEFINITION, 0.9)
                    
                    return {'success': True, 'updated': True}
            
            return {'success': True, 'updated': False}
            
        except Exception as e:
            logger.error(f"Error updating from feedback: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        await self._update_stats()
        
        # Calculate memory efficiency metrics
        avg_connections = len(self.edges) / len(self.nodes) if self.nodes else 0
        
        # Language distribution
        lang_distribution = {}
        for lang, nodes in self.language_index.items():
            lang_distribution[lang] = len(nodes)
        
        # Node type distribution
        type_distribution = {}
        for node_type, nodes in self.type_index.items():
            type_distribution[node_type] = len(nodes)
        
        return {
            'nodes_count': len(self.nodes),
            'edges_count': len(self.edges),
            'languages': list(self.stats['languages']),
            'node_types': list(self.stats['node_types']),
            'relation_types': list(self.stats['relation_types']),
            'average_connections': round(avg_connections, 2),
            'language_distribution': lang_distribution,
            'type_distribution': type_distribution,
            'cache_size': len(self.query_cache),
            'most_accessed_nodes': await self._get_most_accessed_nodes()
        }
    
    async def _get_most_accessed_nodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most frequently accessed nodes"""
        sorted_nodes = sorted(
            self.nodes.values(), 
            key=lambda n: n.access_count, 
            reverse=True
        )
        
        return [
            {
                'content': node.content,
                'type': node.node_type,
                'language': node.language,
                'access_count': node.access_count
            }
            for node in sorted_nodes[:limit]
        ]
    
    def _generate_node_id(self, content: str, language: str, node_type: str) -> str:
        """Generate unique node ID"""
        content_clean = content.lower().strip()
        id_string = f"{content_clean}_{language}_{node_type}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    async def _update_stats(self):
        """Update graph statistics"""
        self.stats['nodes_count'] = len(self.nodes)
        self.stats['edges_count'] = len(self.edges)
        
        # Update language and type sets
        for node in self.nodes.values():
            self.stats['languages'].add(node.language)
            self.stats['node_types'].add(node.node_type)
        
        for edge in self.edges.values():
            self.stats['relation_types'].add(edge.relation_type.value)
    
    async def _load_existing_graph(self):
        """Load existing graph from persistent storage"""
        # This would load from database or files
        pass
    
    async def save_graph(self):
        """Save graph to persistent storage"""
        # This would save to database or files
        pass
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
    
    async def optimize_memory(self):
        """Optimize memory usage by removing least used nodes/edges"""
        try:
            # Remove nodes with very low access count and old timestamp
            current_time = time.time()
            nodes_to_remove = []
            
            for node_id, node in self.nodes.items():
                # Remove if not accessed in 30 days and access count < 2
                if (current_time - node.last_accessed) > (30 * 24 * 3600) and node.access_count < 2:
                    if not node.properties.get('placeholder', False):  # Keep placeholders for now
                        nodes_to_remove.append(node_id)
            
            # Remove identified nodes and their edges
            for node_id in nodes_to_remove[:100]:  # Limit to prevent massive deletions
                await self._remove_node(node_id)
            
            # Clear cache if it's getting too large
            if len(self.query_cache) > self.cache_max_size:
                self.clear_cache()
            
            logger.info(f"Memory optimization removed {len(nodes_to_remove[:100])} nodes")
            
        except Exception as e:
            logger.error(f"Memory optimization error: {str(e)}")
    
    async def _remove_node(self, node_id: str):
        """Remove a node and all its edges"""
        if node_id not in self.nodes:
            return
        
        # Remove all edges connected to this node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            edge = self.edges[edge_id]
            # Clean up adjacency lists
            self.adjacency_list[edge.source_id].discard(edge.target_id)
            self.reverse_adjacency[edge.target_id].discard(edge.source_id)
            # Remove from relation index
            if edge_id in self.relation_index[edge.relation_type]:
                self.relation_index[edge.relation_type].remove(edge_id)
            # Remove edge
            del self.edges[edge_id]
        
        # Remove node from indexes
        node = self.nodes[node_id]
        self.language_index[node.language].discard(node_id)
        self.type_index[node.node_type].discard(node_id)
        
        # Remove node
        del self.nodes[node_id]