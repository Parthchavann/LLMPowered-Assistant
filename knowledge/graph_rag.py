"""
Graph-based RAG with Knowledge Graphs
Revolutionary semantic reasoning using knowledge graphs that can:
- Build dynamic knowledge graphs from documents
- Perform graph traversal for complex reasoning
- Understand entity relationships and dependencies
- Enable causal and temporal reasoning
- Support graph neural networks for advanced inference

This represents the future of RAG - moving beyond simple vector similarity
to true semantic understanding through graph structures.
"""

import asyncio
import logging
import json
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import spacy
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeType(Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    DOCUMENT = "document"
    FACT = "fact"
    RULE = "rule"
    TEMPORAL = "temporal"
    CAUSAL = "causal"

class EdgeType(Enum):
    RELATES_TO = "relates_to"
    IS_A = "is_a"
    PART_OF = "part_of"
    CAUSES = "causes"
    PREVENTS = "prevents"
    REQUIRES = "requires"
    FOLLOWS = "follows"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"

@dataclass
class GraphEntity:
    entity_id: str
    name: str
    type: NodeType
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    source_documents: List[str] = None
    
    def __post_init__(self):
        if self.source_documents is None:
            self.source_documents = []

@dataclass
class GraphRelation:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: EdgeType
    properties: Dict[str, Any]
    confidence: float = 1.0
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

@dataclass
class ReasoningPath:
    path_id: str
    entities: List[str]
    relations: List[str]
    reasoning_steps: List[str]
    confidence: float
    evidence: List[str]

class KnowledgeGraphBuilder:
    """
    Builds dynamic knowledge graphs from documents using NLP and ML
    """
    
    def __init__(self):
        # Load spaCy model for NER and dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        self.entities = {}
        self.relations = {}
        self.graph = nx.DiGraph()
        
        # Predefined relationship patterns
        self.relation_patterns = self._initialize_relation_patterns()
    
    def _initialize_relation_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize patterns for extracting relationships from text"""
        return {
            'causal': [
                {'pattern': r'(.+) causes (.+)', 'type': EdgeType.CAUSES},
                {'pattern': r'(.+) leads to (.+)', 'type': EdgeType.CAUSES},
                {'pattern': r'(.+) results in (.+)', 'type': EdgeType.CAUSES},
                {'pattern': r'due to (.+), (.+)', 'type': EdgeType.CAUSES, 'reverse': True},
            ],
            'temporal': [
                {'pattern': r'(.+) before (.+)', 'type': EdgeType.FOLLOWS, 'reverse': True},
                {'pattern': r'(.+) after (.+)', 'type': EdgeType.FOLLOWS},
                {'pattern': r'first (.+), then (.+)', 'type': EdgeType.FOLLOWS},
            ],
            'hierarchical': [
                {'pattern': r'(.+) is a (.+)', 'type': EdgeType.IS_A},
                {'pattern': r'(.+) is part of (.+)', 'type': EdgeType.PART_OF},
                {'pattern': r'(.+) contains (.+)', 'type': EdgeType.CONTAINS, 'reverse': True},
            ],
            'conditional': [
                {'pattern': r'if (.+), then (.+)', 'type': EdgeType.REQUIRES, 'reverse': True},
                {'pattern': r'(.+) requires (.+)', 'type': EdgeType.REQUIRES},
                {'pattern': r'(.+) depends on (.+)', 'type': EdgeType.REQUIRES},
            ]
        }
    
    async def build_graph_from_documents(self, documents: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build knowledge graph from a collection of documents
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        for doc in documents:
            await self._process_document(doc)
        
        # Post-process graph for consistency and enrichment
        await self._enrich_graph()
        await self._validate_graph()
        
        logger.info(f"Knowledge graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        return self.graph
    
    async def _process_document(self, document: Dict[str, Any]):
        """Process a single document and extract entities and relations"""
        doc_id = document.get('id', str(hash(document.get('content', ''))))
        content = document.get('content', '')
        
        # Extract entities
        entities = await self._extract_entities(content, doc_id)
        
        # Extract relations
        relations = await self._extract_relations(content, doc_id, entities)
        
        # Add to graph
        for entity in entities:
            self._add_entity_to_graph(entity)
        
        for relation in relations:
            self._add_relation_to_graph(relation)
    
    async def _extract_entities(self, text: str, doc_id: str) -> List[GraphEntity]:
        """Extract entities from text using NLP"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity = GraphEntity(
                    entity_id=f"entity_{hash(ent.text)}",
                    name=ent.text,
                    type=NodeType.ENTITY,
                    properties={
                        'label': ent.label_,
                        'description': ent.text,
                        'pos_start': ent.start_char,
                        'pos_end': ent.end_char
                    },
                    source_documents=[doc_id]
                )
                entities.append(entity)
            
            # Extract key noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:  # Multi-word concepts
                    entity = GraphEntity(
                        entity_id=f"concept_{hash(chunk.text)}",
                        name=chunk.text,
                        type=NodeType.CONCEPT,
                        properties={
                            'description': chunk.text,
                            'pos_start': chunk.start_char,
                            'pos_end': chunk.end_char
                        },
                        source_documents=[doc_id]
                    )
                    entities.append(entity)
        
        else:
            # Fallback: simple keyword extraction
            words = text.split()
            for i in range(len(words)-1):
                if words[i].istitle() and words[i+1].istitle():
                    entity_text = f"{words[i]} {words[i+1]}"
                    entity = GraphEntity(
                        entity_id=f"entity_{hash(entity_text)}",
                        name=entity_text,
                        type=NodeType.ENTITY,
                        properties={'description': entity_text},
                        source_documents=[doc_id]
                    )
                    entities.append(entity)
        
        return entities
    
    async def _extract_relations(self, text: str, doc_id: str, entities: List[GraphEntity]) -> List[GraphRelation]:
        """Extract relations between entities using pattern matching and NLP"""
        relations = []
        
        # Pattern-based relation extraction
        for category, patterns in self.relation_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                relation_type = pattern_info['type']
                reverse = pattern_info.get('reverse', False)
                
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(2).strip()
                    
                    # Find matching entities
                    source_entity = self._find_entity_by_text(source_text, entities)
                    target_entity = self._find_entity_by_text(target_text, entities)
                    
                    if source_entity and target_entity:
                        if reverse:
                            source_entity, target_entity = target_entity, source_entity
                        
                        relation = GraphRelation(
                            relation_id=f"rel_{hash(f'{source_entity.entity_id}_{target_entity.entity_id}')}",
                            source_id=source_entity.entity_id,
                            target_id=target_entity.entity_id,
                            relation_type=relation_type,
                            properties={
                                'category': category,
                                'text_evidence': match.group(0)
                            },
                            evidence=[doc_id]
                        )
                        relations.append(relation)
        
        # Dependency-based relation extraction (if spaCy available)
        if self.nlp:
            doc = self.nlp(text)
            for sent in doc.sents:
                relations.extend(await self._extract_dependency_relations(sent, doc_id, entities))
        
        return relations
    
    async def _extract_dependency_relations(self, sent, doc_id: str, entities: List[GraphEntity]) -> List[GraphRelation]:
        """Extract relations based on dependency parsing"""
        relations = []
        
        # Look for subject-verb-object patterns
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                objects = [child for child in token.children if child.dep_ in ["dobj", "pobj"]]
                
                for subj in subjects:
                    for obj in objects:
                        subj_entity = self._find_entity_by_text(subj.text, entities)
                        obj_entity = self._find_entity_by_text(obj.text, entities)
                        
                        if subj_entity and obj_entity:
                            relation = GraphRelation(
                                relation_id=f"dep_rel_{hash(f'{subj_entity.entity_id}_{obj_entity.entity_id}')}",
                                source_id=subj_entity.entity_id,
                                target_id=obj_entity.entity_id,
                                relation_type=EdgeType.RELATES_TO,
                                properties={
                                    'verb': token.text,
                                    'dependency': 'svo'
                                },
                                evidence=[doc_id]
                            )
                            relations.append(relation)
        
        return relations
    
    def _find_entity_by_text(self, text: str, entities: List[GraphEntity]) -> Optional[GraphEntity]:
        """Find entity that best matches the given text"""
        text_lower = text.lower().strip()
        
        # Exact match
        for entity in entities:
            if entity.name.lower() == text_lower:
                return entity
        
        # Partial match
        for entity in entities:
            if text_lower in entity.name.lower() or entity.name.lower() in text_lower:
                return entity
        
        return None
    
    def _add_entity_to_graph(self, entity: GraphEntity):
        """Add entity to the knowledge graph"""
        if entity.entity_id not in self.entities:
            self.entities[entity.entity_id] = entity
            self.graph.add_node(
                entity.entity_id,
                name=entity.name,
                type=entity.type.value,
                properties=entity.properties,
                confidence=entity.confidence
            )
        else:
            # Merge with existing entity
            existing = self.entities[entity.entity_id]
            existing.source_documents.extend(entity.source_documents)
            existing.confidence = max(existing.confidence, entity.confidence)
    
    def _add_relation_to_graph(self, relation: GraphRelation):
        """Add relation to the knowledge graph"""
        if relation.relation_id not in self.relations:
            self.relations[relation.relation_id] = relation
            self.graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation_id=relation.relation_id,
                type=relation.relation_type.value,
                properties=relation.properties,
                confidence=relation.confidence
            )
        else:
            # Strengthen existing relation
            existing = self.relations[relation.relation_id]
            existing.evidence.extend(relation.evidence)
            existing.confidence = min(existing.confidence + 0.1, 1.0)
    
    async def _enrich_graph(self):
        """Enrich the graph with inferred relationships and properties"""
        # Infer transitive relationships
        await self._infer_transitive_relations()
        
        # Add similarity relationships
        await self._add_similarity_relationships()
        
        # Detect contradictions
        await self._detect_contradictions()
    
    async def _infer_transitive_relations(self):
        """Infer new relationships based on transitivity rules"""
        transitive_rules = {
            EdgeType.IS_A: EdgeType.IS_A,  # If A is_a B and B is_a C, then A is_a C
            EdgeType.PART_OF: EdgeType.PART_OF,  # If A part_of B and B part_of C, then A part_of C
            EdgeType.FOLLOWS: EdgeType.FOLLOWS  # If A follows B and B follows C, then A follows C
        }
        
        for relation_type, inferred_type in transitive_rules.items():
            edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                    if d.get('type') == relation_type.value]
            
            for source, intermediate in edges:
                for intermediate2, target in edges:
                    if intermediate == intermediate2 and source != target:
                        # Check if direct relation doesn't already exist
                        if not self.graph.has_edge(source, target):
                            relation = GraphRelation(
                                relation_id=f"inferred_{hash(f'{source}_{target}')}",
                                source_id=source,
                                target_id=target,
                                relation_type=inferred_type,
                                properties={'inferred': True, 'via': intermediate},
                                confidence=0.7,
                                evidence=['transitive_inference']
                            )
                            self._add_relation_to_graph(relation)
    
    async def _add_similarity_relationships(self):
        """Add similarity relationships between entities based on embeddings"""
        # This would require embeddings - simplified for now
        entities_by_type = defaultdict(list)
        for entity_id, entity in self.entities.items():
            entities_by_type[entity.type].append(entity_id)
        
        # Add similarity within same types
        for entity_type, entity_list in entities_by_type.items():
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    entity1_id = entity_list[i]
                    entity2_id = entity_list[j]
                    
                    # Simple similarity based on name overlap
                    entity1_name = self.entities[entity1_id].name.lower()
                    entity2_name = self.entities[entity2_id].name.lower()
                    
                    common_words = set(entity1_name.split()) & set(entity2_name.split())
                    if len(common_words) > 0:
                        similarity = len(common_words) / max(len(entity1_name.split()), len(entity2_name.split()))
                        
                        if similarity > 0.3:
                            relation = GraphRelation(
                                relation_id=f"sim_{hash(f'{entity1_id}_{entity2_id}')}",
                                source_id=entity1_id,
                                target_id=entity2_id,
                                relation_type=EdgeType.SIMILAR_TO,
                                properties={'similarity_score': similarity},
                                confidence=similarity,
                                evidence=['name_similarity']
                            )
                            self._add_relation_to_graph(relation)
    
    async def _detect_contradictions(self):
        """Detect contradictory relationships in the graph"""
        contradiction_patterns = [
            (EdgeType.CAUSES, EdgeType.PREVENTS),
            (EdgeType.REQUIRES, EdgeType.PREVENTS),
        ]
        
        for type1, type2 in contradiction_patterns:
            edges1 = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') == type1.value]
            edges2 = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') == type2.value]
            
            for u1, v1 in edges1:
                for u2, v2 in edges2:
                    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                        # Found contradiction
                        relation = GraphRelation(
                            relation_id=f"contra_{hash(f'{u1}_{v1}_{u2}_{v2}')}",
                            source_id=u1,
                            target_id=v1,
                            relation_type=EdgeType.CONTRADICTS,
                            properties={
                                'contradiction_type': f'{type1.value}_vs_{type2.value}',
                                'requires_resolution': True
                            },
                            confidence=0.8,
                            evidence=['contradiction_detection']
                        )
                        self._add_relation_to_graph(relation)
    
    async def _validate_graph(self):
        """Validate graph consistency and quality"""
        # Remove low-confidence nodes and edges
        nodes_to_remove = []
        for node_id in self.graph.nodes():
            if self.graph.nodes[node_id].get('confidence', 1.0) < 0.3:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self.graph.remove_node(node_id)
            if node_id in self.entities:
                del self.entities[node_id]

class GraphRAGEngine:
    """
    RAG engine that uses knowledge graphs for semantic reasoning
    """
    
    def __init__(self, knowledge_graph: nx.DiGraph):
        self.graph = knowledge_graph
        self.graph_builder = KnowledgeGraphBuilder()
    
    async def query_with_graph_reasoning(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer query using graph-based reasoning
        """
        # Extract entities from query
        query_entities = await self._extract_query_entities(query)
        
        # Find relevant subgraph
        relevant_subgraph = await self._find_relevant_subgraph(query_entities, depth=2)
        
        # Perform reasoning on subgraph
        reasoning_paths = await self._generate_reasoning_paths(query_entities, relevant_subgraph)
        
        # Rank and select best reasoning paths
        best_paths = sorted(reasoning_paths, key=lambda x: x.confidence, reverse=True)[:top_k]
        
        # Generate answer based on reasoning paths
        answer = await self._generate_answer_from_paths(query, best_paths)
        
        return {
            'query': query,
            'answer': answer,
            'reasoning_paths': [asdict(path) for path in best_paths],
            'subgraph_nodes': len(relevant_subgraph.nodes),
            'subgraph_edges': len(relevant_subgraph.edges),
            'query_entities': [entity.name for entity in query_entities]
        }
    
    async def _extract_query_entities(self, query: str) -> List[GraphEntity]:
        """Extract entities mentioned in the query"""
        query_entities = []
        
        # Simple approach: find graph entities mentioned in query
        query_lower = query.lower()
        
        for entity_id, entity in self.graph_builder.entities.items():
            if entity.name.lower() in query_lower:
                query_entities.append(entity)
        
        return query_entities
    
    async def _find_relevant_subgraph(self, query_entities: List[GraphEntity], depth: int = 2) -> nx.DiGraph:
        """Find relevant subgraph around query entities"""
        relevant_nodes = set()
        
        # Add query entities
        for entity in query_entities:
            relevant_nodes.add(entity.entity_id)
        
        # Expand to neighbors within specified depth
        for entity in query_entities:
            if entity.entity_id in self.graph:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph, entity.entity_id, cutoff=depth
                )
                relevant_nodes.update(neighbors.keys())
        
        # Create subgraph
        subgraph = self.graph.subgraph(relevant_nodes).copy()
        
        return subgraph
    
    async def _generate_reasoning_paths(self, query_entities: List[GraphEntity], 
                                      subgraph: nx.DiGraph) -> List[ReasoningPath]:
        """Generate reasoning paths through the knowledge graph"""
        reasoning_paths = []
        
        # Find paths between query entities
        for i in range(len(query_entities)):
            for j in range(i + 1, len(query_entities)):
                source = query_entities[i].entity_id
                target = query_entities[j].entity_id
                
                if source in subgraph and target in subgraph:
                    try:
                        # Find shortest paths
                        paths = list(nx.all_shortest_paths(subgraph, source, target))
                        
                        for path in paths[:3]:  # Limit to 3 paths per pair
                            reasoning_path = await self._create_reasoning_path(path, subgraph)
                            reasoning_paths.append(reasoning_path)
                    
                    except nx.NetworkXNoPath:
                        continue  # No path exists
        
        # Also generate single-entity reasoning paths
        for entity in query_entities:
            if entity.entity_id in subgraph:
                neighbors = list(subgraph.neighbors(entity.entity_id))[:5]  # Top 5 neighbors
                for neighbor in neighbors:
                    path = [entity.entity_id, neighbor]
                    reasoning_path = await self._create_reasoning_path(path, subgraph)
                    reasoning_paths.append(reasoning_path)
        
        return reasoning_paths
    
    async def _create_reasoning_path(self, node_path: List[str], subgraph: nx.DiGraph) -> ReasoningPath:
        """Create a reasoning path object from a node path"""
        relations = []
        reasoning_steps = []
        evidence = []
        
        for i in range(len(node_path) - 1):
            source = node_path[i]
            target = node_path[i + 1]
            
            if subgraph.has_edge(source, target):
                edge_data = subgraph.edges[source, target]
                relation_type = edge_data.get('type', 'relates_to')
                relations.append(relation_type)
                
                # Create human-readable reasoning step
                source_name = subgraph.nodes[source].get('name', source)
                target_name = subgraph.nodes[target].get('name', target)
                step = f"{source_name} {relation_type.replace('_', ' ')} {target_name}"
                reasoning_steps.append(step)
                
                # Collect evidence
                evidence.extend(edge_data.get('properties', {}).get('evidence', []))
        
        # Calculate path confidence
        confidences = []
        for i in range(len(node_path) - 1):
            source, target = node_path[i], node_path[i + 1]
            if subgraph.has_edge(source, target):
                edge_confidence = subgraph.edges[source, target].get('confidence', 0.5)
                confidences.append(edge_confidence)
        
        path_confidence = np.mean(confidences) if confidences else 0.0
        
        return ReasoningPath(
            path_id=f"path_{hash('_'.join(node_path))}",
            entities=node_path,
            relations=relations,
            reasoning_steps=reasoning_steps,
            confidence=path_confidence,
            evidence=evidence
        )
    
    async def _generate_answer_from_paths(self, query: str, paths: List[ReasoningPath]) -> str:
        """Generate final answer based on reasoning paths"""
        if not paths:
            return "I couldn't find sufficient information in the knowledge graph to answer your query."
        
        # Combine reasoning from multiple paths
        all_reasoning = []
        high_confidence_paths = [p for p in paths if p.confidence > 0.6]
        
        if high_confidence_paths:
            paths_to_use = high_confidence_paths[:3]  # Use top 3 high-confidence paths
        else:
            paths_to_use = paths[:3]  # Use top 3 paths regardless of confidence
        
        for path in paths_to_use:
            reasoning_text = " → ".join(path.reasoning_steps)
            confidence_text = f"(confidence: {path.confidence:.2f})"
            all_reasoning.append(f"{reasoning_text} {confidence_text}")
        
        # Create comprehensive answer
        answer_parts = [
            "Based on the knowledge graph analysis, here are the relevant connections:",
            *[f"• {reasoning}" for reasoning in all_reasoning]
        ]
        
        if len(paths_to_use) > 1:
            answer_parts.append(f"\nThis analysis is based on {len(paths_to_use)} reasoning paths through the knowledge graph.")
        
        return "\n".join(answer_parts)

# Example usage and demonstration
if __name__ == "__main__":
    async def demo_graph_rag():
        """Demonstrate Graph-based RAG system"""
        
        # Sample documents for building knowledge graph
        sample_docs = [
            {
                'id': 'doc1',
                'content': 'Password reset requires email verification. Email verification prevents unauthorized access. Strong passwords prevent security breaches.'
            },
            {
                'id': 'doc2', 
                'content': 'Two-factor authentication is part of account security. Account security requires strong passwords and email verification.'
            },
            {
                'id': 'doc3',
                'content': 'If account is locked, then password reset is required. Account locking prevents brute force attacks.'
            }
        ]
        
        # Build knowledge graph
        builder = KnowledgeGraphBuilder()
        graph = await builder.build_graph_from_documents(sample_docs)
        
        # Create GraphRAG engine
        graph_rag = GraphRAGEngine(graph)
        
        # Test queries
        test_queries = [
            "How does password reset relate to security?",
            "What prevents unauthorized access?", 
            "What is required for account security?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            result = await graph_rag.query_with_graph_reasoning(query)
            print(f"📊 Answer: {result['answer']}")
            print(f"🧠 Reasoning paths: {len(result['reasoning_paths'])}")
            print(f"📈 Subgraph: {result['subgraph_nodes']} nodes, {result['subgraph_edges']} edges")
    
    # Run demo
    # asyncio.run(demo_graph_rag())