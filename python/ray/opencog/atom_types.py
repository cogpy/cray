"""
Atom types used in the OpenCog AtomSpace.

This module defines the type hierarchy for atoms in the distributed AtomSpace,
following the OpenCog tradition but adapted for Ray's distributed architecture.
"""

from enum import Enum
from typing import Dict, Set, Optional


class AtomType(Enum):
    """
    Enumeration of atom types in the OpenCog type hierarchy.
    
    This follows the traditional OpenCog type system with nodes and links.
    """
    # Base types
    ATOM = "Atom"
    
    # Node types  
    NODE = "Node"
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    VARIABLE_NODE = "VariableNode"
    GROUNDED_PREDICATE_NODE = "GroundedPredicateNode"
    SCHEMA_NODE = "SchemaNode"
    GROUNDED_SCHEMA_NODE = "GroundedSchemaNode"
    NUMBER_NODE = "NumberNode"
    
    # Link types
    LINK = "Link"
    ORDERED_LINK = "OrderedLink"
    UNORDERED_LINK = "UnorderedLink"
    
    # Logical links
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink" 
    IMPLICATION_LINK = "ImplicationLink"
    EQUIVALENCE_LINK = "EquivalenceLink"
    
    # Set links
    LIST_LINK = "ListLink"
    SET_LINK = "SetLink"
    MEMBER_LINK = "MemberLink"
    SUBSET_LINK = "SubsetLink"
    
    # Evaluative links
    EVALUATION_LINK = "EvaluationLink"
    EXECUTION_LINK = "ExecutionLink"
    
    # Logical operators
    AND_LINK = "AndLink"
    OR_LINK = "OrLink"  
    NOT_LINK = "NotLink"
    
    # Quantifiers
    FOR_ALL_LINK = "ForAllLink"
    EXISTS_LINK = "ExistsLink"
    
    # Pattern matching
    BIND_LINK = "BindLink"
    GET_LINK = "GetLink"
    SATISFACTION_LINK = "SatisfactionLink"


class TypeHierarchy:
    """
    Manages the type hierarchy for atom types.
    
    Provides utilities for checking type relationships and inheritance.
    """
    
    def __init__(self):
        self._type_hierarchy = self._build_hierarchy()
    
    def _build_hierarchy(self) -> Dict[AtomType, Set[AtomType]]:
        """Build the type hierarchy mapping."""
        hierarchy = {}
        
        # All types inherit from ATOM
        for atom_type in AtomType:
            hierarchy[atom_type] = {AtomType.ATOM}
            
        # Nodes inherit from NODE  
        node_types = {
            AtomType.CONCEPT_NODE, AtomType.PREDICATE_NODE, AtomType.VARIABLE_NODE,
            AtomType.GROUNDED_PREDICATE_NODE, AtomType.SCHEMA_NODE, 
            AtomType.GROUNDED_SCHEMA_NODE, AtomType.NUMBER_NODE
        }
        for node_type in node_types:
            hierarchy[node_type].add(AtomType.NODE)
            
        # Links inherit from LINK
        link_types = {
            AtomType.ORDERED_LINK, AtomType.UNORDERED_LINK, AtomType.INHERITANCE_LINK,
            AtomType.SIMILARITY_LINK, AtomType.IMPLICATION_LINK, AtomType.EQUIVALENCE_LINK,
            AtomType.LIST_LINK, AtomType.SET_LINK, AtomType.MEMBER_LINK, 
            AtomType.SUBSET_LINK, AtomType.EVALUATION_LINK, AtomType.EXECUTION_LINK,
            AtomType.AND_LINK, AtomType.OR_LINK, AtomType.NOT_LINK,
            AtomType.FOR_ALL_LINK, AtomType.EXISTS_LINK, AtomType.BIND_LINK,
            AtomType.GET_LINK, AtomType.SATISFACTION_LINK
        }
        for link_type in link_types:
            hierarchy[link_type].add(AtomType.LINK)
            
        # Ordered links
        ordered_links = {
            AtomType.INHERITANCE_LINK, AtomType.IMPLICATION_LINK, AtomType.LIST_LINK,
            AtomType.MEMBER_LINK, AtomType.SUBSET_LINK, AtomType.EVALUATION_LINK,
            AtomType.EXECUTION_LINK, AtomType.BIND_LINK, AtomType.GET_LINK,
            AtomType.SATISFACTION_LINK
        }
        for link_type in ordered_links:
            hierarchy[link_type].add(AtomType.ORDERED_LINK)
            
        # Unordered links
        unordered_links = {
            AtomType.SIMILARITY_LINK, AtomType.EQUIVALENCE_LINK, AtomType.SET_LINK,
            AtomType.AND_LINK, AtomType.OR_LINK
        }
        for link_type in unordered_links:
            hierarchy[link_type].add(AtomType.UNORDERED_LINK)
            
        return hierarchy
    
    def is_subtype(self, child_type: AtomType, parent_type: AtomType) -> bool:
        """Check if child_type is a subtype of parent_type."""
        return parent_type in self._type_hierarchy.get(child_type, set())
    
    def get_supertypes(self, atom_type: AtomType) -> Set[AtomType]:
        """Get all supertypes of the given atom type."""
        return self._type_hierarchy.get(atom_type, set()).copy()


# Global type hierarchy instance
TYPE_HIERARCHY = TypeHierarchy()