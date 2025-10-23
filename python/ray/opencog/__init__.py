"""
OpenCog implementation using Ray distributed computing framework.

This module provides a distributed implementation of the OpenCog cognitive architecture
using Ray's capabilities for scaling across multiple nodes. Key components include:

- AtomSpace: Distributed knowledge representation database
- Atomese: Language for representing knowledge and procedures  
- Pattern Matcher: Distributed pattern matching and mining
- PLN: Probabilistic Logic Networks for reasoning
"""

# Import core components
from .atom_types import AtomType
from .atoms import Atom, Node, Link, TruthValue
from .atoms import concept_node, predicate_node, variable_node
from .atoms import inheritance_link, evaluation_link, list_link
from .atomspace import AtomSpace, DistributedAtomSpace
from .pattern_matcher import PatternMatcher, Pattern
from .pln import PLN

__all__ = [
    "AtomType",
    "Atom", 
    "Node",
    "Link",
    "TruthValue",
    "concept_node",
    "predicate_node", 
    "variable_node",
    "inheritance_link",
    "evaluation_link",
    "list_link",
    "AtomSpace",
    "DistributedAtomSpace",
    "PatternMatcher", 
    "Pattern",
    "PLN"
]