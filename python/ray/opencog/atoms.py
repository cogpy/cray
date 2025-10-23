"""
Atom classes for the OpenCog AtomSpace.

This module provides the base Atom class and specialized Node and Link classes
that represent knowledge in the distributed AtomSpace.
"""

import uuid
from typing import List, Optional, Dict, Any, Union, Tuple
from .atom_types import AtomType, TYPE_HIERARCHY


class TruthValue:
    """
    Represents a truth value with strength and confidence.
    
    In OpenCog, truth values represent probabilistic assessments of statements.
    """
    
    def __init__(self, strength: float = 1.0, confidence: float = 1.0):
        """
        Initialize a truth value.
        
        Args:
            strength: Probability estimate (0.0 to 1.0)
            confidence: Confidence in the strength estimate (0.0 to 1.0)
        """
        self.strength = max(0.0, min(1.0, strength))
        self.confidence = max(0.0, min(1.0, confidence))
    
    def __str__(self) -> str:
        return f"({self.strength:.3f}, {self.confidence:.3f})"
    
    def __repr__(self) -> str:
        return f"TruthValue(strength={self.strength}, confidence={self.confidence})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TruthValue):
            return False
        return (abs(self.strength - other.strength) < 1e-6 and 
                abs(self.confidence - other.confidence) < 1e-6)


class Atom:
    """
    Base class for all atoms in the AtomSpace.
    
    Atoms are the basic units of knowledge representation in OpenCog.
    They can be nodes (terms) or links (relationships between terms).
    """
    
    def __init__(self, 
                 atom_type: AtomType, 
                 name: Optional[str] = None,
                 truth_value: Optional[TruthValue] = None):
        """
        Initialize an atom.
        
        Args:
            atom_type: The type of this atom
            name: Optional name for the atom
            truth_value: Optional truth value
        """
        self.atom_type = atom_type
        self.name = name or ""
        self.truth_value = truth_value or TruthValue()
        self.handle = str(uuid.uuid4())  # Unique identifier
        self.incoming_set: List['Atom'] = []  # Atoms that reference this atom
        self.values: Dict[str, Any] = {}  # Associated values
    
    def __str__(self) -> str:
        return f"({self.atom_type.value} {self.name})"
    
    def __repr__(self) -> str:
        return f"Atom(type={self.atom_type.value}, name='{self.name}', handle={self.handle[:8]}...)"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return False
        return (self.atom_type == other.atom_type and 
                self.name == other.name)
    
    def __hash__(self) -> int:
        return hash((self.atom_type, self.name))
    
    def is_node(self) -> bool:
        """Check if this atom is a node."""
        return TYPE_HIERARCHY.is_subtype(self.atom_type, AtomType.NODE)
    
    def is_link(self) -> bool:
        """Check if this atom is a link."""
        return TYPE_HIERARCHY.is_subtype(self.atom_type, AtomType.LINK)
    
    def is_type(self, atom_type: AtomType) -> bool:
        """Check if this atom is of the specified type or subtype."""
        return (self.atom_type == atom_type or 
                TYPE_HIERARCHY.is_subtype(self.atom_type, atom_type))
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a value associated with this atom."""
        self.values[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value associated with this atom."""
        return self.values.get(key, default)


class Node(Atom):
    """
    A node atom represents a term or concept.
    
    Nodes are atoms that have a name but no outgoing links to other atoms.
    """
    
    def __init__(self, 
                 atom_type: AtomType,
                 name: str,
                 truth_value: Optional[TruthValue] = None):
        """
        Initialize a node.
        
        Args:
            atom_type: Must be a node type
            name: The name of the node
            truth_value: Optional truth value
        """
        if not TYPE_HIERARCHY.is_subtype(atom_type, AtomType.NODE):
            raise ValueError(f"{atom_type} is not a valid node type")
        
        super().__init__(atom_type, name, truth_value)
    
    def __str__(self) -> str:
        return f"({self.atom_type.value} \"{self.name}\")"


class Link(Atom):
    """
    A link atom represents a relationship between other atoms.
    
    Links have outgoing connections to other atoms (both nodes and links).
    """
    
    def __init__(self, 
                 atom_type: AtomType,
                 outgoing: List[Atom],
                 truth_value: Optional[TruthValue] = None):
        """
        Initialize a link.
        
        Args:
            atom_type: Must be a link type
            outgoing: List of atoms this link connects
            truth_value: Optional truth value
        """
        if not TYPE_HIERARCHY.is_subtype(atom_type, AtomType.LINK):
            raise ValueError(f"{atom_type} is not a valid link type")
        
        super().__init__(atom_type, None, truth_value)
        self.outgoing = outgoing.copy()
        
        # Add this link to the incoming set of each target atom
        for atom in self.outgoing:
            atom.incoming_set.append(self)
    
    def __str__(self) -> str:
        outgoing_str = " ".join(str(atom) for atom in self.outgoing)
        return f"({self.atom_type.value} {outgoing_str})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Link):
            return False
        return (self.atom_type == other.atom_type and 
                self.outgoing == other.outgoing)
    
    def __hash__(self) -> int:
        return hash((self.atom_type, tuple(self.outgoing)))
    
    def get_arity(self) -> int:
        """Get the number of outgoing atoms."""
        return len(self.outgoing)
    
    def get_outgoing_atom(self, index: int) -> Atom:
        """Get the atom at the specified index in the outgoing list."""
        return self.outgoing[index]


# Convenience functions for creating common atom types

def concept_node(name: str, truth_value: Optional[TruthValue] = None) -> Node:
    """Create a ConceptNode."""
    return Node(AtomType.CONCEPT_NODE, name, truth_value)

def predicate_node(name: str, truth_value: Optional[TruthValue] = None) -> Node:
    """Create a PredicateNode."""
    return Node(AtomType.PREDICATE_NODE, name, truth_value)

def variable_node(name: str, truth_value: Optional[TruthValue] = None) -> Node:
    """Create a VariableNode."""  
    return Node(AtomType.VARIABLE_NODE, name, truth_value)

def inheritance_link(child: Atom, parent: Atom, truth_value: Optional[TruthValue] = None) -> Link:
    """Create an InheritanceLink."""
    return Link(AtomType.INHERITANCE_LINK, [child, parent], truth_value)

def evaluation_link(predicate: Atom, arguments: List[Atom], truth_value: Optional[TruthValue] = None) -> Link:
    """Create an EvaluationLink."""
    return Link(AtomType.EVALUATION_LINK, [predicate] + arguments, truth_value)

def list_link(atoms: List[Atom], truth_value: Optional[TruthValue] = None) -> Link:
    """Create a ListLink."""
    return Link(AtomType.LIST_LINK, atoms, truth_value)

def and_link(atoms: List[Atom], truth_value: Optional[TruthValue] = None) -> Link:
    """Create an AndLink."""
    return Link(AtomType.AND_LINK, atoms, truth_value)

def or_link(atoms: List[Atom], truth_value: Optional[TruthValue] = None) -> Link:
    """Create an OrLink."""
    return Link(AtomType.OR_LINK, atoms, truth_value)