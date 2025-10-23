#!/usr/bin/env python3
"""
Basic example demonstrating OpenCog functionality using Ray.

This example shows how to:
1. Create atoms (nodes and links)
2. Use the AtomSpace for knowledge storage  
3. Perform pattern matching
4. Use PLN for probabilistic reasoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import opencog
from opencog import concept_node, predicate_node, variable_node
from opencog import inheritance_link, evaluation_link, list_link
from opencog import AtomSpace, PatternMatcher, Pattern, PLN, TruthValue


def basic_atoms_demo():
    """Demonstrate basic atom creation and manipulation."""
    print("=== Basic Atoms Demo ===")
    
    # Create concept nodes
    cat = concept_node("cat")
    mammal = concept_node("mammal")
    animal = concept_node("animal")
    
    # Create inheritance links with truth values
    cat_mammal = inheritance_link(cat, mammal, TruthValue(0.9, 0.8))
    mammal_animal = inheritance_link(mammal, animal, TruthValue(0.8, 0.9))
    
    print(f"Created atoms:")
    print(f"  {cat}")
    print(f"  {mammal}")
    print(f"  {animal}")
    print(f"  {cat_mammal} - TV: {cat_mammal.truth_value}")
    print(f"  {mammal_animal} - TV: {mammal_animal.truth_value}")
    print()


def atomspace_demo():
    """Demonstrate AtomSpace functionality."""
    print("=== AtomSpace Demo ===")
    
    # Create AtomSpace
    atomspace = AtomSpace()
    
    # Add knowledge about animals
    cat = concept_node("cat")
    dog = concept_node("dog")
    mammal = concept_node("mammal")
    animal = concept_node("animal")
    
    atomspace.add_atom(cat)
    atomspace.add_atom(dog)
    atomspace.add_atom(mammal)
    atomspace.add_atom(animal)
    
    # Add relationships
    atomspace.add_atom(inheritance_link(cat, mammal, TruthValue(0.9, 0.8)))
    atomspace.add_atom(inheritance_link(dog, mammal, TruthValue(0.85, 0.9)))
    atomspace.add_atom(inheritance_link(mammal, animal, TruthValue(0.8, 0.95)))
    
    print(f"AtomSpace contains {atomspace.size()} atoms")
    
    # Query by type
    concept_nodes = atomspace.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)
    inheritance_links = atomspace.get_atoms_by_type(opencog.AtomType.INHERITANCE_LINK)
    
    print(f"Found {len(concept_nodes)} concept nodes:")
    for node in concept_nodes:
        print(f"  {node}")
    
    print(f"Found {len(inheritance_links)} inheritance links:")
    for link in inheritance_links:
        print(f"  {link} - TV: {link.truth_value}")
    
    # Query incoming set
    incoming = atomspace.get_incoming_set(mammal)
    print(f"Mammal has {len(incoming)} incoming links:")
    for link in incoming:
        print(f"  {link}")
    
    print()
    return atomspace


def pattern_matching_demo(atomspace):
    """Demonstrate pattern matching."""
    print("=== Pattern Matching Demo ===")
    
    # Create a pattern: (InheritanceLink $X mammal)
    var_x = variable_node("$X")
    mammal = concept_node("mammal")
    pattern_template = inheritance_link(var_x, mammal)
    pattern = Pattern(pattern_template, [var_x])
    
    print(f"Pattern: {pattern_template}")
    
    # Find matches
    matcher = PatternMatcher(atomspace)
    matches = matcher.match(pattern)
    
    print(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches):
        bound_atom = match.get_binding("$X")
        print(f"  Match {i+1}: $X = {bound_atom}")
    
    print()


def pln_reasoning_demo(atomspace):
    """Demonstrate PLN reasoning."""
    print("=== PLN Reasoning Demo ===")
    
    # Initialize PLN system
    pln = PLN(atomspace)
    
    # Perform forward chaining to infer new knowledge
    print("Performing forward chaining inference...")
    initial_size = atomspace.size()
    
    new_atoms = pln.forward_chain(max_iterations=2, min_confidence=0.1)
    
    final_size = atomspace.size()
    print(f"AtomSpace grew from {initial_size} to {final_size} atoms")
    
    if new_atoms:
        print("Newly inferred atoms:")
        for atom in new_atoms:
            print(f"  {atom} - TV: {atom.truth_value}")
    else:
        print("No new atoms inferred (this is expected without Ray)")
    
    # Demonstrate conjunction/disjunction
    cat = concept_node("cat")
    dog = concept_node("dog")
    
    conjunction_tv = pln.compute_conjunction([cat, dog])
    disjunction_tv = pln.compute_disjunction([cat, dog])
    
    print(f"Conjunction TV(cat AND dog): {conjunction_tv}")
    print(f"Disjunction TV(cat OR dog): {disjunction_tv}")
    
    print()


def main():
    """Run all demos."""
    print("OpenCog on Ray - Basic Example")
    print("=" * 40)
    
    # Run basic demos
    basic_atoms_demo()
    atomspace = atomspace_demo()
    pattern_matching_demo(atomspace)
    pln_reasoning_demo(atomspace)
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()