#!/usr/bin/env python3
"""
Distributed OpenCog example using Ray for scaling.

This example demonstrates how to use Ray's distributed computing capabilities
with OpenCog for large-scale knowledge representation and reasoning.

Note: This example requires Ray to be properly initialized.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Some features will not work.")

import opencog
from opencog import concept_node, predicate_node, variable_node
from opencog import inheritance_link, evaluation_link, list_link
from opencog import AtomSpace, DistributedAtomSpace, PatternMatcher, Pattern, PLN, TruthValue


def create_large_knowledge_base(atomspace, num_concepts=1000):
    """Create a large knowledge base for testing scalability."""
    print(f"Creating knowledge base with {num_concepts} concepts...")
    
    start_time = time.time()
    
    # Create concept hierarchy
    concepts = []
    for i in range(num_concepts):
        concept = concept_node(f"concept_{i}")
        concepts.append(concept)
        atomspace.add_atom(concept)
    
    # Create inheritance relationships
    relationships = []
    for i in range(min(500, num_concepts - 1)):
        parent_idx = (i + 1) % num_concepts
        child_idx = i
        
        inheritance = inheritance_link(
            concepts[child_idx], 
            concepts[parent_idx],
            TruthValue(0.8 + (i % 20) * 0.01, 0.7 + (i % 30) * 0.01)
        )
        relationships.append(inheritance)
        atomspace.add_atom(inheritance)
    
    # Create evaluation relationships  
    likes_predicate = predicate_node("likes")
    atomspace.add_atom(likes_predicate)
    
    for i in range(min(200, num_concepts // 2)):
        concept1 = concepts[i]
        concept2 = concepts[(i + 10) % num_concepts]
        
        evaluation = evaluation_link(
            likes_predicate,
            [concept1, concept2],
            TruthValue(0.5 + (i % 50) * 0.01, 0.6 + (i % 40) * 0.01)
        )
        atomspace.add_atom(evaluation)
    
    end_time = time.time()
    print(f"Created {atomspace.size()} atoms in {end_time - start_time:.2f} seconds")
    
    return concepts, relationships


def benchmark_queries(atomspace, name="AtomSpace"):
    """Benchmark common queries on the AtomSpace."""
    print(f"\n=== Benchmarking {name} ===")
    
    start_time = time.time()
    
    # Query by type
    concept_nodes = atomspace.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)
    inheritance_links = atomspace.get_atoms_by_type(opencog.AtomType.INHERITANCE_LINK)
    evaluation_links = atomspace.get_atoms_by_type(opencog.AtomType.EVALUATION_LINK)
    
    type_query_time = time.time() - start_time
    
    print(f"Found {len(concept_nodes)} concept nodes")
    print(f"Found {len(inheritance_links)} inheritance links")
    print(f"Found {len(evaluation_links)} evaluation links")
    print(f"Type queries took: {type_query_time:.3f} seconds")
    
    # Query by name
    start_time = time.time()
    concept_0 = atomspace.get_atoms_by_name("concept_0")
    concept_100 = atomspace.get_atoms_by_name("concept_100") if len(concept_nodes) > 100 else []
    name_query_time = time.time() - start_time
    
    print(f"Name queries took: {name_query_time:.3f} seconds")
    
    # Incoming set queries
    if concept_0:
        start_time = time.time()
        incoming = atomspace.get_incoming_set(concept_0[0])
        incoming_query_time = time.time() - start_time
        print(f"concept_0 has {len(incoming)} incoming links")
        print(f"Incoming query took: {incoming_query_time:.3f} seconds")


def pattern_matching_benchmark(atomspace, name="AtomSpace"):
    """Benchmark pattern matching operations."""
    print(f"\n=== Pattern Matching Benchmark ({name}) ===")
    
    # Pattern 1: Find all inheritance relationships
    var_x = variable_node("$X")
    var_y = variable_node("$Y")
    
    inheritance_pattern = Pattern(
        inheritance_link(var_x, var_y),
        [var_x, var_y]
    )
    
    start_time = time.time()
    matcher = PatternMatcher(atomspace)
    inheritance_matches = matcher.match(inheritance_pattern)
    inheritance_time = time.time() - start_time
    
    print(f"Found {len(inheritance_matches)} inheritance patterns in {inheritance_time:.3f} seconds")
    
    # Pattern 2: Find specific inheritance target
    if len(inheritance_matches) > 10:
        target_concept = inheritance_matches[5].get_binding("$Y")  # Get a concept that's inherited from
        
        specific_pattern = Pattern(
            inheritance_link(variable_node("$X"), target_concept),
            [variable_node("$X")]
        )
        
        start_time = time.time()
        specific_matches = matcher.match(specific_pattern)
        specific_time = time.time() - start_time
        
        print(f"Found {len(specific_matches)} concepts inheriting from {target_concept} in {specific_time:.3f} seconds")


def pln_reasoning_benchmark(atomspace, name="AtomSpace"):
    """Benchmark PLN reasoning operations."""
    print(f"\n=== PLN Reasoning Benchmark ({name}) ===")
    
    pln = PLN(atomspace)
    
    # Forward chaining
    start_time = time.time()
    initial_size = atomspace.size()
    new_atoms = pln.forward_chain(max_iterations=2, min_confidence=0.1)
    forward_time = time.time() - start_time
    final_size = atomspace.size()
    
    print(f"Forward chaining: {initial_size} -> {final_size} atoms ({len(new_atoms)} new)")
    print(f"Forward chaining took: {forward_time:.3f} seconds")
    
    # Conjunction/disjunction operations
    concepts = atomspace.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)[:10]  # Get first 10 concepts
    
    if len(concepts) >= 2:
        start_time = time.time()
        conjunction_tv = pln.compute_conjunction(concepts[:3])
        disjunction_tv = pln.compute_disjunction(concepts[:3])
        logic_time = time.time() - start_time
        
        print(f"Conjunction TV: {conjunction_tv}")
        print(f"Disjunction TV: {disjunction_tv}")
        print(f"Logic operations took: {logic_time:.3f} seconds")


def compare_local_vs_distributed():
    """Compare performance between local and distributed AtomSpace."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON: Local vs Distributed AtomSpace")
    print("=" * 60)
    
    num_concepts = 500  # Smaller for demonstration
    
    # Test local AtomSpace
    print("\n--- Local AtomSpace ---")
    local_atomspace = AtomSpace()
    create_large_knowledge_base(local_atomspace, num_concepts)
    benchmark_queries(local_atomspace, "Local")
    pattern_matching_benchmark(local_atomspace, "Local")
    pln_reasoning_benchmark(local_atomspace, "Local")
    
    # Test distributed AtomSpace (if Ray is available)
    if RAY_AVAILABLE:
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            print("\n--- Distributed AtomSpace ---")
            distributed_atomspace = DistributedAtomSpace(num_partitions=4)
            create_large_knowledge_base(distributed_atomspace, num_concepts)
            benchmark_queries(distributed_atomspace, "Distributed")
            pattern_matching_benchmark(distributed_atomspace, "Distributed")  
            pln_reasoning_benchmark(distributed_atomspace, "Distributed")
            
        except Exception as e:
            print(f"Distributed AtomSpace test failed: {e}")
            print("This may be due to Ray not being properly set up")
    else:
        print("\n--- Distributed AtomSpace ---")
        print("Skipped: Ray not available")


def demonstrate_scaling():
    """Demonstrate how OpenCog scales with Ray."""
    print("=" * 60)
    print("SCALING DEMONSTRATION")
    print("=" * 60)
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n--- Testing with {size} concepts ---")
        
        atomspace = AtomSpace()
        start_time = time.time()
        create_large_knowledge_base(atomspace, size)
        creation_time = time.time() - start_time
        
        start_time = time.time()
        concept_nodes = atomspace.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)
        query_time = time.time() - start_time
        
        print(f"Creation time: {creation_time:.3f}s")
        print(f"Query time: {query_time:.3f}s")  
        print(f"Total atoms: {atomspace.size()}")
        print(f"Concepts found: {len(concept_nodes)}")


def main():
    """Run the distributed OpenCog demonstration."""
    print("OpenCog on Ray - Distributed Computing Example")
    print("=" * 60)
    
    if not RAY_AVAILABLE:
        print("Note: Ray is not available. Running local-only demonstrations.")
    
    # Run scaling demonstration
    demonstrate_scaling()
    
    # Compare local vs distributed
    compare_local_vs_distributed()
    
    print("\n" + "=" * 60)
    print("Distributed example completed!")
    
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()