# OpenCog on Ray

This module implements the OpenCog cognitive architecture using Ray's distributed computing capabilities. It provides a scalable, distributed implementation of OpenCog's key components for artificial general intelligence (AGI) applications.

## Overview

OpenCog is a framework for artificial general intelligence that uses knowledge representation, reasoning, and learning. This Ray implementation distributes OpenCog's computation across multiple nodes for enhanced scalability and performance.

### Key Components

- **AtomSpace**: Distributed knowledge representation database
- **Atomese**: Language for representing knowledge and procedures  
- **Pattern Matcher**: Distributed pattern matching and mining
- **PLN (Probabilistic Logic Networks)**: Probabilistic reasoning system

## Installation

This module is part of Ray. No additional installation is required if you have Ray installed.

```bash
pip install ray
```

## Quick Start

```python
import ray
import ray.opencog as opencog

# Create atoms
cat = opencog.concept_node("cat")
mammal = opencog.concept_node("mammal")
animal = opencog.concept_node("animal")

# Create relationships with truth values
cat_mammal = opencog.inheritance_link(cat, mammal, opencog.TruthValue(0.9, 0.8))
mammal_animal = opencog.inheritance_link(mammal, animal, opencog.TruthValue(0.8, 0.9))

# Use AtomSpace for knowledge storage
atomspace = opencog.AtomSpace()
atomspace.add_atom(cat_mammal)
atomspace.add_atom(mammal_animal)

# Pattern matching
pattern = opencog.Pattern(
    opencog.inheritance_link(opencog.variable_node("$X"), mammal),
    [opencog.variable_node("$X")]
)
matcher = opencog.PatternMatcher(atomspace)
matches = matcher.match(pattern)

# Probabilistic reasoning
pln = opencog.PLN(atomspace)
new_atoms = pln.forward_chain(max_iterations=2)
```

## Core Classes

### Atoms

Atoms are the basic units of knowledge representation:

```python
# Nodes represent entities or concepts
cat = opencog.concept_node("cat")
likes = opencog.predicate_node("likes")
var_x = opencog.variable_node("$X")

# Links represent relationships
inheritance = opencog.inheritance_link(cat, mammal)
evaluation = opencog.evaluation_link(likes, [cat, opencog.concept_node("fish")])

# Truth values represent probabilistic beliefs
tv = opencog.TruthValue(strength=0.85, confidence=0.9)
```

### AtomSpace

The AtomSpace stores and manages atoms:

```python
atomspace = opencog.AtomSpace()

# Add atoms
atomspace.add_atom(cat)
atomspace.add_atom(inheritance)

# Query by type
concept_nodes = atomspace.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)

# Query by name
cat_atoms = atomspace.get_atoms_by_name("cat")

# Get incoming links
incoming = atomspace.get_incoming_set(cat)
```

### Distributed AtomSpace

For distributed processing across multiple nodes:

```python
# Initialize Ray first
ray.init()

# Create distributed AtomSpace
distributed_atomspace = opencog.DistributedAtomSpace(num_partitions=4)

# Use same API as regular AtomSpace
distributed_atomspace.add_atom(cat)
concept_nodes = distributed_atomspace.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)
```

### Pattern Matching

Find atoms that match specified patterns:

```python
# Create pattern: find all things that inherit from mammal
pattern_template = opencog.inheritance_link(
    opencog.variable_node("$X"),  # Variable to bind
    opencog.concept_node("mammal")
)
pattern = opencog.Pattern(pattern_template)

# Execute pattern match
matcher = opencog.PatternMatcher(atomspace)
matches = matcher.match(pattern)

# Access bindings
for match in matches:
    bound_atom = match.get_binding("$X")
    print(f"Found: {bound_atom}")
```

### Probabilistic Logic Networks (PLN)

Perform probabilistic reasoning and inference:

```python
pln = opencog.PLN(atomspace)

# Forward chaining inference
new_atoms = pln.forward_chain(max_iterations=5, min_confidence=0.1)

# Backward chaining
target = opencog.inheritance_link(cat, animal)
truth_value = pln.backward_chain(target)

# Compute conjunctions and disjunctions
conjunction_tv = pln.compute_conjunction([cat, dog])
disjunction_tv = pln.compute_disjunction([cat, dog])
```

## Truth Values

Truth values represent probabilistic beliefs about statements:

```python
# Create truth value with strength and confidence
tv = opencog.TruthValue(strength=0.8, confidence=0.9)

# Attach to atoms
cat_mammal = opencog.inheritance_link(cat, mammal, tv)

# Access truth value
print(f"Strength: {tv.strength}, Confidence: {tv.confidence}")
```

## Atom Types

The type hierarchy follows OpenCog conventions:

### Nodes
- `CONCEPT_NODE`: Represents concepts or entities
- `PREDICATE_NODE`: Represents predicates or relations  
- `VARIABLE_NODE`: Represents variables for pattern matching
- `NUMBER_NODE`: Represents numeric values

### Links  
- `INHERITANCE_LINK`: Represents inheritance relationships
- `EVALUATION_LINK`: Represents predicate evaluations
- `LIST_LINK`: Represents ordered collections
- `AND_LINK`, `OR_LINK`: Logical operators
- `BIND_LINK`: Pattern-based queries

## Examples

### Basic Knowledge Representation

```python
import ray.opencog as opencog

# Create a simple knowledge base
atomspace = opencog.AtomSpace()

# Animals hierarchy
cat = opencog.concept_node("cat")
dog = opencog.concept_node("dog")  
mammal = opencog.concept_node("mammal")
animal = opencog.concept_node("animal")

# Add inheritance relationships
atomspace.add_atom(opencog.inheritance_link(cat, mammal, opencog.TruthValue(0.9, 0.8)))
atomspace.add_atom(opencog.inheritance_link(dog, mammal, opencog.TruthValue(0.85, 0.9)))
atomspace.add_atom(opencog.inheritance_link(mammal, animal, opencog.TruthValue(0.8, 0.95)))

print(f"Knowledge base has {atomspace.size()} atoms")
```

### Inference with PLN

```python
# Set up PLN reasoning
pln = opencog.PLN(atomspace)

# The system can infer: cat -> animal and dog -> animal
# from: cat -> mammal, dog -> mammal, mammal -> animal
new_atoms = pln.forward_chain(max_iterations=3)

for atom in new_atoms:
    print(f"Inferred: {atom} with TV: {atom.truth_value}")
```

### Distributed Processing

```python
import ray

# Initialize Ray cluster
ray.init()

# Create distributed AtomSpace
distributed_as = opencog.DistributedAtomSpace(num_partitions=8)

# Add knowledge across partitions
for i in range(1000):
    concept = opencog.concept_node(f"concept_{i}")
    distributed_as.add_atom(concept)

# Query across all partitions
all_concepts = distributed_as.get_atoms_by_type(opencog.AtomType.CONCEPT_NODE)
print(f"Found {len(all_concepts)} concepts across all partitions")
```

## Architecture

### Local vs Distributed

- **Local AtomSpace**: Single-node storage and processing
- **Distributed AtomSpace**: Multi-node storage with Ray actors
- **Automatic Fallback**: Gracefully handles absence of Ray

### Partitioning Strategy

Atoms are distributed across partitions using consistent hashing:
- Ensures balanced load distribution
- Minimizes cross-partition queries
- Supports dynamic scaling

### Integration with Ray

The implementation leverages Ray's features:
- **Ray Actors**: For distributed AtomSpace partitions
- **Ray Tasks**: For parallel pattern matching and inference
- **Ray Objects**: For efficient atom sharing

## Performance Considerations

### Scalability

- Linear scaling with number of Ray nodes
- Parallel processing of inference tasks
- Distributed pattern matching across partitions

### Memory Management

- Atoms are stored in-memory for fast access
- Partitioning reduces memory pressure per node
- Truth values use efficient representation

### Network Communication

- Minimized through intelligent partitioning
- Batch operations when possible
- Asynchronous processing with Ray futures

## API Reference

### Core Functions

```python
# Atom creation
concept_node(name: str, truth_value: Optional[TruthValue] = None) -> Node
predicate_node(name: str, truth_value: Optional[TruthValue] = None) -> Node
variable_node(name: str, truth_value: Optional[TruthValue] = None) -> Node
inheritance_link(child: Atom, parent: Atom, truth_value: Optional[TruthValue] = None) -> Link
evaluation_link(predicate: Atom, arguments: List[Atom], truth_value: Optional[TruthValue] = None) -> Link

# AtomSpace operations  
AtomSpace.add_atom(atom: Atom) -> Atom
AtomSpace.get_atoms_by_type(atom_type: AtomType, include_subtypes: bool = True) -> List[Atom]
AtomSpace.get_atoms_by_name(name: str) -> List[Atom]
AtomSpace.get_incoming_set(atom: Union[Atom, str]) -> List[Atom]

# Pattern matching
PatternMatcher.match(pattern: Pattern) -> List[PatternMatch]

# PLN reasoning
PLN.forward_chain(max_iterations: int = 10, min_confidence: float = 0.1) -> List[Atom]
PLN.backward_chain(target: Atom, max_depth: int = 3) -> Optional[TruthValue]
```

## Contributing

This implementation provides a solid foundation for OpenCog on Ray. Contributions are welcome for:

- Additional atom types and link types
- Advanced PLN inference rules  
- Performance optimizations
- Integration with other Ray libraries
- Documentation and examples

## References

- [OpenCog Foundation](https://opencog.org/)
- [Ray Documentation](https://docs.ray.io/)  
- [OpenCog AtomSpace](https://wiki.opencog.org/w/AtomSpace)
- [Probabilistic Logic Networks](https://wiki.opencog.org/w/PLN)