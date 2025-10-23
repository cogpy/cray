"""
Basic tests for Ray OpenCog implementation.

These tests verify that the core OpenCog components work correctly
with Ray's distributed computing framework.
"""

import pytest
import ray
from ray.opencog import AtomSpace, DistributedAtomSpace, concept_node, predicate_node
from ray.opencog import inheritance_link, evaluation_link, list_link, TruthValue
from ray.opencog import PatternMatcher, PLN, Pattern, variable_node
from ray.opencog.atom_types import AtomType


class TestAtoms:
    """Test basic atom functionality."""
    
    def test_create_concept_node(self):
        """Test creating concept nodes."""
        node = concept_node("cat")
        assert node.atom_type == AtomType.CONCEPT_NODE
        assert node.name == "cat"
        assert node.is_node()
        assert not node.is_link()
    
    def test_create_inheritance_link(self):
        """Test creating inheritance links."""
        cat = concept_node("cat")
        mammal = concept_node("mammal")
        link = inheritance_link(cat, mammal)
        
        assert link.atom_type == AtomType.INHERITANCE_LINK
        assert link.is_link()
        assert not link.is_node()
        assert link.get_arity() == 2
        assert link.get_outgoing_atom(0) == cat
        assert link.get_outgoing_atom(1) == mammal
    
    def test_truth_values(self):
        """Test truth value operations."""
        tv1 = TruthValue(0.8, 0.9)
        tv2 = TruthValue(0.8, 0.9)
        tv3 = TruthValue(0.7, 0.9)
        
        assert tv1 == tv2
        assert tv1 != tv3
        assert str(tv1) == "(0.800, 0.900)"


class TestAtomSpace:
    """Test AtomSpace functionality."""
    
    def setup_method(self):
        """Set up test AtomSpace."""
        self.atomspace = AtomSpace()
    
    def test_add_atoms(self):
        """Test adding atoms to AtomSpace."""
        cat = concept_node("cat")
        mammal = concept_node("mammal")
        link = inheritance_link(cat, mammal)
        
        added_cat = self.atomspace.add_atom(cat)
        added_mammal = self.atomspace.add_atom(mammal)
        added_link = self.atomspace.add_atom(link)
        
        assert added_cat == cat
        assert added_mammal == mammal
        assert added_link == link
        assert self.atomspace.size() == 3
    
    def test_get_atoms_by_type(self):
        """Test retrieving atoms by type."""
        cat = concept_node("cat")
        dog = concept_node("dog")
        likes = predicate_node("likes")
        
        self.atomspace.add_atom(cat)
        self.atomspace.add_atom(dog)
        self.atomspace.add_atom(likes)
        
        concept_nodes = self.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
        predicate_nodes = self.atomspace.get_atoms_by_type(AtomType.PREDICATE_NODE)
        all_nodes = self.atomspace.get_atoms_by_type(AtomType.NODE)
        
        assert len(concept_nodes) == 2
        assert len(predicate_nodes) == 1
        assert len(all_nodes) == 3
    
    def test_get_atoms_by_name(self):
        """Test retrieving atoms by name."""
        cat1 = concept_node("cat")
        cat2 = concept_node("cat")  # Same name, should be deduplicated
        dog = concept_node("dog")
        
        self.atomspace.add_atom(cat1)
        self.atomspace.add_atom(cat2)
        self.atomspace.add_atom(dog)
        
        cat_atoms = self.atomspace.get_atoms_by_name("cat")
        dog_atoms = self.atomspace.get_atoms_by_name("dog")
        
        assert len(cat_atoms) == 1  # Deduplicated
        assert len(dog_atoms) == 1
        assert self.atomspace.size() == 2


@pytest.mark.skipif(not ray.is_initialized(), reason="Ray not initialized")
class TestDistributedAtomSpace:
    """Test distributed AtomSpace functionality."""
    
    def setup_method(self):
        """Set up test distributed AtomSpace.""" 
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True)
        self.atomspace = DistributedAtomSpace(num_partitions=2)
    
    def test_distributed_add_atoms(self):
        """Test adding atoms to distributed AtomSpace."""
        cat = concept_node("cat")
        mammal = concept_node("mammal")
        
        added_cat = self.atomspace.add_atom(cat)
        added_mammal = self.atomspace.add_atom(mammal)
        
        assert added_cat.name == "cat"
        assert added_mammal.name == "mammal"
        assert self.atomspace.size() == 2
    
    def test_distributed_get_atoms_by_type(self):
        """Test retrieving atoms by type from distributed AtomSpace."""
        cat = concept_node("cat")
        dog = concept_node("dog")
        likes = predicate_node("likes")
        
        self.atomspace.add_atom(cat)
        self.atomspace.add_atom(dog)
        self.atomspace.add_atom(likes)
        
        concept_nodes = self.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
        all_nodes = self.atomspace.get_atoms_by_type(AtomType.NODE)
        
        assert len(concept_nodes) == 2
        assert len(all_nodes) == 3


class TestPatternMatching:
    """Test pattern matching functionality."""
    
    def setup_method(self):
        """Set up test AtomSpace with sample data."""
        self.atomspace = AtomSpace()
        
        # Add sample knowledge
        cat = concept_node("cat")
        mammal = concept_node("mammal")
        animal = concept_node("animal")
        
        self.atomspace.add_atom(inheritance_link(cat, mammal))
        self.atomspace.add_atom(inheritance_link(mammal, animal))
    
    def test_simple_pattern_match(self):
        """Test basic pattern matching."""
        # Pattern: (InheritanceLink $X mammal)
        var_x = variable_node("$X")
        mammal = concept_node("mammal")
        pattern_template = inheritance_link(var_x, mammal)
        pattern = Pattern(pattern_template, [var_x])
        
        matcher = PatternMatcher(self.atomspace)
        matches = matcher.match(pattern)
        
        assert len(matches) == 1
        assert matches[0].get_binding("$X").name == "cat"


class TestPLN:
    """Test PLN reasoning functionality."""
    
    def setup_method(self):
        """Set up test environment for PLN."""
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True)
        
        self.atomspace = AtomSpace()
        self.pln = PLN(self.atomspace, num_inference_actors=2)
        
        # Add sample knowledge with truth values
        cat = concept_node("cat")
        mammal = concept_node("mammal") 
        animal = concept_node("animal")
        
        # cat -> mammal (strength: 0.9, confidence: 0.8)
        cat_mammal = inheritance_link(cat, mammal, TruthValue(0.9, 0.8))
        
        # mammal -> animal (strength: 0.8, confidence: 0.9)
        mammal_animal = inheritance_link(mammal, animal, TruthValue(0.8, 0.9))
        
        self.atomspace.add_atom(cat_mammal)
        self.atomspace.add_atom(mammal_animal)
    
    def test_forward_chaining(self):
        """Test forward chaining inference."""
        initial_size = self.atomspace.size()
        
        # Run forward chaining - should infer cat -> animal
        new_atoms = self.pln.forward_chain(max_iterations=1)
        
        # Should have inferred new knowledge
        final_size = self.atomspace.size()
        assert final_size > initial_size or len(new_atoms) > 0