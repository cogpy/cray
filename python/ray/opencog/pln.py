"""
Probabilistic Logic Networks (PLN) implementation for distributed reasoning.

PLN provides probabilistic inference capabilities for the OpenCog framework,
adapted to work with Ray's distributed computing model.
"""

import math
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
from typing import List, Optional, Dict, Union, Tuple
from .atoms import Atom, Link, TruthValue, inheritance_link, and_link, or_link
from .atom_types import AtomType
from .atomspace import AtomSpace, DistributedAtomSpace


class PLNFormula:
    """
    Base class for PLN inference formulas.
    
    PLN formulas define how truth values are computed for inferred atoms.
    """
    
    @staticmethod
    def deduction(ab_tv: TruthValue, bc_tv: TruthValue) -> TruthValue:
        """
        Deduction rule: From A->B and B->C, infer A->C.
        
        Args:
            ab_tv: Truth value of A->B
            bc_tv: Truth value of B->C
            
        Returns:
            Truth value of A->C
        """
        # PLN deduction formula
        strength = ab_tv.strength * bc_tv.strength
        confidence = min(ab_tv.confidence, bc_tv.confidence) * ab_tv.strength * bc_tv.strength
        return TruthValue(strength, confidence)
    
    @staticmethod
    def inversion(ab_tv: TruthValue) -> TruthValue:
        """
        Inversion rule: From A->B, infer B->A (with modified truth value).
        
        Args:
            ab_tv: Truth value of A->B
            
        Returns:
            Truth value of B->A
        """
        # Simple inversion formula (can be made more sophisticated)
        strength = ab_tv.strength * ab_tv.confidence
        confidence = ab_tv.confidence * 0.5  # Reduce confidence
        return TruthValue(strength, confidence)
    
    @staticmethod
    def conjunction(tvs: List[TruthValue]) -> TruthValue:
        """
        Conjunction: Compute truth value for AND of multiple atoms.
        
        Args:
            tvs: List of truth values to combine
            
        Returns:
            Combined truth value
        """
        if not tvs:
            return TruthValue(0.0, 0.0)
        
        # Product of strengths, minimum confidence
        strength = 1.0
        min_confidence = 1.0
        
        for tv in tvs:
            strength *= tv.strength
            min_confidence = min(min_confidence, tv.confidence)
        
        return TruthValue(strength, min_confidence)
    
    @staticmethod
    def disjunction(tvs: List[TruthValue]) -> TruthValue:
        """
        Disjunction: Compute truth value for OR of multiple atoms.
        
        Args:
            tvs: List of truth values to combine
            
        Returns:
            Combined truth value
        """
        if not tvs:
            return TruthValue(0.0, 0.0)
        
        # Probabilistic OR formula
        complement_product = 1.0
        min_confidence = 1.0
        
        for tv in tvs:
            complement_product *= (1.0 - tv.strength)
            min_confidence = min(min_confidence, tv.confidence)
        
        strength = 1.0 - complement_product
        return TruthValue(strength, min_confidence)
    
    @staticmethod
    def inheritance_to_similarity(inh1_tv: TruthValue, inh2_tv: TruthValue) -> TruthValue:
        """
        Convert inheritance to similarity: From A->C and B->C, infer A<->B.
        
        Args:
            inh1_tv: Truth value of A->C  
            inh2_tv: Truth value of B->C
            
        Returns:
            Truth value of similarity between A and B
        """
        # Simplified similarity formula based on shared inheritance
        strength = min(inh1_tv.strength, inh2_tv.strength)
        confidence = min(inh1_tv.confidence, inh2_tv.confidence) * 0.8
        return TruthValue(strength, confidence)


if RAY_AVAILABLE:
    @ray.remote
    class PLNInferenceActor:
        """
        Ray actor for distributed PLN inference.
        
        Each actor can perform inference tasks on its assigned atoms.
        """
        
        def __init__(self, actor_id: int):
            """Initialize the PLN inference actor."""
            self.actor_id = actor_id
        
        def apply_deduction_rule(self, 
                               premise1: Atom, 
                               premise2: Atom) -> Optional[Atom]:
            """
            Apply deduction rule to two premises.
            
            Args:
                premise1: First premise (A->B)
                premise2: Second premise (B->C)
                
            Returns:
                Inferred conclusion (A->C) or None if not applicable
            """
            # Check if premises are inheritance links
            if (premise1.atom_type != AtomType.INHERITANCE_LINK or
                premise2.atom_type != AtomType.INHERITANCE_LINK):
                return None
            
            # Check if they can be chained (B in first equals A in second)
            if premise1.outgoing[1] != premise2.outgoing[0]:
                return None
            
            # Apply deduction formula
            new_tv = PLNFormula.deduction(premise1.truth_value, premise2.truth_value)
            
            # Create new inheritance link A->C
            conclusion = inheritance_link(
                premise1.outgoing[0],  # A
                premise2.outgoing[1],  # C  
                new_tv
            )
            
            return conclusion
        
        def apply_conjunction_rule(self, premises: List[Atom]) -> Optional[Atom]:
            """
            Apply conjunction rule to multiple premises.
            
            Args:
                premises: List of atoms to combine with AND
                
            Returns:
                Conjunctive atom or None if not applicable
            """
            if len(premises) < 2:
                return None
            
            # Compute conjunction truth value
            tvs = [atom.truth_value for atom in premises]
            new_tv = PLNFormula.conjunction(tvs)
            
            # Create AND link
            return and_link(premises, new_tv)
        
        def apply_disjunction_rule(self, premises: List[Atom]) -> Optional[Atom]:
            """
            Apply disjunction rule to multiple premises.
            
            Args:
                premises: List of atoms to combine with OR
                
            Returns:
                Disjunctive atom or None if not applicable
            """
            if len(premises) < 2:
                return None
            
            # Compute disjunction truth value  
            tvs = [atom.truth_value for atom in premises]
            new_tv = PLNFormula.disjunction(tvs)
            
            # Create OR link
            return or_link(premises, new_tv)
else:
    # Placeholder when Ray is not available
    class PLNInferenceActor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Ray is not available. Cannot use PLNInferenceActor.")


class PLN:
    """
    Main PLN reasoning system for distributed probabilistic inference.
    
    Coordinates inference across multiple Ray actors for scalability.
    """
    
    def __init__(self, 
                 atomspace: Union[AtomSpace, DistributedAtomSpace],
                 num_inference_actors: int = 4):
        """
        Initialize the PLN system.
        
        Args:
            atomspace: The AtomSpace to perform inference on
            num_inference_actors: Number of Ray actors for parallel inference
        """
        self.atomspace = atomspace
        self.num_inference_actors = num_inference_actors
        
        # Only create Ray actors if Ray is available and we have a distributed atomspace
        if RAY_AVAILABLE and isinstance(atomspace, DistributedAtomSpace):
            self.inference_actors = [
                PLNInferenceActor.remote(i) 
                for i in range(num_inference_actors)
            ]
        else:
            self.inference_actors = []
    
    def forward_chain(self, 
                     max_iterations: int = 10,
                     min_confidence: float = 0.1) -> List[Atom]:
        """
        Perform forward chaining inference.
        
        Args:
            max_iterations: Maximum number of inference iterations
            min_confidence: Minimum confidence threshold for new inferences
            
        Returns:
            List of newly inferred atoms
        """
        new_atoms = []
        
        for iteration in range(max_iterations):
            # Get all inheritance links for deduction
            inheritance_links = self.atomspace.get_atoms_by_type(
                AtomType.INHERITANCE_LINK, include_subtypes=False
            )
            
            if len(inheritance_links) < 2:
                break
            
            # Apply deduction rules in parallel
            iteration_inferences = self._apply_deduction_parallel(
                inheritance_links, min_confidence
            )
            
            if not iteration_inferences:
                break  # No new inferences
            
            # Add new inferences to AtomSpace
            for atom in iteration_inferences:
                added_atom = self.atomspace.add_atom(atom)
                new_atoms.append(added_atom)
        
        return new_atoms
    
    def _apply_deduction_parallel(self, 
                                inheritance_links: List[Atom],
                                min_confidence: float) -> List[Atom]:
        """Apply deduction rules in parallel across inference actors."""
        new_inferences = []
        
        # If we have Ray actors, use parallel processing
        if self.inference_actors and RAY_AVAILABLE:
            futures = []
            
            # Distribute deduction tasks across actors
            for i in range(len(inheritance_links)):
                for j in range(i + 1, len(inheritance_links)):
                    actor_index = (i + j) % len(self.inference_actors)
                    actor = self.inference_actors[actor_index]
                    
                    future = actor.apply_deduction_rule.remote(
                        inheritance_links[i], 
                        inheritance_links[j]
                    )
                    futures.append(future)
            
            # Collect results
            if futures:
                results = ray.get(futures)
                
                # Filter out None results and low-confidence inferences
                for result in results:
                    if (result is not None and 
                        result.truth_value.confidence >= min_confidence):
                        new_inferences.append(result)
        else:
            # Fall back to local sequential processing
            for i in range(len(inheritance_links)):
                for j in range(i + 1, len(inheritance_links)):
                    result = self._apply_local_deduction(
                        inheritance_links[i], 
                        inheritance_links[j]
                    )
                    if (result is not None and 
                        result.truth_value.confidence >= min_confidence):
                        new_inferences.append(result)
        
        return new_inferences
    
    def _apply_local_deduction(self, premise1: Atom, premise2: Atom) -> Optional[Atom]:
        """Apply deduction rule locally (without Ray actors)."""
        # Check if premises are inheritance links
        if (premise1.atom_type != AtomType.INHERITANCE_LINK or
            premise2.atom_type != AtomType.INHERITANCE_LINK):
            return None
        
        # Check if they can be chained (B in first equals A in second)
        if premise1.outgoing[1] != premise2.outgoing[0]:
            return None
        
        # Apply deduction formula
        new_tv = PLNFormula.deduction(premise1.truth_value, premise2.truth_value)
        
        # Create new inheritance link A->C
        conclusion = inheritance_link(
            premise1.outgoing[0],  # A
            premise2.outgoing[1],  # C  
            new_tv
        )
        
        return conclusion
    
    def backward_chain(self, 
                      target: Atom,
                      max_depth: int = 3) -> Optional[TruthValue]:
        """
        Perform backward chaining to find support for a target atom.
        
        Args:
            target: The atom to find support for
            max_depth: Maximum search depth
            
        Returns:
            Truth value for the target if support found, None otherwise
        """
        return self._backward_chain_recursive(target, max_depth, set())
    
    def _backward_chain_recursive(self, 
                                target: Atom,
                                depth: int,
                                visited: set) -> Optional[TruthValue]:
        """Recursive backward chaining implementation."""
        if depth <= 0 or target.handle in visited:
            return None
        
        visited.add(target.handle)
        
        # Check if target already exists in AtomSpace
        existing_atoms = []
        if isinstance(target, Link):
            # For links, look for exact match
            all_atoms = self.atomspace.get_atoms_by_type(target.atom_type)
            for atom in all_atoms:
                if isinstance(atom, Link) and atom.outgoing == target.outgoing:
                    existing_atoms.append(atom)
        else:
            # For nodes, look by name
            existing_atoms = self.atomspace.get_atoms_by_name(target.name)
        
        if existing_atoms:
            # Return the highest confidence truth value
            best_tv = max(existing_atoms, key=lambda a: a.truth_value.confidence).truth_value
            return best_tv
        
        # Try to infer the target through deduction
        if target.atom_type == AtomType.INHERITANCE_LINK and len(target.outgoing) == 2:
            child, parent = target.outgoing
            
            # Look for intermediate concepts B such that child->B and B->parent
            inheritance_links = self.atomspace.get_atoms_by_type(AtomType.INHERITANCE_LINK)
            
            for link1 in inheritance_links:
                if len(link1.outgoing) == 2 and link1.outgoing[0] == child:
                    intermediate = link1.outgoing[1]
                    
                    # Look for B->parent
                    intermediate_target = inheritance_link(intermediate, parent)
                    intermediate_tv = self._backward_chain_recursive(
                        intermediate_target, depth - 1, visited.copy()
                    )
                    
                    if intermediate_tv:
                        # Apply deduction
                        inferred_tv = PLNFormula.deduction(link1.truth_value, intermediate_tv)
                        return inferred_tv
        
        visited.remove(target.handle)
        return None
    
    def compute_conjunction(self, atoms: List[Atom]) -> TruthValue:
        """
        Compute the truth value of a conjunction of atoms.
        
        Args:
            atoms: List of atoms to combine
            
        Returns:
            Truth value of the conjunction
        """
        truth_values = []
        
        for atom in atoms:
            # Try to find or infer truth value for each atom
            existing = self.atomspace.get_atoms_by_name(atom.name) if hasattr(atom, 'name') else []
            
            if existing:
                truth_values.append(existing[0].truth_value)
            else:
                # Try backward chaining
                inferred_tv = self.backward_chain(atom)
                if inferred_tv:
                    truth_values.append(inferred_tv)
                else:
                    truth_values.append(TruthValue(0.5, 0.1))  # Default uncertain value
        
        return PLNFormula.conjunction(truth_values)
    
    def compute_disjunction(self, atoms: List[Atom]) -> TruthValue:
        """
        Compute the truth value of a disjunction of atoms.
        
        Args:
            atoms: List of atoms to combine
            
        Returns:
            Truth value of the disjunction
        """
        truth_values = []
        
        for atom in atoms:
            # Try to find or infer truth value for each atom
            existing = self.atomspace.get_atoms_by_name(atom.name) if hasattr(atom, 'name') else []
            
            if existing:
                truth_values.append(existing[0].truth_value)
            else:
                # Try backward chaining
                inferred_tv = self.backward_chain(atom)
                if inferred_tv:
                    truth_values.append(inferred_tv)
                else:
                    truth_values.append(TruthValue(0.5, 0.1))  # Default uncertain value
        
        return PLNFormula.disjunction(truth_values)