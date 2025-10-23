"""
Pattern matching implementation for distributed AtomSpace.

This module provides pattern matching capabilities that can work across
distributed AtomSpace partitions using Ray for parallel processing.
"""

from typing import List, Dict, Set, Optional, Any, Callable, Union
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
from .atoms import Atom, Node, Link, TruthValue, variable_node
from .atom_types import AtomType
from .atomspace import AtomSpace, DistributedAtomSpace


class Pattern:
    """
    Represents a pattern for matching atoms in the AtomSpace.
    
    Patterns can contain variables that will be bound during matching.
    """
    
    def __init__(self, template: Atom, variables: Optional[List[Atom]] = None):
        """
        Initialize a pattern.
        
        Args:
            template: The pattern template containing variables
            variables: List of variable atoms in the pattern
        """
        self.template = template
        self.variables = variables or self._extract_variables(template)
    
    def _extract_variables(self, atom: Atom) -> List[Atom]:
        """Extract all variable nodes from an atom tree."""
        variables = []
        
        if atom.atom_type == AtomType.VARIABLE_NODE:
            variables.append(atom)
        elif isinstance(atom, Link):
            for outgoing in atom.outgoing:
                variables.extend(self._extract_variables(outgoing))
                
        return variables


class PatternMatch:
    """
    Represents a successful pattern match with variable bindings.
    """
    
    def __init__(self, bindings: Dict[str, Atom]):
        """
        Initialize a pattern match.
        
        Args:
            bindings: Dictionary mapping variable names to bound atoms
        """
        self.bindings = bindings
    
    def get_binding(self, variable_name: str) -> Optional[Atom]:
        """Get the binding for a variable."""
        return self.bindings.get(variable_name)
    
    def __str__(self) -> str:
        return f"PatternMatch({self.bindings})"


class PatternMatcher:
    """
    Pattern matcher for finding atoms that match given patterns.
    
    Supports both local and distributed pattern matching.
    """
    
    def __init__(self, atomspace: Union[AtomSpace, DistributedAtomSpace]):
        """
        Initialize the pattern matcher.
        
        Args:
            atomspace: The AtomSpace to search in
        """
        self.atomspace = atomspace
    
    def match(self, pattern: Pattern) -> List[PatternMatch]:
        """
        Find all matches for a pattern in the AtomSpace.
        
        Args:
            pattern: The pattern to match
            
        Returns:
            List of pattern matches with variable bindings
        """
        if isinstance(self.atomspace, DistributedAtomSpace):
            return self._distributed_match(pattern)
        else:
            return self._local_match(pattern)
    
    def _local_match(self, pattern: Pattern) -> List[PatternMatch]:
        """Perform pattern matching on a local AtomSpace."""
        matches = []
        
        # Get candidate atoms based on the root pattern type
        candidates = self._get_candidates(pattern.template)
        
        for candidate in candidates:
            bindings = {}
            if self._match_atom(pattern.template, candidate, bindings):
                matches.append(PatternMatch(bindings))
                
        return matches
    
    def _distributed_match(self, pattern: Pattern) -> List[PatternMatch]:
        """Perform distributed pattern matching across partitions."""
        if not RAY_AVAILABLE:
            # Fall back to local matching
            return self._local_match(pattern)
            
        # Create remote pattern matching tasks for each partition
        futures = []
        for partition in self.atomspace.partitions:
            future = self._match_in_partition(partition, pattern)
            futures.append(future)
        
        # Collect results from all partitions
        if futures:
            results = ray.get(futures)
            all_matches = []
            for partition_matches in results:
                all_matches.extend(partition_matches)
            return all_matches
        else:
            return []
    
    def _match_in_partition(self, partition_actor, pattern: Pattern) -> List[PatternMatch]:
        """Match a pattern in a single partition."""
        # This would be executed on the partition actor
        # For now, we'll implement a simplified version
        return []
    
    def _get_candidates(self, template: Atom) -> List[Atom]:
        """Get candidate atoms for matching based on the template type."""
        if isinstance(template, Node) and template.atom_type != AtomType.VARIABLE_NODE:
            # For concrete nodes, look for atoms with the same type and name
            return self.atomspace.get_atoms_by_name(template.name)
        elif isinstance(template, Link):
            # For links, get all atoms of the same type
            return self.atomspace.get_atoms_by_type(template.atom_type, include_subtypes=False)
        else:
            # For variable nodes or other cases, get all atoms of the same type
            return self.atomspace.get_atoms_by_type(template.atom_type, include_subtypes=True)
    
    def _match_atom(self, pattern: Atom, candidate: Atom, bindings: Dict[str, Atom]) -> bool:
        """
        Check if a candidate atom matches a pattern atom.
        
        Args:
            pattern: The pattern atom (may contain variables)
            candidate: The candidate atom to match
            bindings: Current variable bindings
            
        Returns:
            True if the candidate matches the pattern
        """
        # Handle variable nodes
        if pattern.atom_type == AtomType.VARIABLE_NODE:
            var_name = pattern.name
            if var_name in bindings:
                # Variable already bound, check consistency
                return bindings[var_name] == candidate
            else:
                # Bind the variable
                bindings[var_name] = candidate
                return True
        
        # Handle concrete nodes  
        if isinstance(pattern, Node):
            return (isinstance(candidate, Node) and 
                    pattern.atom_type == candidate.atom_type and
                    pattern.name == candidate.name)
        
        # Handle links
        if isinstance(pattern, Link):
            if not isinstance(candidate, Link):
                return False
            
            if pattern.atom_type != candidate.atom_type:
                return False
                
            if len(pattern.outgoing) != len(candidate.outgoing):
                return False
            
            # Recursively match outgoing atoms
            for pat_out, cand_out in zip(pattern.outgoing, candidate.outgoing):
                if not self._match_atom(pat_out, cand_out, bindings):
                    return False
                    
            return True
        
        return False


class BindLink:
    """
    Represents a bind link for executing pattern-based queries.
    
    A bind link consists of a pattern and a rewrite rule.
    """
    
    def __init__(self, variables: List[Atom], pattern: Atom, rewrite: Atom):
        """
        Initialize a bind link.
        
        Args:
            variables: List of variables in the pattern
            pattern: The pattern to match
            rewrite: The rewrite rule to apply to matches
        """
        self.variables = variables
        self.pattern = pattern
        self.rewrite = rewrite
    
    def execute(self, atomspace: Union[AtomSpace, DistributedAtomSpace]) -> List[Atom]:
        """
        Execute the bind link on an AtomSpace.
        
        Args:
            atomspace: The AtomSpace to execute on
            
        Returns:
            List of atoms created by applying the rewrite rule
        """
        matcher = PatternMatcher(atomspace)
        pattern_obj = Pattern(self.pattern, self.variables)
        matches = matcher.match(pattern_obj)
        
        results = []
        for match in matches:
            # Apply the rewrite rule with the bindings
            rewritten = self._apply_rewrite(self.rewrite, match.bindings)
            if rewritten:
                results.append(rewritten)
                
        return results
    
    def _apply_rewrite(self, template: Atom, bindings: Dict[str, Atom]) -> Optional[Atom]:
        """
        Apply variable bindings to a rewrite template.
        
        Args:
            template: The rewrite template
            bindings: Variable bindings from pattern matching
            
        Returns:
            The rewritten atom with variables substituted
        """
        if template.atom_type == AtomType.VARIABLE_NODE:
            return bindings.get(template.name)
        
        if isinstance(template, Link):
            rewritten_outgoing = []
            for outgoing in template.outgoing:
                rewritten = self._apply_rewrite(outgoing, bindings)
                if rewritten is None:
                    return None
                rewritten_outgoing.append(rewritten)
            
            # Create new link with rewritten outgoing atoms
            return Link(template.atom_type, rewritten_outgoing, template.truth_value)
        
        # Return the template as-is for concrete nodes
        return template


# Convenience functions for common patterns

def create_simple_query(predicate_name: str, arg_variable: str) -> Pattern:
    """
    Create a simple query pattern for evaluations.
    
    Args:
        predicate_name: Name of the predicate
        arg_variable: Name of the argument variable
        
    Returns:
        Pattern for matching evaluations of the predicate
    """
    from ray.opencog.atoms import predicate_node, evaluation_link
    
    pred = predicate_node(predicate_name)
    var = variable_node(arg_variable)
    template = evaluation_link(pred, [var])
    
    return Pattern(template, [var])


def create_inheritance_query(child_variable: str, parent_name: str) -> Pattern:
    """
    Create a pattern for finding inheritance relationships.
    
    Args:
        child_variable: Name of the child variable
        parent_name: Name of the parent concept
        
    Returns:
        Pattern for matching inheritance links
    """
    from ray.opencog.atoms import concept_node, inheritance_link
    
    child_var = variable_node(child_variable)
    parent = concept_node(parent_name)
    template = inheritance_link(child_var, parent)
    
    return Pattern(template, [child_var])