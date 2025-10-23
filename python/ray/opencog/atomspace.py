"""
AtomSpace implementation using Ray for distributed knowledge representation.

The AtomSpace is the central database that stores all atoms (knowledge) in OpenCog.
This implementation uses Ray actors to distribute the AtomSpace across multiple nodes.
"""

from typing import List, Optional, Dict, Set, Any, Union, Callable
from collections import defaultdict
from .atoms import Atom, Node, Link, TruthValue
from .atom_types import AtomType, TYPE_HIERARCHY

# Ray imports are optional - only used for distributed functionality
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class AtomSpace:
    """
    Local AtomSpace for storing and managing atoms.
    
    This is a non-distributed version that can be used locally or as part of
    the distributed AtomSpace implementation.
    """
    
    def __init__(self):
        """Initialize an empty AtomSpace."""
        self._atoms: Dict[str, Atom] = {}  # handle -> atom
        self._atoms_by_type: Dict[AtomType, Set[str]] = defaultdict(set)  # type -> handles  
        self._atoms_by_name: Dict[str, Set[str]] = defaultdict(set)  # name -> handles
        self._incoming_index: Dict[str, Set[str]] = defaultdict(set)  # handle -> incoming handles
    
    def add_atom(self, atom: Atom) -> Atom:
        """
        Add an atom to the AtomSpace.
        
        Args:
            atom: The atom to add
            
        Returns:
            The atom that was added (may be existing atom if duplicate)
        """
        # Check if atom already exists
        existing = self._find_existing_atom(atom)
        if existing:
            return existing
            
        # Add the new atom
        self._atoms[atom.handle] = atom
        self._atoms_by_type[atom.atom_type].add(atom.handle)
        
        if atom.name:
            self._atoms_by_name[atom.name].add(atom.handle)
            
        # Update incoming index for links
        if isinstance(atom, Link):
            for target_atom in atom.outgoing:
                self._incoming_index[target_atom.handle].add(atom.handle)
                
        return atom
    
    def _find_existing_atom(self, atom: Atom) -> Optional[Atom]:
        """Find an existing atom that matches the given atom."""
        # For nodes, check by type and name
        if isinstance(atom, Node):
            for handle in self._atoms_by_name.get(atom.name, set()):
                existing = self._atoms[handle]
                if existing.atom_type == atom.atom_type:
                    return existing
        
        # For links, check by type and outgoing atoms
        elif isinstance(atom, Link):
            for handle in self._atoms_by_type.get(atom.atom_type, set()):
                existing = self._atoms[handle]
                if (isinstance(existing, Link) and 
                    existing.outgoing == atom.outgoing):
                    return existing
                    
        return None
    
    def get_atom(self, handle: str) -> Optional[Atom]:
        """Get an atom by its handle."""
        return self._atoms.get(handle)
    
    def remove_atom(self, atom: Union[Atom, str]) -> bool:
        """
        Remove an atom from the AtomSpace.
        
        Args:
            atom: The atom or its handle to remove
            
        Returns:
            True if the atom was removed, False if not found
        """
        handle = atom.handle if isinstance(atom, Atom) else atom
        
        if handle not in self._atoms:
            return False
            
        atom_obj = self._atoms[handle]
        
        # Remove from indices
        del self._atoms[handle]
        self._atoms_by_type[atom_obj.atom_type].discard(handle)
        
        if atom_obj.name:
            self._atoms_by_name[atom_obj.name].discard(handle)
            
        # Remove from incoming index
        if isinstance(atom_obj, Link):
            for target_atom in atom_obj.outgoing:
                self._incoming_index[target_atom.handle].discard(handle)
                
        # Remove from incoming sets of target atoms
        for incoming_handle in self._incoming_index[handle]:
            incoming_atom = self._atoms.get(incoming_handle)
            if incoming_atom:
                incoming_atom.incoming_set = [
                    a for a in incoming_atom.incoming_set if a.handle != handle
                ]
                
        del self._incoming_index[handle]
        
        return True
    
    def get_atoms_by_type(self, atom_type: AtomType, include_subtypes: bool = True) -> List[Atom]:
        """
        Get all atoms of a specific type.
        
        Args:
            atom_type: The type to search for
            include_subtypes: Whether to include subtypes
            
        Returns:
            List of matching atoms
        """
        handles = set()
        
        if include_subtypes:
            # Include all atoms whose type is a subtype of the requested type
            for atype, type_handles in self._atoms_by_type.items():
                if atype == atom_type or TYPE_HIERARCHY.is_subtype(atype, atom_type):
                    handles.update(type_handles)
        else:
            handles = self._atoms_by_type.get(atom_type, set())
            
        return [self._atoms[handle] for handle in handles]
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name."""
        handles = self._atoms_by_name.get(name, set())
        return [self._atoms[handle] for handle in handles]
    
    def get_incoming_set(self, atom: Union[Atom, str]) -> List[Atom]:
        """Get all atoms that have the given atom in their outgoing set."""
        handle = atom.handle if isinstance(atom, Atom) else atom
        incoming_handles = self._incoming_index.get(handle, set())
        return [self._atoms[h] for h in incoming_handles if h in self._atoms]
    
    def size(self) -> int:
        """Get the number of atoms in the AtomSpace."""
        return len(self._atoms)
    
    def clear(self) -> None:
        """Remove all atoms from the AtomSpace."""
        self._atoms.clear()
        self._atoms_by_type.clear()
        self._atoms_by_name.clear()
        self._incoming_index.clear()


if RAY_AVAILABLE:
    @ray.remote
    class DistributedAtomSpaceActor:
        """
        Ray actor that manages a partition of the distributed AtomSpace.
        
        Each actor maintains a local AtomSpace and handles requests for its partition.
        """
        
        def __init__(self, partition_id: int):
            """Initialize the distributed AtomSpace actor."""
            self.partition_id = partition_id
            self.atomspace = AtomSpace()
        
        def add_atom(self, atom: Atom) -> Atom:
            """Add an atom to this partition."""
            return self.atomspace.add_atom(atom)
        
        def get_atom(self, handle: str) -> Optional[Atom]:
            """Get an atom by handle."""
            return self.atomspace.get_atom(handle)
        
        def remove_atom(self, handle: str) -> bool:
            """Remove an atom by handle."""
            return self.atomspace.remove_atom(handle)
        
        def get_atoms_by_type(self, atom_type: AtomType, include_subtypes: bool = True) -> List[Atom]:
            """Get atoms by type in this partition."""
            return self.atomspace.get_atoms_by_type(atom_type, include_subtypes)
        
        def get_atoms_by_name(self, name: str) -> List[Atom]:
            """Get atoms by name in this partition."""
            return self.atomspace.get_atoms_by_name(name)
        
        def get_incoming_set(self, handle: str) -> List[Atom]:
            """Get incoming set for an atom."""
            return self.atomspace.get_incoming_set(handle)
        
        def size(self) -> int:
            """Get the number of atoms in this partition."""
            return self.atomspace.size()
        
        def clear(self) -> None:
            """Clear this partition."""
            self.atomspace.clear()
else:
    # Create a placeholder when Ray is not available
    class DistributedAtomSpaceActor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Ray is not available. Cannot use DistributedAtomSpaceActor.")


class DistributedAtomSpace:
    """
    Distributed AtomSpace using Ray actors.
    
    This distributes atoms across multiple Ray actors for scalability and parallel processing.
    """
    
    def __init__(self, num_partitions: int = 4):
        """
        Initialize the distributed AtomSpace.
        
        Args:
            num_partitions: Number of partitions to create
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not available. Cannot use DistributedAtomSpace.")
            
        self.num_partitions = num_partitions
        self.partitions = [
            DistributedAtomSpaceActor.remote(i) 
            for i in range(num_partitions)
        ]
    
    def _get_partition(self, atom: Atom) -> 'DistributedAtomSpaceActor':
        """Get the partition for a given atom based on hash."""
        partition_index = hash(atom) % self.num_partitions
        return self.partitions[partition_index]
    
    def _get_partition_by_handle(self, handle: str) -> 'DistributedAtomSpaceActor':
        """Get the partition for a given handle."""
        partition_index = hash(handle) % self.num_partitions
        return self.partitions[partition_index]
    
    def add_atom(self, atom: Atom) -> Atom:
        """
        Add an atom to the distributed AtomSpace.
        
        Args:
            atom: The atom to add
            
        Returns:
            The added atom
        """
        partition = self._get_partition(atom)
        return ray.get(partition.add_atom.remote(atom))
    
    def get_atom(self, handle: str) -> Optional[Atom]:
        """Get an atom by its handle."""
        partition = self._get_partition_by_handle(handle)
        return ray.get(partition.get_atom.remote(handle))
    
    def remove_atom(self, atom: Union[Atom, str]) -> bool:
        """Remove an atom from the distributed AtomSpace."""
        handle = atom.handle if isinstance(atom, Atom) else atom
        partition = self._get_partition_by_handle(handle)
        return ray.get(partition.remove_atom.remote(handle))
    
    def get_atoms_by_type(self, atom_type: AtomType, include_subtypes: bool = True) -> List[Atom]:
        """
        Get all atoms of a specific type across all partitions.
        
        Args:
            atom_type: The type to search for
            include_subtypes: Whether to include subtypes
            
        Returns:
            List of matching atoms from all partitions
        """
        # Query all partitions in parallel
        futures = [
            partition.get_atoms_by_type.remote(atom_type, include_subtypes)
            for partition in self.partitions
        ]
        
        # Collect results from all partitions
        results = ray.get(futures)
        all_atoms = []
        for partition_atoms in results:
            all_atoms.extend(partition_atoms)
            
        return all_atoms
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name across all partitions."""
        futures = [
            partition.get_atoms_by_name.remote(name)
            for partition in self.partitions
        ]
        
        results = ray.get(futures)
        all_atoms = []
        for partition_atoms in results:
            all_atoms.extend(partition_atoms)
            
        return all_atoms
    
    def get_incoming_set(self, atom: Union[Atom, str]) -> List[Atom]:
        """Get incoming set for an atom across all partitions."""
        handle = atom.handle if isinstance(atom, Atom) else atom
        
        # Query all partitions since incoming links could be anywhere
        futures = [
            partition.get_incoming_set.remote(handle)
            for partition in self.partitions
        ]
        
        results = ray.get(futures)
        all_atoms = []
        for partition_atoms in results:
            all_atoms.extend(partition_atoms)
            
        return all_atoms
    
    def size(self) -> int:
        """Get the total number of atoms across all partitions."""
        futures = [partition.size.remote() for partition in self.partitions]
        sizes = ray.get(futures)
        return sum(sizes)
    
    def clear(self) -> None:
        """Clear all partitions."""
        futures = [partition.clear.remote() for partition in self.partitions]
        ray.get(futures)  # Wait for all clears to complete