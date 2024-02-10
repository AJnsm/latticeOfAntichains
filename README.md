# latticeOfAntichains

A Python script to calculate the Möbius function on the lattice of antichains of the poset of subsets ordered by inclusion. 

These values can make solving the equations of the Partial Information Decomposition much easier and faster. 

The directory `antichain_outputs` contains the output of the script for up to four variables. It contains the following files:

- `antiChains_N={N}.csv`: a list of the antichains
- `antiChainLattice_asjMat_N={N}.csv`: the adjacency matrix of the lattice
- `antiChainLattice_asjMat_N={N}.csv`: the transitive reduction of this lattice
- `antiChainLattice_mobiusFn_N={N}.csv`: the Möbius function on every pair of elements. 
- `antichain_lattice_N={N}_layout={layout}.png`: the Hasse diagram of the lattice (the graph associated to its transitive reduction).


For N=5, the output currently only contains the list of antichains, as it's computationally very expensive to run the full script for N=5.