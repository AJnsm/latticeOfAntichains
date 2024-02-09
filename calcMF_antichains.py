from itertools import combinations
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig
import os

outdir = './antichain_outputs'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def vertices_in_between_adMat(adMat, start_vertex, end_vertex):
    '''
    Returns the nodes that appear in between the start_vertex and the end_vertex in the poset. 
    adMat_exp: The poset adjacency matrix
    '''
    
    vInBetween = []
    for vertex in range(adMat.shape[0]):
        if (adMat[start_vertex, vertex]>0) and (adMat[vertex, end_vertex]>0):
            vInBetween.append(vertex)
    return list(set(vInBetween) - set([start_vertex]) | set([end_vertex]))

# def mobius_function_from_adMat(adMat, i, j):
#     '''
#     Recursively calculates the Moebius function between any two indices i and j. 
#     adMat: The poset adjacency matrix
#     this recursive implementaion is very inefficient as it recalculates the same values many times.
#     '''
#     if i == j:
#         return 1
#     if adMat[i, j] == 0:
#         return 0
#     mu=0
#     for v in vertices_in_between_adMat(adMat, i, j):
#         mu += mobius_function_from_adMat(adMat, v, j)
#     return -mu

def reduceAdjacencyMatrix(A):
    '''
    Removes connections implied by transitivity, so that only direct connections remain. That is, if A>B>C and A>C, then only A>B>C is stored. 
    This is mainly useful for drawing the graph. 
    '''
    A_direct = A.copy()
    
    for i in range(2, A.shape[0]):
        A_direct -= np.linalg.matrix_power(A_direct, i)
        A_direct = A_direct.clip(min=0)
    
    return A_direct


for N in [2, 3, 4, 5]:
    print('starting calculation on N=', N)
    
    subsets = [list(subset) for k in range(1, N+1) for subset in combinations(range(N), k)]

    def check_anti_chain(comb):
        '''
        comb: a list of lists. Checks if comb is an antichain. 
        Returns comb if it's an antichain, otherwise returns the integer 0. 
        '''
        for source1 in comb:
            for source2 in comb:
                if source1 == source2:
                    continue
                if (set(source1).issubset(set(source2))) or (set(source1).issuperset(set(source2))):
                    return 0
        return comb


    # A maximum of n choose n/2 subsets can appear in an antichain
    maxSize = len(list(combinations(np.arange(N), np.floor(N/2).astype(int))))+1

    print('Determining antichains')
    # Check all possible combinations of subsets in parallel:
    results = Parallel(n_jobs=-1)(delayed(check_anti_chain)(comb)  for i in range(1, maxSize) for comb in [list(x) for x in combinations(subsets, i)])

    antiChains = []
    for i in range(len(results)):
        if results[i] != 0:
            antiChains.append(results[i])

    print('Number of antichains:', len(antiChains))
    print('(This should be 1, 4, 18, 166, 7579, etc. That is, the N-th Dedekind number minus two.)')
    pd.Series(antiChains).to_csv(f'antichain_outputs/antiChains_N={N}.csv')


    # A stores the adjacency matrix of the antichain lattice
    A = np.zeros((len(antiChains), len(antiChains)))

    # order: A < B if for every b in B, there is an a in A such that a subseteq b
    def add_edges(i, j, antiChains):
        if i == j:
            return 0
        if all([any([(set(source2).issubset(set(source1))) for source2 in antiChains[j]]) for source1 in antiChains[i]]):
            return (i, j)
        else:
            return 0

    print('Ordering antichains')
    # Order all antichains
    results = Parallel(n_jobs=-1)(delayed(add_edges)(i, j, antiChains) for i in range(len(antiChains)) for j in range(len(antiChains)) if i != j)

    results = [x for x in results if x != 0]

    for (i, j) in results:
        A[j, i] = 1
    pd.DataFrame(A).astype(int).to_csv(f'antichain_outputs/antiChainLattice_adjMat_N={N}.csv')

    # A more legible notation for antichains. 
    labs = np.array([''.join([str(s) for s in part]).replace(', ', '').replace('][', '|').replace(']', '').replace('[', '') for part in antiChains])

    print('Removing transitivity paths')
    # Then remove all indirect paths to plot the lattice
    A_direct = reduceAdjacencyMatrix(A)
    
    graph = ig.Graph.Adjacency((A_direct > 0).tolist(), mode="directed")
    if N <=3:
        ly = 'sugiyama'
    else:
        ly = 'fruchterman_reingold'

    print('plotting poset graph')
    boxSize = 400*2**(N-2)
    ig.plot(graph, layout=ly, bbox=(boxSize, boxSize), vertex_label=labs, 
            vertex_size=5, vertex_color='black', vertex_label_dist=1.5, 
            vertex_label_color='magenta', margin=50, edge_arrow_size=1).save(f'antichain_outputs/antichain_lattice_N={N}_layout={ly}.png')
    plt.show()
    
    pd.DataFrame(A_direct).astype(int).to_csv(f'antichain_outputs/antiChainLattice_adjMat_direct_N={N}.csv')

    
    def calcMFnsOfNode(node, antiChains, A, A_red):
        '''
        Calculate the vector of Möbius function evals mu(i, node) of a node in the lattice.
        An efficient way to calculate the vector mu(i, node) is to start at the top element mu(node, node), which is 1 by def.
        Then we take increasingly big steps on the reversed lattice to go down in the poset. At each iteration, we can simply sum all of the values of the Möbius functions above it, which are already calculated. 
        This bypasses the need for recursion, and is much faster because it reuses already calculated values.
        node: The index of the node in the lattice
        antiChains: The list of antichains
        A: The adjacency matrix of the lattice
        A_red: The reduced adjacency matrix with only direct connections
        '''
        v = np.zeros(len(antiChains))
        v[node] = 1

        # The adjacency matrix for paths of a given length is A^length.
        AdjAtCurrentDist = A_red.copy().T
        for distance in range(len(antiChains)):
            # For all nodes that are reachable by paths with the current distance:
            nextNodes = np.where(AdjAtCurrentDist[node, :] > 0)[0]
            for nextNode in nextNodes:
                # Sum the value of all the Möbius function 'above' it. That is, the ancestors on the reversed lattice. 
                v[nextNode] = -sum([v[ancestor] for ancestor in np.where(A.T[:, nextNode] > 0)[0]])

            # Increase the distance by one
            AdjAtCurrentDist = AdjAtCurrentDist@A_red.T

        return v


    print('Calculating Moebius functions')
    MFmat = np.zeros((len(antiChains), len(antiChains)))

    results = Parallel(n_jobs=-1)(delayed(calcMFnsOfNode)(node, antiChains, A, A_direct) for node in range(len(antiChains)))

    for i, v in enumerate(results):
        MFmat[:, i] = v
    MFmat = MFmat.astype(int)
    
    print('Saving Moebius functions')
    # If the lattice is small (N<=4), we can save the full matrix. Otherwise, we save only the non-zero values.
    MFdf = pd.DataFrame(MFmat, columns=labs, index=labs)
    if N<=4:
        MFdf.to_csv(f'antichain_outputs/antiChainLattice_mobiusFns_N={N}.csv')
    else:
        nonzeroMFs = pd.DataFrame(MFdf[MFdf!=0].stack().reset_index().to_records(index=False))
        nonzeroMFs.columns = ['a', 'b', 'mu(a, b)']
        nonzeroMFs.to_csv(f'antichain_outputs/antiChainLattice_mobiusFns_N={N}.csv')
