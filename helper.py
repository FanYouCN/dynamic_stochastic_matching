import itertools
import networkx as nx
from networkx import from_numpy_matrix, is_connected
from networkx.algorithms.bipartite.matching import maximum_matching
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import build_residual_network
import numpy as np
import copy

'''
    helper functions for:
        blossom separation using gomory_hu cut tree
        probablisitc allocation policy using birkhoff von neumann decomposition
        and others
'''

TOLERANCE = np.finfo(float).eps * 10.
default_flow_func = edmonds_karp

def simple_cycles(GG, limit):
    if not nx.is_directed(GG):
        G = nx.DiGraph()
        for i in GG:
            G.add_node(i)
        for e in GG.edges:
            G.add_edge(e[0],e[1])
            G.add_edge(e[1],e[0])
    else:
        G = copy.deepcopy(GG)
    limit += 1
    subG = type(G)(G.edges())
    sccs = list(nx.strongly_connected_components(subG))
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]
        while stack:
            thisnode, nbrs = stack[-1]

            if nbrs and len(path) < limit:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= limit:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()
        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))
    return sccs

def to_permutation_matrix(matches):
    """Converts a permutation into a permutation matrix.
    `matches` is a dictionary whose keys are vertices and whose values are
    partners. For each vertex ``u`` and ``v``, entry (``u``, ``v``) in the
    returned matrix will be a ``1`` if and only if ``matches[u] == v``.
    Pre-condition: `matches` must be a permutation on an initial subset of the
    natural numbers.
    Returns a permutation matrix as a square NumPy array.
    """
    n = len(matches)
    P = np.zeros((n, n))
    for (u, v) in matches.items():
        P[u, v] = 1
    return P


def zeros(m, n):
    """Convenience function for ``numpy.zeros((m, n))``."""
    return np.zeros((m, n))


def hstack(left, right):
    """Convenience function for ``numpy.hstack((left, right))``."""
    return np.hstack((left, right))


def vstack(top, bottom):
    """Convenience function for ``numpy.vstack((top, bottom))``."""
    return np.vstack((top, bottom))


def four_blocks(topleft, topright, bottomleft, bottomright):
    """Convenience function that creates a block matrix with the specified
    blocks.
    Each argument must be a NumPy matrix. The two top matrices must have the
    same number of rows, as must the two bottom matrices. The two left matrices
    must have the same number of columns, as must the two right matrices.
    """
    return vstack(hstack(topleft, topright),
                  hstack(bottomleft, bottomright))


def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.
    `A` must be a NumPy array.
    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.
    """
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))


def to_pattern_matrix(D):
    """Returns the Boolean matrix in the same shape as `D` with ones exactly
    where there are nonzero entries in `D`.
    `D` must be a NumPy array.
    """
    result = np.zeros_like(D)
    result[D.nonzero()] = 1
    return result


def birkhoff_von_neumann_decomposition(D):
    """Returns the Birkhoff--von Neumann decomposition of the doubly
    stochastic matrix `D`.
    """
    m, n = D.shape
    if m != n:
        raise ValueError('Input matrix must be square ({} x {})'.format(m, n))
    indices = list(itertools.product(range(m), range(n)))
    # These two lists will store the result as we build it up each iteration.
    coefficients = []
    permutations = []
    # Create a copy of D so that we don't modify it directly. Cast the
    # entries of the matrix to floating point numbers, regardless of
    # whether they were integers.
    S = D.astype('float')
    while not np.all(abs(S) < 1e-12):
        # Create an undirected graph whose adjacency matrix contains a 1
        # exactly where the matrix S has a nonzero entry.
        W = to_pattern_matrix(S)
        # Construct the bipartite graph whose left and right vertices both
        # represent the vertex set of the pattern graph (whose adjacency matrix
        # is ``W``).
        X = to_bipartite_matrix(W)
        # Convert the matrix of a bipartite graph into a NetworkX graph object.
        G = from_numpy_matrix(X)
        # Compute a perfect matching for this graph. The dictionary `M` has one
        # entry for each matched vertex (in both the left and the right vertex
        # sets), and the corresponding value is its partner.
        if not is_connected(G):
            top_nodes, bottom_nodes = [], []
            for e in G.edges:
                if e[0] in top_nodes:
                    bottom_nodes.append(e[1])
                elif e[0] in bottom_nodes:
                    top_nodes.append(e[1])
                elif e[1] in top_nodes:
                    bottom_nodes.append(e[0])
                elif e[1] in bottom_nodes:
                    top_nodes.append(e[0])
                else:
                    top_nodes.append(e[0])
                    bottom_nodes.append(e[1])
            M = maximum_matching(G, top_nodes)
        else:
            M = maximum_matching(G)
        # However, since we have both a left vertex set and a right vertex set,
        # each representing the original vertex set of the pattern graph
        # (``W``), we need to convert any vertex greater than ``n`` to its
        # original vertex number. To do this,
        #
        #   - ignore any keys greater than ``n``, since they are already
        #     covered by earlier key/value pairs,
        #   - ensure that all values are less than ``n``.
        #
        M = {u: v % n for u, v in M.items() if u < n}
        # Convert that perfect matching to a permutation matrix.
        P = to_permutation_matrix(M)
        # Get the smallest entry of S corresponding to the 1 entries in the
        # permutation matrix.
        q = min(S[i, j] for (i, j) in indices if P[i, j] == 1)
        # Store the coefficient and the permutation matrix for later.
        coefficients.append(q)
        permutations.append(P)
        # Subtract P scaled by q. After this subtraction, S has a zero entry
        # where the value q used to live.
        S -= q * P
        # PRECISION ISSUE: There seems to be a problem with floating point
        # precision here, so we need to round down to 0 any entry that is very
        # small.
        S[np.abs(S) < TOLERANCE] = 0.0
    return list(zip(coefficients, permutations))


def gomory_hu_tree(G, capacity='capacity', flow_func = None):
    '''
        Gusfield implementation
    '''
    if flow_func is None:
        flow_func = default_flow_func

    if len(G) == 0:  # empty graph
        msg = 'Empty Graph does not have a Gomory-Hu tree representation'
        raise nx.NetworkXError(msg)

    R = build_residual_network(G, capacity)
    p = {}
    fl = {}
    nodes = list(G.nodes)
    root = nodes[0]
    for i in nodes:
        p[i] = root

    for s in nodes[1:]:
        t = p[s]
        cut_value, partition = nx.minimum_cut(G, s, t, capacity=capacity, flow_func=flow_func, residual=R)
        X, Y = partition
        fl[s] = cut_value
        for i in nodes:
            if i != s and i in X and p[i] == t:
                p[i] = s
        if p[t] in X:
            p[s] = p[t]
            p[t] = s
            fl[s] = fl[t]
            fl[t] = cut_value
    T = nx.Graph()
    T.add_nodes_from(G)
    T.add_weighted_edges_from(((u, p[u], fl[u]) for u in nodes[1:]))
    return T


def min_odd_cut(T, V_odd):
    moc = max(nx.get_edge_attributes(T,'weight').values())
    W = 0
    for e in T.edges():
        this_T = T.copy()
        this_T.remove_edge(*e)
        V1 = nx.node_connected_component(this_T,e[0])
        V2 = nx.node_connected_component(this_T,e[1])
        n_odd_node = 0
        for v in V_odd:
            n_odd_node += (1 if v in V1 else 0)
        if n_odd_node % 2 == 1:
            this_cut = T.get_edge_data(*e)
            if this_cut['weight'] < moc:
                moc = this_cut['weight']
                if 'v' in V1:
                    W = V2
                else:
                    W = V1
    return moc, W


def blossom_separation(Graph, b, x):
    '''
    1. add dummy slack vertex check
    2. add dummy edges with appropriate weights, x is a dict check
    3. lable all vertices, denote T the odd vertices set
    4. build gomory_hu_tree for the resulting graph with T as the terminal set
    5. check the |T|-1 edges of the cut-tree
    '''
    G = Graph.copy()
    G.add_node('v')
    nx.set_node_attributes(G, b, 'capacity')
    terminal_set = set()

    for i in G:
        if i!='v':
            G.add_edge(i, 'v')
            if b[i] % 2 == 1.0:
                terminal_set.add(i)

    if sum(b.values()) % 2 == 1.0:
        terminal_set.add('v')
    for e in G.edges('v'):
        i = e[0] if e[1] == 'v' else e[1]
        x[frozenset(e)] = b[i]
        for ee in Graph.edges(i):
            x[frozenset(e)] -= x[frozenset(ee)]

    nx.set_edge_attributes(G, x, 'weight')

    gh_tree = gomory_hu_tree(G, capacity='weight')
    moc, W = min_odd_cut(gh_tree, terminal_set)
    if moc < 0.99:
        return W
    else:
        return None


def are_blood_compatible(donor_blood, recipient_blood):
    """ Returns 1 if donor and recipient are ABO compatible, 0 otherwise. """

    if(donor_blood == 'O'):
        return True
    if(donor_blood == recipient_blood):
        return True
    if(recipient_blood == 'AB'):
        return True
    return False