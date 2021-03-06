# The code below was taken from a Jupyter Notebook file on a public GitHub repo, with the publisher called
# Arseny Khakhalin (username; khakhalin)
# https://github.com/khakhalin/Sketches/blob/master/classic/generate_all_graphs.ipynb
# (last accessed 2021-10-20)

def make_graphs(n=2, i=None, j=None):
    """Make a graph recursively, by either including, or skipping each edge.
    Edges are given in lexicographical order by construction."""
    out = []
    if i is None:  # First call
        out = [[(0, 1)]+r for r in make_graphs(n=n, i=0, j=1)]
    elif j < n-1:
        out += [[(i, j+1)]+r for r in make_graphs(n=n, i=i, j=j+1)]
        out += [r for r in make_graphs(n=n, i=i, j=j+1)]
    elif i < n-1:
        out = make_graphs(n=n, i=i+1, j=i+1)
    else:
        out = [[]]
    return out


def filter(gs, target_nv):
    """Filter all improper graphs: those with not enough nodes, 
    those not fully connected, and those isomorphic to previously considered."""
    mem = set({})
    gs2 = []
    for g in gs:
        nv = len(set([i for e in g for i in e]))
        if nv != target_nv:
            continue
        if not connected(g):
            continue
        if tuple(g) not in mem:
            gs2.append(g)
            mem |= set(permute(g, target_nv))
        #print('\n'.join([str(a) for a in mem]))
    return gs2


def connected(g):
    """Check if the graph is fully connected, with Union-Find."""
    nodes = set([i for e in g for i in e])
    roots = {node: node for node in nodes}

    def _root(node, depth=0):
        if node == roots[node]:
            return (node, depth)
        else:
            return _root(roots[node], depth+1)

    for i, j in g:
        ri, di = _root(i)
        rj, dj = _root(j)
        if ri == rj:
            continue
        if di <= dj:
            roots[ri] = rj
        else:
            roots[rj] = ri
    return len(set([_root(node)[0] for node in nodes])) == 1


def permute(g, n):
    """Create a set of all possible isomorphic codes for a graph, 
    as nice hashable tuples. All edges are i<j, and sorted lexicographically."""
    ps = perm(n)
    out = set([])
    for p in ps:
        out.add(
            tuple(sorted([(p[i], p[j]) if p[i] < p[j] else (p[j], p[i]) for i, j in g])))
    return list(out)


def perm(n, s=None):
    """All permutations of n elements."""
    if s is None:
        return perm(n, tuple(range(n)))
    if not s:
        return [[]]
    return [[i]+p for i in s for p in perm(n, tuple([k for k in s if k != i]))]

# END Copied Code
