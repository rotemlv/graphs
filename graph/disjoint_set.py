def make_set_old(x, forest: set, parent: dict, size: dict, rank: dict):
    # safe-lower
    if x not in forest:
        parent[x] = x
        size[x] = 1
        rank[x] = 0


def make_set_ready_for_size(x, parent: dict, size: dict, rank: dict):
    # no forest check, don't call this with duplicate elements
    parent[x] = x  # as if x is a one-element tree
    size[x] = 1
    rank[x] = 0


def make_set(x, parent: dict, rank: dict):
    # No size AND no forest check (tight fit for current version)
    parent[x] = x  # as if x is a one-element tree
    rank[x] = 0  # tree is not empty - height of a single node is 0


def find(x, parent: dict):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # correct the parent (hook such that (x->a->b->c) --> x->(a->c))
        x = parent[x]  # correct x (both on the "fly" (mosquito)) (x->(a->c)) --> (x->c)
    # run some tests
    return x


def union(x, y, parent: dict, rank: dict):
    x = find(x, parent)  # which tree x is in
    y = find(y, parent)  # ditto with y
    if x == y:
        return
    if rank[x] < rank[y]:
        # rank is the height of the tree - move elements to **bigger** tree
        x, y = y, x
    parent[y] = x  # set y's parent tree to be x
    if rank[x] == rank[y]:
        rank[x] += 1