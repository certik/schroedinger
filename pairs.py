def make_pairs(s, rs, d1, d2, method="robust"):
    """
    Pairs the eigenvectors in s and rs using the distances d1 and d2.

    s, rs ... a list of items of any kind, they will get passed to the d1 and
                d2 functions
    d1(x, y) ... |x - y|
    d2(x, y) ... |x + y|

    The eigenvectors are determined up to a sign.

    The best solution is the one where all the "d" for all pairs is the lowest
    possible and ideally the pairs are "not so much shuffled".

    Where d = min(d1, d2).

    Both d1() and d2() are expensive operations.

    It returns (pairs, flips) where pairs is a list of indices into the "rs"
    array and flips is a list of True/False showing if the eigenvalue is
    flipped or not. See the test() function in this file for more information.
    """

    if method=="robust":
        return make_pairs_robust(s, rs, d1, d2)
    raise NotImplementedError()

def make_pairs_robust(s, rs, d1, d2):
    """
    Calculates all possible norms and chooses always the best.
    """
    if len(s) != len(rs):
        raise Exception("The length of both 's' and 'rs' must be the same.")
    rs_orig = rs[:]
    r = []
    flips = []
    for x in s:
        min_d = None
        for i, y in enumerate(rs):
            _d1 = d1(x, y)
            _d2 = d2(x, y)
            d = min(_d1, _d2)
            if min_d is None:
                min_d = d
                min_i = i
                min_flip = _d1 > _d2
            elif d < min_d:
                min_d = d
                min_i = i
                min_flip = _d1 > _d2
        assert min_d is not None
        r.append(rs_orig.index(rs[min_i]))
        del rs[min_i]
        flips.append(min_flip)
    rs[:] = rs_orig[:]
    return r, flips

def test():
    def d1(x, y):
        return abs(x-y)
    def d2(x, y):
        return abs(x+y)
    s = [1, 2, 3, 4]
    rs = [1, 4, 2, 3]
    assert make_pairs(s, rs, d1, d2) == ([0, 2, 3, 1],
        [False, False, False, False])
    s = [1, 2, 3, 4]
    rs = [-1, 4, 2, 3]
    assert make_pairs(s, rs, d1, d2) == ([0, 2, 3, 1],
        [True, False, False, False])
    s = [1, 2, 3, 4]
    rs = [3, 4, 2, -1]
    assert make_pairs(s, rs, d1, d2) == ([3, 2, 0, 1],
        [True, False, False, False])
    s = [1, 2, 3, 4]
    rs = [-1, -4, 2, 3]
    assert make_pairs(s, rs, d1, d2) == ([0, 2, 3, 1],
        [True, False, False, True])
    s = [1, 2, 3, 4]
    rs = [-1.8, -4, 2.8, 3]
    assert make_pairs(s, rs, d1, d2) == ([0, 2, 3, 1],
        [True, False, False, True])
    s = [1, 2, 3, 4]
    rs = [-4, 3, 2.8, -1.8]
    assert make_pairs(s, rs, d1, d2) == ([3, 2, 1, 0],
        [True, False, False, True])

    def d1(x, y):
        y = y+5
        return abs(x-y)
    def d2(x, y):
        y = y+5
        return abs(x+y)
    s = [1, 2, 3, 4]
    rs = [1-5, 4-5, 2-5, 3-5]
    assert make_pairs(s, rs, d1, d2) == ([0, 2, 3, 1],
        [False, False, False, False])

if __name__ == "__main__":
    test()
