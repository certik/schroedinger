#! /usr/bin/env python
"""
Electronic structure solver.

Type:

$ ./schroedinger.py

for usage and help.

"""

from math import pi
from optparse import OptionParser

from numpy import zeros, array
from scipy.sparse import coo_matrix
from pysparse import spmatrix, jdsym, precon, itsolvers

from hermes2d import initialize, finalize, Mesh, H1Shapeset, \
        PrecalcShapeset, H1Space, DiscreteProblem, Solution, ScalarView, \
        BaseView, MeshView, H1OrthoHP, OrderView, \
        MatrixView

from cschroed import set_forms7, set_forms8

def load_mat(filename):
    print "Loading a matrix '%s' in COO format..." % filename
    f = open(filename)
    n1, n2, non_zero = [int(x) for x in f.readline().split(" ")]
    x = zeros(non_zero)
    y = zeros(non_zero)
    data = zeros(non_zero)
    for i in range(non_zero):
        k, l, v = f.readline().split(" ")
        k, l, v = int(k), int(l), float(v)
        x[i] = k - 1
        y[i] = l - 1
        data[i] = v
    return coo_matrix((data, [x, y]), dims=(n1, n2))

def convert_mat(mtx):
    """
    Converts a scipy matrix "mtx" to a pysparse matrix.
    """
    mtx = mtx.tocsr()
    A = spmatrix.ll_mat(*mtx.shape)
    for i in xrange( mtx.indptr.shape[0] - 1 ):
        ii = slice( mtx.indptr[i], mtx.indptr[i+1] )
        n_in_row = ii.stop - ii.start
        A.update_add_at( mtx.data[ii], [i] * n_in_row, mtx.indices[ii] )
    return A

def solve(A, B):
    """
    Solves the generalized eigenvalue problem.

    A, B ... scipy matrices
    """
    print "converting to pysparse"
    n = A.shape[0]
    A = convert_mat(A)
    B = convert_mat(B)
    print "solving (%d x %d)" % (n, n)
    Atau = A.copy()
    tau = -1
    Atau.shift(-tau, B)
    K = precon.jacobi(Atau)
    A = A.to_sss()
    B = B.to_sss()
    n_eigs = 4
    kconv, lmbd, Q, it, it_in = jdsym.jdsym(A, B, K, n_eigs, tau, 1e-6, 150,
            itsolvers.qmrs)
    print "number of converged eigenvalues:", kconv
    #levels = []
    #for n1 in range(1, 10):
    #    for n2 in range(1, 10):
    #        levels.append(n1**2 + n2**2)
    #levels.sort()

    # well
    #E_exact = [pi**2/2 * m for m in levels]

    # oscillator
    #E_exact = [1] + [2]*2 + [3]*3 + [4]*4 + [5]*5 + [6]*6

    # hydrogen
    E_exact = [-1./2/(n-0.5)**2/4 for n in [1]+[2]*3+[3]*5 + [4]*8 + [5]*15]
    print "eigenvalues (i, FEM, exact, error):"
    for i, E in enumerate(lmbd):
        a = E
        b = E_exact[i]
        print "%2d: %10f %10f %f%%" % (i, a, b, abs((a-b)/b)*100)
    return Q

def show_sol(s):
    view = ScalarView("Eigenvector", 0, 0, 400, 400)
    view.show(s)

def schroedinger_solver(iter=2, plot=False, potential="hydrogen"):
    """
    One particle Schroedinger equation solver.

    iter ... the number of adaptive iterations to do
    plot ... plot the progress (solutions, refined solutions, errors)

    Returns the eigenvalues and eigenvectors.
    """
    mesh = Mesh()
    mesh.load("square.mesh")
    #mesh.refine_element(0)
    mesh.refine_all_elements()
    #mesh.refine_all_elements()
    #mesh.refine_all_elements()
    #mesh.refine_all_elements()

    #mview = MeshView()
    #mview.show(mesh)

    shapeset = H1Shapeset()
    space = H1Space(mesh, shapeset)
    space.set_uniform_order(2)
    space.assign_dofs()

    pss = PrecalcShapeset(shapeset)
    #bview = BaseView()
    #bview.show(space)

    dp1 = DiscreteProblem()
    dp1.set_num_equations(1)
    dp1.set_spaces(space)
    dp1.set_pss(pss)
    set_forms8(dp1)
    dp2 = DiscreteProblem()
    dp2.set_num_equations(1)
    dp2.set_spaces(space)
    dp2.set_pss(pss)
    set_forms7(dp2)

    rmesh = Mesh()
    rspace = H1Space(rmesh, shapeset)

    rp1 = DiscreteProblem()
    rp1.copy(dp1)
    rp1.set_spaces(rspace);
    set_forms8(rp1)

    rp2 = DiscreteProblem()
    rp2.copy(dp2)
    rp2.set_spaces(rspace);
    set_forms7(rp2)

    w = 320
    h = 320
    views = [ScalarView("", i*w, 0, w, h) for i in range(4)]
    viewsm = [ScalarView("", i*w, h, w, h) for i in range(4)]
    viewse = [ScalarView("", i*w, 2*h, w, h) for i in range(4)]
    for v in viewse:
        v.set_min_max_range(0, 10**-4)
    ord = OrderView("Polynomial Orders", 0, 2*h, w, h)
    mat1 = MatrixView("Matrix A", w, 2*h, w, h)
    #mat2 = MatrixView("Matrix A'", 2*w, 2*h, w, h)

    rs = None

    precision = 30.0

    for it in range(iter):

        mesh.save("refined2.mesh")
        dp1.create_matrix()
        dp1.assemble_matrix_and_rhs()
        dp2.create_matrix()
        dp2.assemble_matrix_and_rhs()
        print "converting matrices A, B"
        A = dp1.get_matrix()
        B = dp2.get_matrix()
        sols = solve(A, B)
        s = []

        n = sols.shape[1]
        for i in range(n):
            sln = Solution()
            vec = sols[:, i]
            sln.set_fe_solution(space, pss, vec)
            s.append(sln)

        if rs is not None:
            def minus2(sols, i):
                sln = Solution()
                vec = sols[:, i]
                sln.set_fe_solution(space, pss, -vec)
                return sln
            pairs, flips = make_pairs(rs, s, d1, d2)
            #print "_"*40
            #print pairs, flips
            #print len(rs), len(s)
            #from time import sleep
            #sleep(3)
            #stop
            s2 = []
            for j, flip in zip(pairs, flips):
                if flip:
                    s2.append(minus2(sols,j))
                else:
                    s2.append(s[j])
            s = s2

        if plot:
            ord.show(space)
            for i in range(min(len(s), 4)):
                views[i].show(s[i])
                views[i].set_title("Iter: %d, eig: %d" % (it, i))
            #mat1.show(dp1)

        rmesh.copy(mesh)
        rmesh.refine_all_elements()
        rspace.copy_orders(space, 1)
        rspace.assign_dofs()

        rp1.create_matrix()
        rp1.assemble_matrix_and_rhs()
        rp2.create_matrix()
        rp2.assemble_matrix_and_rhs()
        print "converting matrices A, B"
        A = rp1.get_matrix()
        B = rp2.get_matrix()
        sols = solve(A, B)
        rs = []

        n = sols.shape[1]
        for i in range(n):
            sln = Solution()
            vec = sols[:, i]
            sln.set_fe_solution(rspace, pss, vec)
            rs.append(sln)

        def minus(sols, i):
            sln = Solution()
            vec = sols[:, i]
            sln.set_fe_solution(rspace, pss, -vec)
            return sln

        # segfaults
        #mat2.show(rp1)

        def d1(x, y):
            return (x-y).l2_norm()
        def d2(x, y):
            return (x+y).l2_norm()
        from pairs import make_pairs
        pairs, flips = make_pairs(s, rs, d1, d2)
        rs2 = []
        for j, flip in zip(pairs, flips):
            if flip:
                rs2.append(minus(sols,j))
            else:
                rs2.append(rs[j])
        rs = rs2

        if plot:
            for i in range(min(len(s), len(rs), 4)):
                views[i].show(s[i])
                views[i].set_title("Iter: %d, eig: %d" % (it, i))
                viewsm[i].show(rs[i])
                viewsm[i].set_title("Ref. Iter: %d, eig: %d" % (it, i))
                viewse[i].show((s[i]-rs[i])**2)
                viewse[i].set_title("Error plot Iter: %d, eig: %d" % (it, i))


        hp = H1OrthoHP(space)
        #print "-"*40
        #print hp.calc_error(s[1], rs[0]) * 100
        #print hp.calc_error(s[1], rs[1]) * 100
        #print hp.calc_error(s[1], rs[2]) * 100
        #print hp.calc_error(s[1], rs[3]) * 100
        #print "-"*40
        print "-"*60
        print "calc error (iter=%d):" % it
        eig_converging = 0
        errors = []
        for i in range(min(len(s), len(rs))):
            error = hp.calc_error(s[i], rs[i]) * 100
            errors.append(error)
            prec = precision
            print "eig %d: %g%%  precision goal: %g%%" % (i, error, prec)
        if errors[0] > precision:
            eig_converging = 0
        elif errors[3] > precision:
            eig_converging = 3
        elif errors[1] > precision:
            eig_converging = 1
        elif errors[2] > precision:
            eig_converging = 2
        else:
            precision /= 2
        print "picked: %d" % eig_converging
        print "-"*60
        error = hp.calc_error(s[eig_converging], rs[eig_converging]) * 100
        hp.adapt(0.3)
        space.assign_dofs()



def main():
    version = "0.0-git"

    parser = OptionParser(usage="[options] args", version = "%prog " + version )
    parser.add_option( "--well",
                       action = "store_true", dest = "well",
                       default = False, help = "solve infinite potential well (particle in a box) problem" )
    parser.add_option( "--oscillator",
                       action = "store_true", dest = "oscillator",
                       default = False, help = "solve spherically symmetric linear harmonic oscillator (1 electron) problem" )
    parser.add_option( "--hydrogen",
                       action = "store_true", dest = "hydrogen",
                       default = False, help = "solve the hydrogen atom" )
    parser.add_option( "--dft",
                       action = "store_true", dest = "dft",
                       default = False, help = "perform dft calculation" )
    parser.add_option( "-p", "--plot",
                       action = "store_true", dest = "plot",
                       default = False, help = "plot the solver progress (solutions, refined solutions, errors)" )
    parser.add_option( "--exit",
                       action = "store_true", dest = "exit",
                       default = False, help = "exit at the end of calculation (i.e. do not leave the plot windows open)" )
    options, args = parser.parse_args()
    if options.well:
        schroedinger_solver(iter=2, plot=options.plot, potential="well")
    elif options.oscillator:
        schroedinger_solver(iter=2, plot=options.plot, potential="oscillator")
    elif options.hydrogen:
        schroedinger_solver(iter=2, plot=options.plot, potential="hydrogen")
    elif options.dft:
        raise NotImplementedError()
    else:
        parser.print_help()
        return

    if options.plot and not options.exit:
        # leave the plot windows open, the user needs to close them with
        # "ctrl-C":
        finalize()

if __name__ == '__main__':
    main()
