#! /usr/bin/env python
"""
Electronic structure solver.

Type:

$ ./schroedinger.py

for usage and help.

"""

from math import pi, sqrt
from optparse import OptionParser

from numpy import zeros, array
from scipy.sparse import coo_matrix
from pysparse import spmatrix, jdsym, precon, itsolvers

from hermes2d import initialize, finalize, Mesh, H1Shapeset, \
        PrecalcShapeset, H1Space, DiscreteProblem, Solution, ScalarView, \
        BaseView, MeshView, H1OrthoHP, OrderView, \
        MatrixView, set_verbose

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

def solve(A, B, n_eigs=4, verbose=False):
    """
    Solves the generalized eigenvalue problem.

    A, B ... scipy matrices

    returns (lmbd, Q), where lmbd are the eigenvalues and Q is a numpy array of
    solutions
    """
    if verbose:
        print "converting to pysparse"
    n = A.shape[0]
    A = convert_mat(A)
    B = convert_mat(B)
    if verbose:
        print "solving (%d x %d)" % (n, n)
    Atau = A.copy()
    tau = -1
    Atau.shift(-tau, B)
    K = precon.jacobi(Atau)
    A = A.to_sss()
    B = B.to_sss()
    kconv, lmbd, Q, it, it_in = jdsym.jdsym(A, B, K, n_eigs, tau, 1e-6, 150,
            itsolvers.qmrs)
    if verbose:
        print "number of converged eigenvalues:", kconv
    return lmbd, Q

def show_sol(s):
    view = ScalarView("Eigenvector", 0, 0, 400, 400)
    view.show(s)

def print_eigs(eigs, E_exact=None):
    """
    Nicely prints the eigenvalues "eigs", together with analytical solution.

    """
    assert E_exact is not None

    assert len(eigs) <= len(E_exact)
    print "      n      FEM       exact    error"
    for i, E in enumerate(eigs):
        a = E
        b = E_exact[i]
        print "    %2d: %10f %10f %f%%" % (i, a, b, abs((a-b)/b)*100)

def schroedinger_solver(n_eigs=4, iter=2, verbose_level=1, plot=False,
        potential="hydrogen", report=False):
    """
    One particle Schroedinger equation solver.

    n_eigs ... the number of the lowest eigenvectors to calculate
    iter ... the number of adaptive iterations to do
    verbose_level ...
            0 ... quiet
            1 ... only moderate output (default)
            2 ... lot's of output
    plot ... plot the progress (solutions, refined solutions, errors)
    potential ... the V(x) for which to solve, one of:
            well, oscillator, hydrogen
    report ... it will save raw data to a file, useful for creating graphs etc.

    Returns the eigenvalues and eigenvectors.
    """
    set_verbose(verbose_level == 2)
    pot = {"well": 0, "oscillator": 1, "hydrogen": 2, "three-points": 3}
    pot_type = pot[potential]
    if report:
        from timeit import default_timer as clock
        from tables import IsDescription, UInt32Col, Float32Col, openFile, \
                Float64Col
        class Iteration(IsDescription):
            n = UInt32Col()
            DOF = UInt32Col()
            DOF_reference = UInt32Col()
            cpu_solve = Float32Col()
            cpu_solve_reference = Float32Col()
            eig_errors = Float64Col(shape=(n_eigs,))
        h5file = openFile("report.h5", mode = "w", title = "Simulation data")
        group = h5file.createGroup("/", 'schroed', 'Schroedinger solver')
        table = h5file.createTable(group, 'sim', Iteration, "Simulation")
        iteration = table.row

    mesh = Mesh()
    mesh.load("square.mesh")
    if potential == "well":
        # Read the width of the mesh automatically. This assumes there is just
        # one square element:
        a = sqrt(mesh.get_element(0).get_area())
        # set N high enough, so that we get enough analytical eigenvalues:
        N = 10
        levels = []
        for n1 in range(1, N):
            for n2 in range(1, N):
                levels.append(n1**2 + n2**2)
        levels.sort()

        E_exact = [pi**2/(2.*a**2) * m for m in levels]
    elif potential == "oscillator":
        E_exact = [1] + [2]*2 + [3]*3 + [4]*4 + [5]*5 + [6]*6
    elif potential == "hydrogen":
        Z = 1 # atom number
        E_exact = [-float(Z)**2/2/(n-0.5)**2/4 for n in [1]+[2]*3+[3]*5 +\
                                    [4]*8 + [5]*15]
    else:
        E_exact = [1.]*50
    if len(E_exact) < n_eigs:
        print n_eigs
        print E_exact
        raise Exception("We don't have enough analytical eigenvalues.")
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
    # this is induced by set_verbose():
    #dp1.set_quiet(not verbose)
    dp1.set_num_equations(1)
    dp1.set_spaces(space)
    dp1.set_pss(pss)
    set_forms8(dp1, pot_type)
    dp2 = DiscreteProblem()
    # this is induced by set_verbose():
    #dp2.set_quiet(not verbose)
    dp2.set_num_equations(1)
    dp2.set_spaces(space)
    dp2.set_pss(pss)
    set_forms7(dp2)

    rmesh = Mesh()
    rspace = H1Space(rmesh, shapeset)

    rp1 = DiscreteProblem()
    rp1.copy(dp1)
    rp1.set_spaces(rspace);
    set_forms8(rp1, pot_type)

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

    if verbose_level >= 1:
        print "Problem initialized. Starting calculation."

    for it in range(iter):
        if verbose_level >= 1:
            print "-"*80
            print "Starting iteration %d." % it
        if report:
            iteration["n"] = it

        #mesh.save("refined2.mesh")
        if verbose_level >= 1:
            print "Assembling the matrices A, B."
        dp1.create_matrix()
        dp1.assemble_matrix_and_rhs()
        dp2.create_matrix()
        dp2.assemble_matrix_and_rhs()
        if verbose_level == 2:
            print "converting matrices A, B"
        A = dp1.get_matrix()
        B = dp2.get_matrix()
        if verbose_level >= 1:
            n = A.shape[0]
            print "Solving the problem Ax=EBx  (%d x %d)." % (n, n)
        if report:
            n = A.shape[0]
            iteration["DOF"] = n
        if report:
            t = clock()
        eigs, sols = solve(A, B, n_eigs, verbose_level == 2)
        if report:
            t = clock() - t
            iteration["cpu_solve"] = t
        if verbose_level >= 1:
            print "   \-Done."
            print_eigs(eigs, E_exact)
        s = []

        n = sols.shape[1]
        for i in range(n):
            sln = Solution()
            vec = sols[:, i]
            sln.set_fe_solution(space, pss, vec)
            s.append(sln)

        if verbose_level >= 1:
            print "Matching solutions."
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
            if verbose_level >= 1:
                print "plotting: solution"
            ord.show(space)
            for i in range(min(len(s), 4)):
                views[i].show(s[i])
                views[i].set_title("Iter: %d, eig: %d" % (it, i))
            #mat1.show(dp1)

        if verbose_level >= 1:
            print "reference: initializing mesh."
        rmesh.copy(mesh)
        rmesh.refine_all_elements()
        rspace.copy_orders(space, 1)
        rspace.assign_dofs()

        if verbose_level >= 1:
            print "reference: assembling the matrices A, B."
        rp1.create_matrix()
        rp1.assemble_matrix_and_rhs()
        rp2.create_matrix()
        rp2.assemble_matrix_and_rhs()
        if verbose_level == 2:
            print "converting matrices A, B"
        A = rp1.get_matrix()
        B = rp2.get_matrix()
        if verbose_level >= 1:
            n = A.shape[0]
            print "reference: solving the problem Ax=EBx  (%d x %d)." % (n, n)
        if report:
            n = A.shape[0]
            iteration["DOF_reference"] = n
        if report:
            t = clock()
        eigs, sols = solve(A, B, n_eigs, verbose_level == 2)
        if report:
            t = clock() - t
            iteration["cpu_solve_reference"] = t
        if verbose_level >= 1:
            print "   \-Done."
            print_eigs(eigs, E_exact)
        rs = []

        n = sols.shape[1]
        for i in range(n):
            sln = Solution()
            vec = sols[:, i]
            sln.set_fe_solution(rspace, pss, vec)
            rs.append(sln)

        if verbose_level >= 1:
            print "reference: matching solutions."
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
            if verbose_level >= 1:
                print "plotting: solution, reference solution, errors"
            for i in range(min(len(s), len(rs), 4)):
                #views[i].show(s[i])
                #views[i].set_title("Iter: %d, eig: %d" % (it, i))
                viewsm[i].show(rs[i])
                viewsm[i].set_title("Ref. Iter: %d, eig: %d" % (it, i))
                viewse[i].show((s[i]-rs[i])**2)
                viewse[i].set_title("Error plot Iter: %d, eig: %d" % (it, i))


        if verbose_level >= 1:
            print "Calculating errors."
        hp = H1OrthoHP(space)
        if verbose_level == 2:
            print "-"*60
            print "calc error (iter=%d):" % it
        eig_converging = 0
        errors = []
        for i in range(min(len(s), len(rs))):
            error = hp.calc_error(s[i], rs[i]) * 100
            errors.append(error)
            prec = precision
            if verbose_level >= 1:
                print "eig %d: %g%%  precision goal: %g%%" % (i, error, prec)
        if report:
            iteration["eig_errors"] = array(errors)
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
        # uncomment the following line to only converge to some eigenvalue:
        #eig_converging = 3
        if verbose_level >= 1:
            print "picked: %d" % eig_converging
        error = hp.calc_error(s[eig_converging], rs[eig_converging]) * 100
        if verbose_level >= 1:
            print "Adapting the mesh."
        hp.adapt(0.3)
        space.assign_dofs()
        if report:
            iteration.append()
            table.flush()
    if report:
        h5file.close()



def main():
    version = "0.1-git"

    parser = OptionParser(usage="[options] args", version="%prog " + version)
    parser.add_option("-v", "--verbose",
                       action="store_true", dest="verbose",
                       default=False, help="produce verbose output during solving")
    parser.add_option("-q", "--quiet",
                       action="store_true", dest="quiet",
                       default=False, help="be totally quiet, e.g. only errors are written to stdout")
    parser.add_option("--well",
                       action="store_true", dest="well",
                       default=False, help="solve infinite potential well (particle in a box) problem")
    parser.add_option("--oscillator",
                       action="store_true", dest="oscillator",
                       default=False, help="solve spherically symmetric linear harmonic oscillator (1 electron) problem")
    parser.add_option("--hydrogen",
                       action="store_true", dest="hydrogen",
                       default=False, help="solve the hydrogen atom")
    parser.add_option("--dft",
                       action="store_true", dest="dft",
                       default=False, help="perform dft calculation")
    parser.add_option("--three-points",
                       action="store_true", dest="three",
                       default=False, help="three points geometry calculation")
    parser.add_option("--iter",
                       action="store", type="int", dest="iter",
                       default=5, help="the number of iterations to calculate [default %default]")
    parser.add_option("--neigs",
                       action="store", type="int", dest="neigs",
                       default=4, help="the number of eigenvectors to calculate [default %default]")
    parser.add_option("-p", "--plot",
                       action="store_true", dest="plot",
                       default=False, help="plot the solver progress (solutions, refined solutions, errors)")
    parser.add_option("--exit",
                       action="store_true", dest="exit",
                       default=False, help ="exit at the end of calculation (with --plot), i.e. do not leave the plot windows open")
    parser.add_option("--report",
                       action="store_true", dest="report",
                       default=False, help="create a report")
    options, args = parser.parse_args()
    if options.verbose:
        verbose_level = 2
    elif options.quiet:
        verbose_level = 0
    else:
        verbose_level = 1
    kwargs = {
            "n_eigs": options.neigs,
            "iter": options.iter,
            "verbose_level": verbose_level,
            "plot": options.plot,
            "report": options.report,
            }
    if options.well:
        kwargs.update({"potential": "well"})
    elif options.oscillator:
        kwargs.update({"potential": "oscillator"})
    elif options.hydrogen:
        kwargs.update({"potential": "hydrogen"})
    elif options.dft:
        raise NotImplementedError()
    elif options.three:
        kwargs.update({"potential": "three-points"})
    else:
        parser.print_help()
        return
    schroedinger_solver(**kwargs)

    if options.plot and not options.exit:
        # leave the plot windows open, the user needs to close them with
        # "ctrl-C":
        finalize()

if __name__ == '__main__':
    main()
