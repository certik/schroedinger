from hermes2d._hermes2d cimport scalar, RealFunction, RefMap, WeakForm, \
        int_grad_u_grad_v, int_v, H1Space, Solution, int_u_dvdx, \
        int_u_dvdy, int_w_nabla_u_v, int_u_v, BF_ANTISYM, BC_ESSENTIAL, \
        BC_NONE, int_F_u_v, c_sqrt, BC_NATURAL, int_F_v, MeshFunction, \
        c_MeshFunction

from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_grad_u_grad_v, int_grad_u_grad_v_ord, int_v, malloc, ExtDataReal, \
        c_Ord, create_Ord, FuncOrd, GeomOrd, ExtDataOrd, int_v_ord

cdef int bc_type_schroed(int marker):
    if marker == 1 or marker == 3:
        return BC_NATURAL
    return BC_ESSENTIAL


cdef scalar bc_values_schroed_1d(int marker, double x, double y):
    return 0

cdef scalar bilinear_form_schroed(int n, double *wt, FuncReal *u, FuncReal *v,
        GeomReal *e, ExtDataReal *ext):
    cdef int i
    cdef double result=0
    for i in range(n):
        result += wt[i] * u.val[i] * v.val[i]
    return result

cdef int potential_type
cdef int use_other_terms
cdef c_MeshFunction *potential_other_terms

cdef double F(double x, double y):
    global potential_type
    global potential_other_terms
    global use_other_terms
    cdef double V
    if potential_type == 0:
        return 0.
    elif potential_type == 1:
        if use_other_terms:
            V = potential_other_terms.get_pt_value(x, y)
        else:
            V = 0
        return (x**2+y**2)/2+0.00001*x + V
    elif potential_type == 2:
        return -0.5/c_sqrt(x**2+y**2)+0.00001*x
    else: # 3
        return -0.5/c_sqrt(x**2+(y-10)**2)-0.5/c_sqrt((x-10)**2+y**2)-0.5/c_sqrt((x+10)**2+y**2)


cdef double F2(double x, double y):
    #return 0.
    #return (x**2+y**2)/2
    cdef double r = x
    cdef double l = 2.0
    return -1/r + l*(l+1)/(2*r**2)

cdef scalar bilinear_form_schroed1(int n, double *wt, FuncReal *u, FuncReal *v,
        GeomReal *e, ExtDataReal *ext):
    cdef int i
    cdef double result=0
    for i in range(n):
        result += wt[i] * ((u.dx[i]*v.dx[i] + u.dy[i]*v.dy[i])/2 + \
                F(e.x[i], e.y[i]) * u.val[i]*v.val[i])
    return result

#cdef scalar bilinear_form_schroed1_1d(RealFunction* fu, RealFunction* fv,
#                RefMap* ru, RefMap* rv):
#    return int_grad_u_grad_v(fu, fv, ru, rv) / 2 + int_F_u_v(&F2, fu, fv, ru, rv)

#def set_bc_schroed_1d(H1Space space):
#    space.thisptr.set_bc_types(&bc_type_schroed)
#    space.thisptr.set_bc_values(&bc_values_schroed_1d)

cdef c_Ord bilinear_form_ord(int n, double *wt, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return create_Ord(20)

cdef c_Ord linear_form_ord(int n, double *wt, FuncOrd *u, GeomOrd
        *e, ExtDataOrd *ext):
    return create_Ord(20)


def set_forms7(WeakForm dp):
    dp.thisptr.add_biform(0, 0, &bilinear_form_schroed, &bilinear_form_ord)

def set_forms8(WeakForm dp, pot_type, MeshFunction potential2):
    """
    Set the forms for the matrix A.

    pot_type ..... 0, 1, or 2
    potential2 ... other terms of the potential in the form of a MeshFunction
    """
    global potential_type
    potential_type = pot_type
    global potential_other_terms
    global use_other_terms
    if potential2 is None:
        use_other_terms = 0
    else:
        use_other_terms = 1
        potential_other_terms = <c_MeshFunction *>(potential2.thisptr)
    dp.thisptr.add_biform(0, 0, &bilinear_form_schroed1, &bilinear_form_ord)

cdef scalar bilinear_form(int n, double *wt, FuncReal *u, FuncReal *v,
        GeomReal *e, ExtDataReal *ext):
    cdef int i
    cdef double result=0
    for i in range(n):
        result += wt[i] * (u.dx[i]*v.dx[i] + u.dy[i]*v.dy[i])
    return result

cdef c_MeshFunction *rho_poisson

cdef double F_poisson(double x, double y):
    # XXX --- here we should probably return something like 4*pi*value, let's
    # check it
    return rho_poisson.get_pt_value(x, y)

cdef scalar linear_form(int n, double *wt, FuncReal *u, GeomReal
        *e, ExtDataReal *ext):
    cdef int i
    cdef double result=0
    for i in range(n):
        result += wt[i] * (F_poisson(e.x[i], e.y[i]) * u.val[i])
    return result

def set_forms_poisson(WeakForm dp, MeshFunction rho=None):
    """
    rho ... the right hand side of the Poisson equation
    """
    global rho_poisson
    if rho is not None:
        rho_poisson = <c_MeshFunction *>(rho.thisptr)
    dp.thisptr.add_biform(0, 0, &bilinear_form, &bilinear_form_ord)
    dp.thisptr.add_liform(0, &linear_form, &linear_form_ord);

cdef extern from "dft.h":
    double vxc(double n, int relat)

def get_vxc(n, relat):
    """
    Calculates the xc-potential from the charge density "n".

    relat:
        0 ... nonrelativistic potential
        1 ... relativistic potential
    """
    return vxc(n, relat)
