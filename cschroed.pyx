from hermes2d cimport scalar, RealFunction, RefMap, DiscreteProblem, \
        int_grad_u_grad_v, int_v, H1Space, Solution, int_u_dvdx, \
        int_u_dvdy, int_w_nabla_u_v, int_u_v, BF_ANTISYM, BC_ESSENTIAL, \
        BC_NONE, int_F_u_v, c_sqrt, BC_NATURAL, int_F_v, MeshFunction, \
        c_MeshFunction

cdef int bc_type_schroed(int marker):
    if marker == 1 or marker == 3:
        return BC_NATURAL
    return BC_ESSENTIAL


cdef scalar bc_values_schroed_1d(int marker, double x, double y):
    return 0

cdef scalar bilinear_form_schroed(RealFunction* fu, RealFunction* fv,
    RefMap* ru, RefMap* rv):
  return int_u_v(fu, fv, ru, rv)

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
        return -0.5/c_sqrt(x**2+y**2)#+0.00001*x
    else: # 3
        return -0.7/c_sqrt(x**2+(y-10)**2)-0.6/c_sqrt((x-10)**2+y**2)-0.5/c_sqrt((x+10)**2+y**2)


cdef double F2(double x, double y):
    #return 0.
    #return (x**2+y**2)/2
    cdef double r = x
    cdef double l = 2.0
    return -1/r + l*(l+1)/(2*r**2)

cdef scalar bilinear_form_schroed1(RealFunction* fu, RealFunction* fv,
                RefMap* ru, RefMap* rv):
    return int_grad_u_grad_v(fu, fv, ru, rv) / 2 + int_F_u_v(&F, fu, fv, ru, rv)

cdef scalar bilinear_form_schroed1_1d(RealFunction* fu, RealFunction* fv,
                RefMap* ru, RefMap* rv):
    return int_grad_u_grad_v(fu, fv, ru, rv) / 2 + int_F_u_v(&F2, fu, fv, ru, rv)

def set_bc_schroed_1d(H1Space space):
    space.thisptr.set_bc_types(&bc_type_schroed)
    space.thisptr.set_bc_values(&bc_values_schroed_1d)

def set_forms7(DiscreteProblem dp):
    dp.thisptr.set_bilinear_form(0, 0, &bilinear_form_schroed)
    #dp.thisptr.set_linear_form(0, &linear_form);

def set_forms8(DiscreteProblem dp, pot_type, MeshFunction potential2):
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
    dp.thisptr.set_bilinear_form(0, 0, &bilinear_form_schroed1)
    #dp.thisptr.set_linear_form(0, &linear_form);

def set_forms8_1d(DiscreteProblem dp):
    dp.thisptr.set_bilinear_form(0, 0, &bilinear_form_schroed1_1d)
    #dp.thisptr.set_linear_form(0, &linear_form);

cdef scalar bilinear_form(RealFunction *fu, RealFunction *fv,
        RefMap *ru, RefMap *rv):
    return int_grad_u_grad_v(fu, fv, ru, rv)

cdef c_MeshFunction *rho_poisson

cdef double F_poisson(double x, double y):
    # XXX --- here we should probably return something like 4*pi*value, let's
    # check it
    return rho_poisson.get_pt_value(x, y)

cdef scalar linear_form(RealFunction *fv, RefMap *rv):
    return int_F_v(&F_poisson, fv, rv)


def set_forms_poisson(DiscreteProblem dp, MeshFunction rho=None):
    """
    rho ... the right hand side of the Poisson equation
    """
    global rho_poisson
    if rho is not None:
        rho_poisson = <c_MeshFunction *>(rho.thisptr)
    dp.thisptr.set_bilinear_form(0, 0, &bilinear_form)
    dp.thisptr.set_linear_form(0, &linear_form);

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
