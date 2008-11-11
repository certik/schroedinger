from hermes2d cimport scalar, RealFunction, RefMap, DiscreteProblem, \
        int_grad_u_grad_v, int_v, H1Space, Solution, int_u_dvdx, \
        int_u_dvdy, int_w_nabla_u_v, int_u_v, BF_ANTISYM, BC_ESSENTIAL, \
        BC_NONE, int_F_u_v, c_sqrt, BC_NATURAL

cdef int bc_type_schroed(int marker):
    if marker == 1 or marker == 3:
        return BC_NATURAL
    return BC_ESSENTIAL


cdef scalar bc_values_schroed_1d(int marker, double x, double y):
    return 0

cdef scalar bilinear_form_schroed(RealFunction* fu, RealFunction* fv,
    RefMap* ru, RefMap* rv):
  return int_u_v(fu, fv, ru, rv)

cdef double F(double x, double y):
    #return 0.
    #return (x**2+y**2)/2
    return -0.5/c_sqrt(x**2+y**2)

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

def set_forms8(DiscreteProblem dp):
    dp.thisptr.set_bilinear_form(0, 0, &bilinear_form_schroed1)
    #dp.thisptr.set_linear_form(0, &linear_form);

def set_forms8_1d(DiscreteProblem dp):
    dp.thisptr.set_bilinear_form(0, 0, &bilinear_form_schroed1_1d)
    #dp.thisptr.set_linear_form(0, &linear_form);

