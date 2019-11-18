from dolfin import Function
from collections import namedtuple
import numpy as np


ComplexFunction = namedtuple('ComplexFunction', ('real', 'imag'))


def dmd(functions, dmd_object, dt=1, modal_analysis=[]):
    '''
    Dynamic mode decomposition:
      J. Fluid Mech. (2010), vol. 656, pp. 5-28  (idea)
      On dynamic mode decomposition: theory and application; Tu, J. H et al. (implement)
      
    DMD of (ordered, dt-equispaced) snapshots. dmd_object is the configured DMDBase instance.
    '''
    assert all(isinstance(f, Function) for f in functions)
    # Wrap for pydmd
    X = np.array([f.vector().get_local() for f in functions]).T
    
    # Rely on pydmd
    dmd_object.fit(X)
    dmd_object.original_time['dt'] = dt
    
    V = functions[0].function_space()
    eigs = dmd_object.eigs
        
    modes = []
    # NOTE: unlike with pod where the basis was only real here the
    # modes might have complex components so ...
    for x in dmd_object.modes.T:
        f_real = Function(V)
        f_real.vector().set_local(x.real)
            
        f_imag = Function(V)
        f_imag.vector().set_local(x.imag)

        modes.append(ComplexFunction(f_real, f_imag))

    if len(modal_analysis):
        return eigs, modes, dmd_object.dynamics[modal_analysis]

    return eigs, modes

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, Expression, FunctionSpace, interpolate, File
    from dolfin import XDMFFile, inner, grad, dx, assemble
    from .interpreter import Eval
    # Build a monomial basis for x, y, x**2, xy, y**2, ...

    try:
        from pydmd import DMD
        # https://github.com/mathLab/PyDMD
    except ImportError:
        from xcalc.dmdbase import DMD

    deg = 4

    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, 'CG', 1)
    f = interpolate(Expression('x[0]+x[1]', degree=1), V).vector().get_local()
    A = np.diag(np.random.rand(V.dim()))

    basis = []
    for i in range(deg):
        for j in range(deg):
            f = A.dot(f)
            Af = Function(V); Af.vector().set_local(f)
            basis.append(Af)
            
    # NOTE: skipping 1 bacause Eval of it is not a Function
    dmd_ = DMD(svd_rank=-1, exact=False)
    energy, pod_basis = dmd(basis[1:], dmd_)

    print(np.linalg.norm(dmd_.snapshots - dmd_.reconstructed_data.real))
    print(len(pod_basis), len(basis[1:]))
