from pydmd import DMD
# https://github.com/mathLab/PyDMD

from dolfin import Function
from collections import namedtuple
import numpy as np


ComplexFunction = namedtuple('ComplexFunction', ('real', 'imag'))


def dmd(functions, dt=1, modal_analysis=[]):
    '''
    Dynamic mode decomposition:
      J. Fluid Mech. (2010), vol. 656, pp. 5-28  (idea)
      On dynamic mode decomposition: theory and application; Tu, J. H et al. (implement)
      
    DMD of (ordered, dt-equispaced) snapshots.
    '''
    assert all(isinstance(f, Function) for f in functions)

    X = np.array([f.vector().get_local() for f in functions]).T
    
    # Rely on pydmd
    dmd_ = DMD(svd_rank=0, exact=True)    
    dmd_.fit(X)
    dmd_.original_time['dt'] = dt
    
    V = functions[0].function_space()
    eigs = dmd_.eigs
        
    modes = []
    # NOTE: unlike with pod where the basis was only real here the
    # modes might have complex components so ...
    for x in dmd_.modes.T:
        f_real = Function(V)
        f_real.vector().set_local(x.real)
            
        f_imag = Function(V)
        f_imag.vector().set_local(x.imag)

        modes.append(ComplexFunction(f_real, f_imag))

    if len(modal_analysis):
        return eigs, modes, dmd_.dynamics[modal_analysis]

    return eigs, modes

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, Expression, FunctionSpace, interpolate, File
    from dolfin import XDMFFile, inner, grad, dx, assemble
    from interpreter import Eval
    # Build a monomial basis for x, y, x**2, xy, y**2, ...

    deg = 4

    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, 'CG', 3)
    x = interpolate(Expression('x[0]', degree=1), V)
    y = interpolate(Expression('x[1]', degree=1), V)

    basis = []
    for i in range(deg):
        for j in range(deg):
            basis.append(Eval((x**i)*(y**j)))

    # NOTE: skipping 1 bacause Eval of it is not a Function
    energy, pod_basis = dmd(basis[1:])

    out = File('mmd_test.pvd')
    for i, f in enumerate(pod_basis):
        fr = f.real
        fr.rename('f', '0')
        out << (fr, float(i))
