from dolfin import *
from xcalc import Eval
from xcalc.operators import Eigw, Eigv


mesh = UnitSquareMesh(5, 5)
V = VectorFunctionSpace(mesh, 'CG', 1)

u = interpolate(Expression(('2*x[0]', 'x[1]'), degree=1), V)

for i in range(2):
    expr = Eigw(grad(u))[i]
    me = Eval(expr)  
    File('eigw_%d.pvd' % i) << me

for i in range(2):
    expr = Eigv(grad(u))[:, i]
    me = Eval(expr)
    File('eigv_%d.pvd' % i) << me
    
