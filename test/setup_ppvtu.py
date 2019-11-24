from dolfin import *

mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, 'CG', 1)

f = Expression('t*(x[0]+x[1])', degree=1, t=1)
v = interpolate(f, V)

out = File('scalar.pvd')
for t in range(1, 5):
    f.t = t
    v.assign(interpolate(f, V))
    out << v, float(t)


V = VectorFunctionSpace(mesh, 'CG', 1)

f = Expression(('t*(x[0]+x[1])', 't'), degree=1, t=1)
v = interpolate(f, V)

out = File('vector.pvd')
for t in range(1, 5):
    f.t = t
    v.assign(interpolate(f, V))
    out << v, float(t)
