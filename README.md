# fenics-calc
This module provides a simple lazy evaluator of algebraic expressions over FEniCS functions. Expressions are defined in the `UFL` language. They can be evaluated provided that the function spaces of each `Function` type arguments are of the tensor product form `V x V ... x V`. The expression then results in a `Function` in some appropriate tensor product space with `V`. In general coefficients of the function for expression `(op u v)` are computed as `(op U V)` where `U, V` are coefficient vectors of the arguments. This approach means that the result is exact iff `op` is linear; otherwise there is an interpolation error.

````python
from dolfin import *
from xcalc import Eval

mesh = UnitSquareMesh(5, 5)
T = TensorFunctionSpace(mesh, 'CG', 1)

A = interpolate(Expression((('x[0]', 'x[1]'),
                                    ('2*x[0]+x[1]', 'x[0]+3*x[1]')), degree=1), T)
expr = tr(sym(A) + skew(A))
me = Eval(expr)  # Function in VectorFunctionSpace(mesh, 'CG', 1)
                 # Exact due to linearity
````

Functions can be grouped into `TempSeries` (avoiding name collision with FEniCS's native `TimeSeries`). Same algebraic operations over these objects are supported as with normal functions. `TempSeries` are collapsed into functions by time-averaging operations such as mean

````python
from dolfin import *
from xcalc.tempseries import PVDTempSeries
from xcalc.operators import Mean

# Let there be a time series stored in pvd of function in V
mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, 'CG', 1)
series = PVDTempSeries('pvd_test.pvd', V)

mean = Mean(abs(series))  # A lazy node! 
````

## What works and what does not
At the moment the interpreter supports most commonly used nodes in UFL except for
- differentiation nodes, e.g. `grad` (hence algebraic expressions)
- FEM specific nodes such as `FacetNormal`, `jump`, `avg` and so on 
- Currently MPI support is missing.

### Limitations
In order for `Eval` to be successfull operations on the dofs must translate nicely to `numpy`. Among other things this means that the spaces where the expression terminals are defined must be collapseable to scalars (in order to access components). This rules out vector-valued elements such as Raviart-Thomas

````python
from dolfin import *
from xcalc import Eval

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'RT', 1)

x = Function(V)
y = Function(V)

 Eval(inner(x, y))  # Fails - what is the scalar space here?
 Eval(as_vector((x[0], y[1])))  # Fails - same reason
 Eval(2*x + y)  # Fails - handled internally as 2*as_vector((x[0], x[1])) + y
 Eval(dot(outer(x, y), x))  # Fails - numpy eval of the mat is array of length 2
                            #         numpy eval of the vec is single number
                            #         numpy op produces array of 2 where 1 is expected 
````

## FEniCS compatibility
This package is CI tested against FEniCS packages for `ubuntu 16.04 LTS` [![Build Status](https://travis-ci.org/MiroK/fenics-calc.svg?branch=master)](https://travis-ci.org/MiroK/fenics-calc)

## Dependencies and installation
In addition to FEniCS stack `h5py` is needed if one wants to use `XDMFTempSeries`. Dynamic mode decomposition of series relies on [PyDMD](https://github.com/mathLab/PyDMD) module. After that, `python setup.py install --user` (or variants) is how to install this module. 