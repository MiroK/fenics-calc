from xcalc.function_read import read_vtu_mesh
from xcalc.timeseries import PVDTempSeries
from xcalc.interpreter import Eval
from xcalc.operators import Mean, RMS
from xcalc.pod import pod
from dolfin import *
import numpy as np


dir = '/mn/sarpanitu/eksterne-u1/mirok/Documents/Programming/DragANN/data/ActiveControl'

# Since we want to combine the two series loading the mesh automatically would 
# give different mesh IDs - xcalc is strict about this assumption so declare the 
# mesh/space ourselves
cell = triangle
mesh = read_vtu_mesh('%s/p_out000000.vtu' % dir, cell)

# Load series
Q = FunctionSpace(mesh, FiniteElement('Lagrange', cell, 1))
p = PVDTempSeries('%s/p_out.pvd' % dir, Q)

V = FunctionSpace(mesh, VectorElement('Lagrange', cell, 1))
v = PVDTempSeries('%s/u_out.pvd' % dir, V)

# Chop to region of interest
temp_slice = slice(200, 401, None)

p = p.getitem(temp_slice)
v = v.getitem(temp_slice)

# Some funny series that we want to look at 
function_series = Eval(sqrt(p**2 + inner(v, v)/2))
# NOTE: nodes are the actual functions in the series
functions = function_series.nodes

# Run pod analysis on it
nmodes = 6
energy, pod_basis, C = pod(functions, modal_analysis=range(nmodes))

out = File('%s/pod_%s.pvd' % (dir, dir.lower()))
for index, f in enumerate(pod_basis[slice(0, nmodes)]):
    f.rename('f', '0')
    out << (f, float(index))

# Dump the energy
np.savetxt('%s/pod_%s_energy.txt' % (dir, dir.lower()), energy)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.figure()
plt.semilogy(energy, color='blue', marker='o', linestyle=':')
plt.savefig('%s/pod_%s_energy.png' % (dir, dir.lower()))

times = function_series.times

plt.figure()
for i, coef_i in enumerate(C):
    plt.plot(times, coef_i, label=str(i))
plt.legend(loc='best')
plt.savefig('%s/pod_%s_modes.png' % (dir, dir.lower()))

plt.show()
