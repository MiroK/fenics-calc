import xml.etree.ElementTree as ET
from dolfin import Function
from utils import space_of
import numpy as np
import os


class TempSeries(Function):
    '''Collection of snapshots that are function in same V'''
    def __init__(self, ft_pairs):
        # NOTE: this is derived from Function just to allow nice
        # interplay with the interpreter. If there were space time 
        # elements then we could have eval f(t, x) support
        functions, times = list(zip(*ft_pairs))
        
        # Check that every f has same f
        V = space_of(functions)
        # Time interval check
        dt = np.diff(times)
        assert (dt > 0).all()
        
        self.functions = functions
        self.times = times
        self.V = V

        Function.__init__(self, V)

    def __iter__(self):
        # op(series) = series(op(functions))
        for f in self.functions: yield f 

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, index):
        # NOTE: by having this [] is access to functions not a new time series
        # of the components
        if isinstance(index, int):
            return self.functions[index]
        else:
            return TempSeries(zip(self.functions[index], self.times[index]))


def common_interval(series):
    '''Series are compatible if they over same V and cover same interval'''
    series = filter(lambda s: isinstance(s, TempSeries), series)

    interval, V = [], None
    for s in series:
        V_ = s.V
        assert V is None or (V.mesh().id() == V_.mesh().id()
                             and 
                             V.ufl_element() == V_.ufl_element())

        interval_ = s.times
        assert not interval or np.linalg.norm(interval - interval_) < 1E-14

        V = V_
        interval = interval_
    return interval


def PVDTempSeries(path, V, first=0, last=None):
    '''Read in the temp series of functions in V from PVD file'''
    _, ext = os.path.splitext(path)
    assert ext == '.pvd'

    tree = ET.parse(path)
    collection = list(tree.getroot())[0]
    # Read in paths/timestamps for VTUs. NOTE: as thus is supposed to be serial 
    # assert part 0
    vtus = []
    for dataset in collection:
        assert dataset.attrib['part'] == '0'
        vtus.append((dataset.attrib['file'], float(dataset.attrib['timestep'])))
    
    vtus = vtus[slice(first, last, None)]
    # path.vtu -> function
    ft_pairs = [(read_vtu_function(path, V), t) for path, t in vtus]

    return TempSeries(ft_pairs)
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    from dolfin import *

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    a = Function(V)
    b = Function(V)

    c = TempSeries([(a, 0), (b, 1)])
    print len(c)

    for f in c:
        print f

    print type(c[0:1])

    # ---

    mesh = UnitSquareMesh(64, 64)
    V = FunctionSpace(mesh, 'CG', 1)

    PVDTempSeries('pod_test.pvd', V)
