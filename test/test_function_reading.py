from xcalc.timeseries import XDMFTempSeries, PVDTempSeries
from xcalc.function_read import read_vtu_mesh, read_h5_mesh, union_mesh
from dolfin import *
import numpy as np
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) getting series from VTU/XDMF files'''
    try:
        import h5py
        has_h5py = True
    except ImportError:
        has_h5py = False

    @unittest.skipIf(not has_h5py, 'missing h5py')
    def test_xdmf_scalar(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression('x[0]', degree=1), V)
        f1 = interpolate(Expression('x[1]', degree=1), V)

        with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
            f0.rename('f', '0')
            out.write(f0, 0.)

            f1.rename('f', '0')
            out.write(f1, 1.)

        # PVDTempSeries('pod_test.pvd', V)
        series = XDMFTempSeries('xdmf_test.xdmf', V)
        self.assertTrue(error(f0, series[0]) < 1E-14)
        self.assertTrue(error(f1, series[1]) < 1E-14)
        
    @unittest.skipIf(not has_h5py, 'missing h5py')
    def test_xdmf_vector(self):
        mesh = UnitSquareMesh(3, 3)
        V = VectorFunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression(('x[0]', 'x[1]'), degree=1), V)
        f1 = interpolate(Expression(('2*x[0]', '-3*x[1]'), degree=1), V)

        with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
            f0.rename('f', '0')
            out.write(f0, 0.)

            f1.rename('f', '0')
            out.write(f1, 1.)

        # PVDTempSeries('pod_test.pvd', V)
        series = XDMFTempSeries('xdmf_test.xdmf', V)
        self.assertTrue(error(f0, series.getitem(0)) < 1E-14)
        self.assertTrue(error(f1, series.getitem(1)) < 1E-14)
        
    @unittest.skipIf(not has_h5py, 'missing h5py')
    def test_xdmf_tensor(self):
        mesh = UnitSquareMesh(3, 3)
        V = TensorFunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression((('x[0]', 'x[1]'), ('x[0]', '-x[1]')), degree=1), V)
        f1 = interpolate(Expression((('2*x[0]', '-3*x[1]'), ('4*x[0]', '-3*x[1]')), degree=1), V)
        
        with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
            f0.rename('f', '0')
            out.write(f0, 0.)

            f1.rename('f', '0')
            out.write(f1, 1.)

        # PVDTempSeries('pod_test.pvd', V)
        series = XDMFTempSeries('xdmf_test.xdmf', V)
        self.assertTrue(error(f0, series.getitem(0)) < 1E-14)
        self.assertTrue(error(f1, series.getitem(1)) < 1E-14)

    @unittest.skipIf(not has_h5py, 'missing h5py')
    def test_xdmf_scalar(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression('x[0]', degree=1), V)
        f1 = interpolate(Expression('x[1]', degree=1), V)

        with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
            f0.rename('f', '0')
            out.write(f0, 0.)

            f1.rename('f', '0')
            out.write(f1, 1.)

        # PVDTempSeries('pod_test.pvd', V)
        series = XDMFTempSeries('xdmf_test.xdmf', V=V.ufl_element())
    
    @unittest.skipIf(not has_h5py, 'missing h5py')
    def test_h5_mesh_restored(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression('x[0]', degree=1), V)
        f1 = interpolate(Expression('x[1]', degree=1), V)

        with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
            f0.rename('f', '0')
            out.write(f0, 0.)

            f1.rename('f', '0')
            out.write(f1, 1.)

        mesh_ = read_h5_mesh('xdmf_test.h5', mesh.ufl_cell())

        error = np.linalg.norm(mesh.coordinates() - mesh_.coordinates(), np.inf)
        self.assertTrue(error < 1E-14)

        error = np.linalg.norm(mesh.cells() - mesh_.cells(), np.inf)
        self.assertTrue(error < 1E-14)

    # ---

    def test_vtu_scalar(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression('x[0]', degree=1), V)
        f1 = interpolate(Expression('x[1]', degree=1), V)

        out = File('pvd_test.pvd')
        f0.rename('f', '0')
        out << (f0, 0.)
        
        f1.rename('f', '0')
        out << (f1, 1.)

        series = PVDTempSeries('pvd_test.pvd', V)
        self.assertTrue(error(f0, series.getitem(0)) < 1E-14)
        self.assertTrue(error(f1, series.getitem(1)) < 1E-14)

    def test_vtu_vector(self):
        mesh = UnitSquareMesh(3, 3)
        V = VectorFunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression(('x[0]', 'x[1]'), degree=1), V)
        f1 = interpolate(Expression(('2*x[0]', '-3*x[1]'), degree=1), V)

        out = File('pvd_test.pvd')
        f0.rename('f', '0')
        out << (f0, 0.)
        
        f1.rename('f', '0')
        out << (f1, 1.)

        series = PVDTempSeries('pvd_test.pvd', V)
        self.assertTrue(error(f0, series.getitem(0)) < 1E-14)
        self.assertTrue(error(f1, series.getitem(1)) < 1E-14)

    def test_vtu_tensor(self):
        mesh = UnitSquareMesh(3, 3)

        V = TensorFunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression((('x[0]', 'x[1]'), ('2*x[0]', '-3*x[1]')),
                                    degree=1), V)
        f1 = interpolate(Expression((('2*x[0]', '-3*x[1]'), ('x[0]', 'x[1]')),
                                    degree=1), V)

        out = File('pvd_test.pvd')
        f0.rename('f', '0')
        out << (f0, 0.)
        
        f1.rename('f', '0')
        out << (f1, 1.)

        series = PVDTempSeries('pvd_test.pvd', V)
        self.assertTrue(error(f0, series.getitem(0)) < 1E-14)
        self.assertTrue(error(f1, series.getitem(1)) < 1E-14)

    def test_vtu_mesh_recovery(self):
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression('x[0]', degree=1), V)
        f1 = interpolate(Expression('x[1]', degree=1), V)

        out = File('pvd_test.pvd')
        f0.rename('f', '0')
        out << (f0, 0.)

        mesh_ = read_vtu_mesh('pvd_test000000.vtu', mesh.ufl_cell())

        error = np.linalg.norm(mesh.coordinates() - mesh_.coordinates(), np.inf)
        self.assertTrue(error < 1E-14)

        error = np.linalg.norm(mesh.cells() - mesh_.cells(), np.inf)
        self.assertTrue(error < 1E-14)
        
    def test_vtu_scalar_restored(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f0 = interpolate(Expression('x[0]', degree=1), V)
        f1 = interpolate(Expression('x[1]', degree=1), V)

        out = File('pvd_test.pvd')
        f0.rename('f', '0')
        out << (f0, 0.)
        
        f1.rename('f', '0')
        out << (f1, 1.)

        series = PVDTempSeries('pvd_test.pvd', V=V.ufl_element())
        self.assertTrue(error(f0, series.getitem(0)) < 1E-14)
        self.assertTrue(error(f1, series.getitem(1)) < 1E-14)


    def test_mesh_union(self):
        dx, dy = 0.5, 0.5

        meshes = []
        for j in range(3):
            for i in range(3):
                x0, y0 = dx*i, dy*j
                x1, y1 = x0 + dx, y0 + dy
                meshes.append(RectangleMesh(Point(x0, y0), Point(x1, y1), 4, 4))
                
        union = union_mesh(meshes)
        self.assertTrue(union.num_cells() == sum(m.num_cells() for m in meshes))

        o1 = np.linspace(0, 1.5, 13)
        self.assertTrue(np.linalg.norm(np.sort(np.unique(union.coordinates()[:, 0])) - o1) < 1E-13)
        self.assertTrue(np.linalg.norm(np.sort(np.unique(union.coordinates()[:, 1])) - o1) < 1E-13)

        # Check mapping
        y = union.coordinates()
        for mesh in meshes:
            found = False
            for m_id, m_map in union.leafs:
                found = m_id == mesh.id()
                if found: break
            self.assertTrue(found)
                
            x = mesh.coordinates()
            self.assertTrue(np.linalg.norm(x - y[m_map]) < 1E-13)
