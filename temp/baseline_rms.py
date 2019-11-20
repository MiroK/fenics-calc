from xcalc.timeseries import PVDTempSeries
from xcalc.timeseries import clip
from xcalc.interpreter import Eval
from xcalc.operators import Mean, RMS, STD
from dolfin import *

print("Hi there!")

# list_path_to_data = ["/home/jrlab/Desktop/Git/article_ann_optimal_control_Nature/data_for_plots/Simu40000/abs_of_mean_copy/results",
#                      "/home/jrlab/Desktop/Git/article_ann_optimal_control_Nature/data_for_plots/Simu40000/angle0_10/results",
#                      "/home/jrlab/Desktop/Git/article_ann_optimal_control_Nature/data_for_plots/Simu40000/base_line_10/results"]

list_path_to_data = ["/home/fenics/shared/mesh_refinement/short_sim_res/tests/results"]


for what in ['velocity', 'pressure']:
    for path_to_data in list_path_to_data:

        print("snippet applied on {} for path {}".format(what, path_to_data))

        if what == 'pressure':
            data_to_analyze = PVDTempSeries('%s/p.pvd' % path_to_data, FiniteElement('Lagrange', triangle, 1))

        elif what == 'velocity':
            # I chose to look here at velocity magnitude, pod of velocity field should work as well
            velocity = PVDTempSeries('%s/u.pvd' % path_to_data, VectorElement('Lagrange', triangle, 1))
            data_to_analyze = Eval(sqrt(inner(velocity, velocity)))

        first_dump = 40
        last_dump = 80

        clipped_data_to_analyze = clip(data_to_analyze, first_dump, last_dump)

        computed_mean = Mean(clipped_data_to_analyze)

        out = File('%s/mean_rms/mean_%s.pvd' % (path_to_data, what))
        computed_mean.rename('f', '0')
        out << (computed_mean, 0.)

        computed_rms = RMS(clipped_data_to_analyze)

        out = File('%s/mean_rms/rms_%s.pvd' % (path_to_data, what))
        computed_rms.rename('f', '0')
        out << (computed_rms, 0.)

        computed_std = STD(clipped_data_to_analyze)

        out = File('%s/mean_rms/std_%s.pvd' % (path_to_data, what))
        computed_std.rename('f', '0')
        out << (computed_std, 0.)
