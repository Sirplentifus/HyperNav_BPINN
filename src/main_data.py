from utility import set_directory
from utility import switch_problem
from setup import AnalyticalData

set_directory()
AnalyticalData(switch_problem("Laplace1D_cos"), do_plots = True, test_only = False, save_plot = True, is_main = True).show_plot()
#AnalyticalData(switch_problem("Laplace2D_cos"), do_plots = True, test_only = True, save_plot = True, is_main = True).show_plot()
