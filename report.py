#! /usr/bin/env python
print "loading"

from tables import openFile
from pylab import plot, show, legend, grid, xlabel, ylabel, yscale, \
        savefig, title
import sys

print "plotting"
if len(sys.argv) == 2:
    sim_name = sys.argv[1]
else:
    sim_name = "sim"
h5file = openFile("report.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "b-", linewidth=2, label="h-FEM total error")
plot(x, y, "bD")
h5file.close()
h5file = openFile("report-august-hp.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "g-", linewidth=2, label="hp-FEM total error")
#plot(x, y, "gD")
h5file.close()

xlabel("DOF")
ylabel("error [%]")
yscale("log", basey=10)
title("adapt to one eigenvector per iteration")
grid(True)
legend(loc="upper right")
show()
#savefig("single-single.png")
