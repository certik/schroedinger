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
h5file = openFile("report-h.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "b-", linewidth=2, label="h-FEM (linear) total error")
plot(x, y, "bD")
h5file.close()

h5file = openFile("report-h2-c.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "c-", linewidth=2, label="h-FEM (quadratic, thr=0.1) total error")
plot(x, y, "cD")
h5file.close()

h5file = openFile("report-h2.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "m-", linewidth=2, label="h-FEM (quadratic, thr=0.3) total error")
plot(x, y, "mD")
h5file.close()


h5file = openFile("report-h2-b.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "y-", linewidth=2, label="h-FEM (quadratic, thr=0.7) total error")
plot(x, y, "yD")
h5file.close()


h5file = openFile("report-hp.h5")
table = getattr(h5file.root, sim_name).iterations
x = table.col("DOF")
y = table.col("total_error")
plot(x, y, "g-", linewidth=2, label="hp-FEM total error")
plot(x, y, "gD")
h5file.close()

xlabel("DOF")
ylabel("error [%]")
yscale("log", basey=10)
title("adapting to sum of all eigenvectors per iteration")
grid(True)
legend(loc="upper right")
#show()
savefig("adapt-all.png")
