#! /usr/bin/env python
print "loading"

from tables import openFile
from pylab import plot, show, legend, grid, xlabel, ylabel, yscale

print "plotting"
h5file = openFile("report.h5")
table = h5file.root.sim.iterations
x = table.col("DOF")
errs = table.col("eig_errors")
for i in range(4):
    y = errs[:, i]
    plot(x, y, linewidth=2, label="single, eig %d" % i)
    plot(x, y, "kD")

h5file.close()

xlabel("DOF")
ylabel("error [%]")
yscale("log", basey=10)
grid(True)
legend(loc="upper right")
show()
