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
errs = table.col("eig_errors")
for i in range(4):
    y = errs[:, i]
    plot(x, y, linewidth=2, label="eig %d" % i)
    plot(x, y, "kD")

h5file.close()

xlabel("DOF")
ylabel("error [%]")
yscale("log", basey=10)
title("adapt to one eigenvector per iteration")
grid(True)
legend(loc="upper right")
#show()
savefig("single-single.png")
