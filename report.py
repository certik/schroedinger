#! /usr/bin/env python
print "loading"

from tables import openFile
from pylab import plot, show, legend, grid, xlabel, ylabel, yscale

print "plotting"
h5file = openFile("report.h5")
table = h5file.root.schroed.single
x = table.col("DOF")
y = table.col("cpu_solve")
plot(x, y, linewidth=2, label="single")
plot(x, y, "kD")

for i in range(4):
    table = getattr(h5file.root.schroed, "eig%d" % i)
    x = table.col("DOF")
    y = table.col("cpu_solve")
    plot(x, y, linewidth=2, label="multimesh eigenvector %d" % i)
    plot(x, y, "rD")
    print i
h5file.close()

xlabel("DOF")
ylabel("cpu time [s]")
yscale("log", basey=10)
grid(True)
legend(loc="lower right")
show()
