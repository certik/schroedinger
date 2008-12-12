#! /usr/bin/env python
print "loading"

from tables import openFile
from pylab import plot, show, legend, grid, xlabel, ylabel, yscale

print "plotting"
h5file = openFile("report.h5")
table = h5file.root.schroed.single
x = table.col("DOF")
eig_errors = table.col("eig_errors")
for i in range(len(eig_errors[0])):
    eig = [a[i] for a in eig_errors]
    plot(x, eig, linewidth=2, label="eigenvector %d" % i)
    plot(x, eig, "kD")
    print i

for i in range(4):
    table = getattr(h5file.root.schroed, "eig%d" % i)
    x = table.col("DOF")
    eig_errors = table.col("eig_errors")
    eig = [a[i] for a in eig_errors]
    plot(x, eig, linewidth=2, label="multimesh eigenvector %d" % i)
    plot(x, eig, "rD")
    print i
h5file.close()

xlabel("DOF")
ylabel("error in %")
yscale("log", basey=10)
grid(True)
legend()
show()
