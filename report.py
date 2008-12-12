#! /usr/bin/env python

from tables import openFile
from pylab import plot, show, legend, grid, xlabel, ylabel

print "plotting"
h5file = openFile("report.h5")
table = h5file.root.schroed.sim
#for iter in table:
#    print "n:", iter["n"]
#    print "cpu_solve:", iter["cpu_solve"]
#    print "cpu_solve_reference:", iter["cpu_solve_reference"]
#    print "DOF:", iter["DOF"], iter["DOF_reference"]
#    print "errors:", iter["eig_errors"]
x = table.col("DOF")
eig_errors = table.col("eig_errors")
for i in range(len(eig_errors[0])):
    eig = [a[i] for a in eig_errors]
    plot(x, eig, linewidth=2, label="eigenvector %d" % i)
    print i
h5file.close()

xlabel("DOF")
ylabel("error in %")
grid(True)
legend()
show()
