#! /usr/bin/env python

from tables import openFile

h5file = openFile("report.h5")
table = h5file.root.schroed.sim
for iter in table:
    print "n:", iter["n"]
    print "cpu_solve:", iter["cpu_solve"]
    print "cpu_solve_reference:", iter["cpu_solve_reference"]
    print "DOF:", iter["DOF"], iter["DOF_reference"]
h5file.close()
