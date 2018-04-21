#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver
import numpy, unittest, matplotlib

# construct topology, geometry and basis

verts = numpy.linspace(-0.5**0.5, 0.5**0.5, 9)
domain, geom = mesh.rectilinear([verts, verts])
ns = function.Namespace()
ns.x = geom
ns.basis = domain.basis('spline', degree=2)
ns.u = 'basis_n ?lhs_n'
# construct matrix
A = domain.integral('-basis_n,i u_n,i' @ns, geometry=geom, ischeme='gauss3')

x0, x1 = geom
f = function.sin(x0) * function.exp(x1)

# construct dirichlet boundary constraints
cons = domain.boundary.project(f, onto=basis, geometry=geom, ischeme='gauss3')

# solve linear system
w = solver.solve('lhs',constrain=cons)
print('klaar')