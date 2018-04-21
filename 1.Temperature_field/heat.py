#! /usr/bin/env python3

from nutils import cli, mesh, function, plot, log, solver
import numpy

def main(alpha=0.01, nelems=64, degree=2, dt=0.01, tend=1):

    # construct topology, geometry and basis
    verts = numpy.linspace(0, 1, nelems+1)
    domain, geom = mesh.rectilinear([verts, verts])
    basis = domain.basis('spline', degree=degree)
    ischeme = 'gauss4'
    print(basis)
    exit()
    
    # construct matrices
    A = 1/dt * basis['i'] * basis['j'] + alpha / 2 * basis['i,k'] * basis['j,k']
    B = 1/dt * basis['i'] * basis['j'] - alpha / 2 * basis['i,k'] * basis['j,k']
    A, B = domain.integrate(
        [A, B], geometry=geom, ischeme=ischeme)

    # construct dirichlet boundary constraints
    cons = domain.boundary.project(0, onto=basis, geometry=geom, ischeme=ischeme)

    # construct initial condition
    x0, x1 = geom
    l = function.max(
        # level set of a square centred at (0.3,0.4)
        0.15 - (abs(0.3 - x0) + abs(0.4 - x1)),
        # level set of a circle centred at (0.7,0.6)
        0.15 - ((0.7 - x0)**2 + (0.6 - x1)**2)**0.5,
    )
    # smooth heaviside of level set
    u0 = 0.5 + 0.5*function.tanh(nelems/2*l)

    for n in log.range('timestep', round(tend/dt) + 1):

        if n == 0:
            # project initial condition on `basis`
            w = domain.project(
                u0, onto=basis, geometry=geom,
                ischeme=ischeme)
        else:
            # time step
            w = A.solve(B.matvec(w), constrain=cons)

        # construct solution
        u = basis.dot(w)

        # plot
        points, colors = domain.elem_eval(
            [geom, u], ischeme='bezier3', separate=True)
        with plot.PyPlot('temperature') as plt:
            plt.title('t={:5.2f}'.format(n*dt))
            plt.mesh(points, colors)
            plt.colorbar()
            plt.clim(0, 1)

main()