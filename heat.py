#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver
import numpy, unittest, matplotlib, winsound, time

def main(
    nelems: 'number of elements' = 32,
    degree: 'polynomial degree' = 2,
    timescale: 'time scale (timestep=timescale/nelems)' = 1.,
    tol: 'solver tolerance' = 1e-5,
    ndims: 'spatial dimension' = 1,
    endtime: 'end time' = 1,
    figures: 'create figures' = True,
    ):

    # Construct topology, geometry and basis
    verts = numpy.linspace(0, 1, nelems+1)
    domain, geom = mesh.rectilinear([verts, verts])

    # Create namespace and add objects
    ns = function.Namespace()
    ns.x = geom
    basistype = 'spline'
    ns.basis = domain.basis(basistype, degree=degree)
    
    #Define variables
    ns.T = 'basis_n ?lhs_n'
    ns.rho = 7820
    ns.c = 250
    ns.k = 500
    ns.L = 0
    ns.Tl = 55
    ns.Ts = 50
    ns.S = 1
    ns.qstart = 0.
    ns.q = function.exp( -((geom-0.5)**2).sum(0)/(0.1**2) ) 
    ns.ph = 0.5 * function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 ))

    #construct residual + implementation of neumann conditions
    res = domain.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
    res += domain.boundary['left,right,top,bottom'].integral('basis_n qstart' @ ns, geometry=ns.x, degree=degree) 

    #Time dependant factors
    inertia = domain.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
    inertia += domain.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)
    
    # construct dirichlet boundary constraints
    cons = domain.boundary['top'].integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
    cons = solver.optimize('lhs', cons, droptol=1e-15)  

    #Create initial condition    
    ns.l = function.exp( -((geom-0.5)**2).sum(0)/(0.5**2) )
    lhs0 = domain.integral('(T - l)^2' @ ns, geometry=ns.x, degree=degree*2)
    lhs0 = solver.optimize('lhs', lhs0, droptol=1e-15)
    lhs0 = lhs0*200

    timestep =1.5
    for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0,constrain=cons, newtontol=tol)):
#    for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0, constrain=cons, newtontol=tol)):
        # plot       
        nsplot = ns(lhs=lhs)
        
##        #update deze automatisch?        
#        ns.q = function.exp( -((geom-0.5-0.025*itime)**2).sum(0)/(0.1**2) )    
#        res = domain.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree*2)
#        res += domain.integral('basis_n q' @ ns, geometry=ns.x, degree=degree*2) 
##        
        
        points, colors = domain.elem_eval([geom, nsplot.T], ischeme='vertex1', separate=True)
        with plot.PyPlot('temperature') as plt:
            plt.title('t={:5.2f}'.format(itime*timestep))
            plt.mesh(points, colors)
            plt.colorbar()
            plt.clim(0,numpy.max(lhs0))
            
        
        points, colors = domain.elem_eval([geom, nsplot.ph], ischeme='vertex1', separate=True)
        with plot.PyPlot('phase change') as plt:
            plt.title('t={:5.2f}'.format(itime*timestep))
            plt.mesh(points, colors)
            plt.colorbar()
            plt.clim(0, 1)
            
        #Maxiumum iterations    
        if itime == 100:
            break

if __name__ == '__main__':
    main()
    winsound.Beep(2500, 1000)
    time.sleep(1)
    winsound.Beep(2500, 1000)
