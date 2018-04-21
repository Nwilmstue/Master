#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver
import numpy, unittest, matplotlib, winsound, time

def main0(
    nelems: 'number of elements' = 16,
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
    ns.Tl = 100
    ns.Ts = 75
    ns.S = 1
    ns.qstart = 0.
    ns.q = function.exp( -((geom-0.5)**2).sum(0)/(0.1**2) ) 
    ns.ph = 0.5 * function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 ))

    #construct residual + implementation of neumann conditions
    res = domain.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
    res += domain.boundary['left,right,bottom'].integral('basis_n qstart' @ ns, geometry=ns.x, degree=degree) 

    #Time dependant factors
    inertia = domain.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
    inertia += domain.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)
    
    # construct dirichlet boundary constraints
    cons = domain.boundary['top'].integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
    cons = solver.optimize('lhs', cons, droptol=1e-15) + 20
    
    #Create initial condition    
    ns.l = function.exp( -((geom-0.5)**2).sum(0)/(0.5**2) )
    lhs0 = domain.integral('(T - l)^2' @ ns, geometry=ns.x, degree=degree*2)
    lhs0 = solver.optimize('lhs', lhs0, droptol=1e-15)
    lhs0 = lhs0*200
    print(len(lhs0))
    TotalT = numpy.array((10,256))
    timestep = 1
    for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0, constrain=cons, newtontol=tol)):
        # plot       
        nsplot = ns(lhs=lhs)
        points, colors = domain.elem_eval([geom, nsplot.T], ischeme='vertex1', separate=True)
        print(len(colors))
        print(colors[:,0])
        TotalT[:,itime] = colors[:,0]
        if itime == 10:
            break
    return TotalT

def main1(
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
    ns.L = 1000
    ns.Tl = 100
    ns.Ts = 75
    ns.S = 1
    ns.qstart = 0.
    ns.q = function.exp( -((geom-0.5)**2).sum(0)/(0.1**2) ) 
    ns.ph = 0.5 * function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 ))

    #construct residual + implementation of neumann conditions
    res = domain.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
    res += domain.boundary['left,right,bottom'].integral('basis_n qstart' @ ns, geometry=ns.x, degree=degree) 

    #Time dependant factors
    inertia = domain.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
    inertia += domain.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)
    
    # construct dirichlet boundary constraints
    cons = domain.boundary['top'].integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
    cons = solver.optimize('lhs', cons, droptol=1e-15) + 20
    
    #Create initial condition    
    ns.l = function.exp( -((geom-0.5)**2).sum(0)/(0.5**2) )
    lhs0 = domain.integral('(T - l)^2' @ ns, geometry=ns.x, degree=degree*2)
    lhs0 = solver.optimize('lhs', lhs0, droptol=1e-15)
    lhs0 = lhs0*200
    
    TotalT = numpy.array((10,1))
    timestep = 1
    for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0, constrain=cons, newtontol=tol)):
        # plot       
        nsplot = ns(lhs=lhs)
        TotalT[:, itime] = nsplot.T 
        print(TotalT)
        if itime == 10:
            break
    return TotalT

if __name__ == '__main__':
    TnoL = main0()
    TL = main1()
    
    print(TL - TnoL)
    
    winsound.Beep(2500, 1000)
    time.sleep(1)
    winsound.Beep(2500, 1000)
