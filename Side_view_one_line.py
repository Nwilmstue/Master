#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver
import numpy, unittest, matplotlib, winsound, time, collections, itertools, functools, numbers

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
    verts1 = numpy.linspace(0, 0.1, (200))
    verts2 = numpy.linspace(0, 0.005, (10))
    domain, geom = mesh.rectilinear([verts1, verts2])

   
    # Create namespace and add objects
    ns = function.Namespace()
    ns.x = geom
    basistype = 'spline'
    ns.basis = domain.basis(basistype, degree=degree)
    
    #Define variables
    x0,x1 =geom
    ns.T = 'basis_n ?lhs_n'
    ns.rho = 7820
    ns.c = 250
    ns.k = 500
    ns.L = 100
    ns.Tl = 7.5
    ns.Ts = 5
    ns.S = 4
    ns.Q = 0
    ns.qboundary = 0.
    ns.itime = 0 
    ns.ph = 0.5 * (function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 )) + 1)
#    ns.phend = ns.ph
#    ns.phend = function.max(ns.ph,ns.phend)


    #construct residual + implementation of neumann conditions
    residual1 = domain.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
    residual1 += domain.boundary['left,right,bottom'].integral('basis_n qboundary' @ ns, geometry=ns.x, degree=degree) 
   
    #res += domain.integral('-1 basis_n Q' @ns, geometry=ns.x, degree=degree)
    #Time dependant factors
    inertia = domain.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
    inertia += domain.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)
    
    # construct dirichlet boundary constraints
    cons = None
    #cons = domain.boundary.integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
    #cons = solver.optimize('lhs', cons, droptol=1e-15) + 20

    #Create initial condition    
    ns.l = function.exp( -((geom-0.5)**2).sum(0)/(0.5**2) )
    lhs0 = domain.integral('(T)^2' @ ns, geometry=ns.x, degree=degree*2)
    lhs0 = solver.optimize('lhs', lhs0, droptol=1e-15)
    lhs = lhs0
    

    timestep = 0.5
    maxiter = 20
    newtontol= 1e-4
    
    target = 'lhs'
    target0 = '_thetamethod_target0'
    arguments = None
    for itime in range(0,maxiter):       
        
        # Create time dependant boundary condition
        ns.q = 1e5 * function.exp( -((x0-0.05-0.005*timestep*itime)**2)/(0.01**2) ) 
        residual2 = domain.boundary['top'].integral('-basis_n q' @ ns, geometry=ns.x, degree=degree)   
        residual  = residual1 + residual2 

        # Non-linear solver (implicit backwards euler)        
        res0 = residual + inertia / timestep
        res1 = - inertia / timestep
        res = res0 + res1.replace({target: function.Argument(target0, lhs.shape)})
        jac = res.derivative(target)
        lhs = solver.solve(solver.newton(target, residual=res, jacobian=jac, lhs0=lhs, constrain=cons, arguments=collections.ChainMap(arguments or {}, {target0: lhs})), tol=newtontol)

        #Updating the solution
        nsplot = ns(lhs=lhs)
        #Plotting of the variables      
        points, Tvalues, phvalues = domain.elem_eval([geom, nsplot.T, nsplot.ph], ischeme='vertex1', separate=True)
        with plot.PyPlot('2. figures/temperature {:d}'.format(itime)) as plt:
            plt.title('t={:5.2f}'.format(itime*timestep))
            plt.mesh(points, Tvalues)
            plt.colorbar()
            plt.clim(0,10)
            plt.show()
              
        points, PHvalues = domain.elem_eval([geom, nsplot.ph], ischeme='vertex1', separate=True)
        with plot.PyPlot('2. figures/phase change {:d}'.format(itime)) as plt:
            plt.title('t={:5.2f}'.format(itime*timestep))
            plt.mesh(points, PHvalues)
            plt.colorbar()
            plt.clim(0, 1)

#        print(time.clock())
        if itime == 0:    
            ns.phend = nsplot.ph
        else:
            ns.phend = function.add(ns.phend,function.abs(function.subtract(nsplot.ph,ns.phend)))
#        print(ns.phend)
#        
        
        if itime == 0:    
            phend = PHvalues
        else:
            phend = numpy.add(phend,numpy.abs(numpy.subtract(PHvalues,phend)))
            
#        points, values = domain.elem_eval([geom, ns.phend], ischeme='vertex1', separate=True)
        with plot.PyPlot('2. figures/phase final {:d}'.format(itime)) as plt:
            plt.title('t={:5.2f}'.format(itime*timestep))
            plt.mesh(points, phend)
            plt.colorbar()
            plt.clim(0, 1)
            plt.show()
        

if __name__ == '__main__':
    starttime = time.clock()
    main()
    print('runtime =', time.clock()-starttime)
    winsound.Beep(2500, 1000)
    time.sleep(1)
    winsound.Beep(2500, 1000)
    
