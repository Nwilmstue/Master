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
    LayerResolution = 5    #Amount of elements per layer
    n = 5                   #Amount of layers
    
    ex1 = 60    #Elements in x1 direction
    ex2 = LayerResolution * n + 1   #Elements in x2 direction
    verts1 = numpy.linspace(0, 0.3, (ex1))
    verts2 = numpy.linspace(0, 0.1, (ex2))
    domain, geom = mesh.rectilinear([verts1, verts2])
    

    # Create namespace and add objects
    ns = function.Namespace()
    ns.x = geom
    basistype = 'spline'
    knotm = [1] * ex2
    wbc = [LayerResolution * (i + 1) - 1  for i in range(n - 1)]

    for index in [5,10,15,20]:
        knotm[index] = degree + 1
#    print(knotm)                 
#    print(knotm)
#    knotm[0] = degree + 1
#    knotm[-1] = degree + 1
    print(knotm)        
    starttime2 = time.clock()
 
           
    ns.basis = domain.basis(basistype, degree=degree, knotmultiplicities=[None,knotm])
    print('runtime =', time.clock()-starttime2)  
#    print(ns.basis)
#    ns.basis.append(domain[:,5:].basis('discont', degree=degree))
#    ns.cbasis, ns.dcbasis = function.chain([domain[:, : 4].basis('spline', degree=2), domain[: , 5 :].basis('discont', degree=2)])
#    ns.basis_n =  'cbasis_n + dcbasis_n'
#    
#    ns.cbasis, ns.dcbasis1, ns.dcbasis2 = function.chain([domain.basis('spline', degree=2,removedofs=(5,None)), domain.basis('discont', degree=2, removedofs=((0,4),None)])
#    ns.basis_n =  'cbasis_n + dcbasis_n'
    
    #Define variables
    x0,x1 =geom
    ns.T = 'basis_n ?lhs_n'
    ns.rho = 7820               # density
    ns.c = 250                  # Specific heat capicity
    ns.k = 500                  # Heat conductivity
    ns.L = 100                  # Latent heat generation
    ns.Tl = 7.5                 # Heat generation 
    ns.Ts = 5
    ns.S = 4
    ns.Q = 0
    ns.h = 200.                  # Convective Heat Loss
    ns.Toutside = 0.	            # Temperature at infinity
    ns.qboundary = 0.
    ns.itime = 0 
    ns.ph = 0.5 * (function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 )) + 1)
    
    #Residual of entire Domain
    residualSTART0 = domain.integral('basis_n Q' @ns, geometry=ns.x, degree=degree)
    
    for i in range(n):
        
        domainEVAL = domain[:, : (i+1)*LayerResolution]
         
        #construct residual + implementation of neumann conditions
        residualEVAL1 = domainEVAL.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
        
        #Adiabatic surroundings
        #residualEVAL1 += domainEVAL.boundary['left,right,bottom'].integral('basis_n qboundary' @ ns, geometry=ns.x, degree=degree) 
        #Convectional Energy loss
        residualEVAL1 += domainEVAL.boundary['left,right,bottom'].integral('-basis_n h (Toutside - T)' @ ns, geometry=ns.x, degree=degree) 
        
        #res += domain.integral('-1 basis_n Q' @ns, geometry=ns.x, degree=degree)
        #Time dependant factors
        inertiaEVAL = domainEVAL.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
        inertiaEVAL += domainEVAL.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)
        inertiaTOTAL = inertiaEVAL
        
        # construct dirichlet boundary constraints
        #cons = None
        #cons = domain.boundary.integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
        #cons = solver.optimize('lhs', cons, droptol=1e-15) + 20
    
        #Create initial condition    
        if i == 0:
            lhs0 = domain.integral('(T)^2' @ ns, geometry=ns.x, degree=degree*2)
            lhs0 = solver.optimize('lhs', lhs0, droptol=1e-15)
            lhs = lhs0
            
            consEVAL = domain[:,(LayerResolution )  :].integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
            consEVAL = solver.optimize('lhs', consEVAL, droptol=1e-15)
        elif i+1 == n:
            consEVAL = None
        else:
            consEVAL = domain[:,((i+1)*LayerResolution )  :].integral('T^2' @ ns, geometry=ns.x, degree=degree*2)               #continu bc creeren 
            consEVAL = solver.optimize('lhs', consEVAL, droptol=1e-15)
    
        timestep = 0.5
        maxiter = 5
        newtontol= 1e-4
        
        target = 'lhs'
        target0 = '_thetamethod_target0'
        arguments = None
        for itime in range(0,maxiter):       
            
            # Create time dependant boundary condition
            ns.q = 1e6 * function.exp( -((x0-0.1-0.05*timestep*itime)**2)/(0.01**2) )\
            #Energy added by the laser
            residualEVAL2 = domainEVAL.boundary['top'].integral('-basis_n q' @ ns, geometry=ns.x, degree=degree)   
            residualTOTAL  = residualSTART0 + residualEVAL1 + residualEVAL2 
    
            # Non-linear solver (implicit backwards euler)        
            res0 = residualTOTAL + inertiaTOTAL / timestep
            res1 = - inertiaTOTAL / timestep
            res = res0 + res1.replace({target: function.Argument(target0, lhs.shape)})
            jac = res.derivative(target)
            lhs = solver.solve(solver.newton(target, residual=res, jacobian=jac, lhs0=lhs, constrain=consEVAL, arguments=collections.ChainMap(arguments or {}, {target0: lhs})), tol=newtontol)
    
            #Updating the solution
            nsplot = ns(lhs=lhs)
            #Plotting of the variables 
            
            if itime == 0 and i == 0:    
                ns.phend = nsplot.ph
            else:
                ns.phend = function.max(ns.phend,nsplot.ph)
            
            #Plotting the variables
            points, Tvalues, PHvalues, PHFINAL = domain.elem_eval([geom, nsplot.T, nsplot.ph, ns.phend], ischeme='vertex1', separate=True)
            with plot.PyPlot('2. figures/temperature {} {}'.format(i, itime), dpi = 300) as plt:
                plt.title('t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                plt.mesh(points, Tvalues)
                plt.colorbar()
                plt.clim(0,10)
    
            with plot.PyPlot('2. figures/phase change {} {}'.format(i,itime), dpi = 300) as plt:
                plt.title('t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                plt.mesh(points, PHvalues)
                plt.colorbar()
                plt.clim(0, 1)
    
            with plot.PyPlot('2. figures/phase final {} {}'.format(i,itime), dpi = 300) as plt:
                plt.title('t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                plt.mesh(points, PHFINAL)
                plt.colorbar()
                plt.clim(0, 1)


if __name__ == '__main__':
    starttime = time.clock()
    main()
    print('runtime =', time.clock()-starttime)
    winsound.Beep(2500, 1000)
    time.sleep(1)
    winsound.Beep(2500, 1000)
    
