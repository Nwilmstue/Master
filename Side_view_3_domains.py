#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver
import numpy, unittest, matplotlib, winsound, time, collections, itertools, functools, numbers

## Build a knotmultiplicity vector for every layer C0 continuous
#   @ Parameter     L = The amount of elements a layer consists of (integer)
#   @ Parameter     n = The amount of Layers (integer)
#   @ Parameter     degree = The polynomial degree of the spline functions (integer)
#   @ return        knotm = A knotmultiplicity list (list)

def BuildKnotMult(L,n,degree):
    knotm = [1] * ((L*n) + 1)
    for index in [0,-1]:
        knotm[index] = degree + 1   
    wbc = [L * (i + 1) for i in range(n - 1)]
    for index in wbc:
        knotm[index] = degree    
    return knotm



def main(
    nelems: 'number of elements' = 32,
    degree: 'polynomial degree' = 2,
    timescale: 'time scale (timestep=timescale/nelems)' = 1.,
    tol: 'solver tolerance' = 1e-5,
    ndims: 'spatial dimension' = 1,
    endtime: 'end time' = 1,
    figures: 'create figures' = True,
    adiabatic: 'adiabatic conditions' = True,
    ):

    starttime = time.clock()
    
    # Construct topology, geometry and basis
    LayerResolution = 50    #Amount of elements per layer
    n = 2                 #Amount of layers
    
    ex1 = 100    #Elements in x1 direction
    ex2 = LayerResolution * n + 1   #Elements in x2 direction
    height = 200e-6
    width = 3e-3
    laserspotsize = 70e-6
    laserpower = 1500
    scanspeed =0.1
    timestep  = 0.01
    absorptivity = 0.09
    Tambient = 20
    
    verts1 = numpy.linspace(0, width, (ex1))
    verts2 = numpy.linspace(0, height , (ex2))
    domain, geom = mesh.rectilinear([verts1, verts2])
    
    # Create namespace and add objects
    ns = function.Namespace()
    ns.x = geom
    basistype = 'spline'
    ischeme = 'gauss4'

    knotm = BuildKnotMult(LayerResolution,n,degree)    
    ns.basis = domain.basis(basistype, degree=degree, knotmultiplicities=[None,knotm])

    #Define variables
    x0,x1 =geom
    ns.T = 'basis_n ?lhs_n'
    ns.rho = 4510               # density
    ns.c = 520                  # Specific heat capicity
    ns.k = 16                  # Heat conductivity
    ns.L = 325000                  # Latent heat generation
    ns.Tl = 2000                 # Heat generation 
    ns.Ts = 1500
    #ns.Ti = 0.01                    #Initial Temperature
    ns.S = 4
    ns.Q = 0
    ns.h = 200.                  # Convective Heat Loss
    ns.Toutside = 0.	            # Temperature at infinity
    ns.qboundary = 0.
    ns.itime = 0 
    ns.ph = 0.5 * (function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 )) + 1)
    
    print('Time building domain and initializing =', time.clock()-starttime)
    
    #Residual of entire Domain
    residualSTART0 = domain.integral('basis_n Q' @ns, geometry=ns.x, degree=degree)
    
    #check if adiabatic
    if adiabatic:
        func = 'basis_n qboundary'
        print('Adibatic boundary conditions')
    else:
        func = '-basis_n h (Toutside - T)'
        print('convective boundary conditions')

    
    maxiter   = 10
    newtontol = 1e-6
    target    = 'lhs'
    target0   = '_thetamethod_target0'
    arguments = None
 
    print('First parameters =', time.clock()-starttime)
    starttime = time.clock()
    for i in range(n):
        
        ## Three phases (first layer, last layer, and the rest of the layers)
        #  First Layer
        if i ==0:
            #Update domain
            domainEVAL   = domain[: ,                   : (i + 1) * LayerResolution]
            domainUPDATE = domain[: , i*LayerResolution : (i + 1) * LayerResolution]
            
            # Create residual
            residualUPDATE  = domainUPDATE.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
            residualUPDATE += domainUPDATE.boundary['left,right,bottom'].integral(func @ ns, geometry=ns.x, degree=degree)
            residualEVAL1   = residualUPDATE
            
            print('Constructing residuals =', time.clock()-starttime)
            starttime = time.clock()
            
            #Time dependant factors
            inertiaUPDATE  = domainUPDATE.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
            inertiaUPDATE += domainUPDATE.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)        
            inertiaTOTAL   = inertiaUPDATE

            print('Constructing inertia =', time.clock()-starttime)
            starttime = time.clock()
            
            # Initialization of the field
            lhs = numpy.ones(len(ns.basis))#*Tambient
            
            print('Initialisation lhs =', time.clock()-starttime)
            starttime = time.clock()
            
            # Create constrains
            consEVAL = domain[:,(LayerResolution + 1  )  :].project(0., onto=ns.basis, geometry=ns.x, ischeme=ischeme)
#            consEVAL = consEVAL*Tambient
#            consEVAL = domain[:,(LayerResolution + 1  )  :].integral('T^2' @ ns, geometry=ns.x, degree=degree)               
#            consEVAL = solver.optimize('lhs', consEVAL, droptol=1e-15)

            print('Constructing constrains =', time.clock()-starttime)
            starttime = time.clock()
        
        # Last layer
        elif i == n-1:
            #Update domain
            domainEVAL      = domain
            domainUPDATE    = domain[ : , i * LayerResolution : ]
            
            # Create residual
            residualUPDATE  = domainUPDATE.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)  
            residualUPDATE += domainUPDATE.boundary['left,right'].integral( func @ ns, geometry=ns.x, degree=degree)
            residualEVAL1   = residualEVAL1 + residualUPDATE
            
            #Time dependant factors
            inertiaUPDATE  = domainUPDATE.integral('basis_n T rho c'  @ns, geometry=ns.x, degree=degree)
            inertiaUPDATE += domainUPDATE.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)             
            inertiaTOTAL   = inertiaTOTAL + inertiaUPDATE
            
            #Creat constrains
            consEVAL = None
        
        # Rest of the layers
        else:
            domainEVAL      = domain[:, : (i+1)*LayerResolution]
            domainUPDATE    = domain[:,i*LayerResolution : (i+1)*LayerResolution]
            
            # Create residual
            residualUPDATE  = domainUPDATE.integral('basis_n,i (k T_,i)' @ns, geometry=ns.x, degree=degree)
            residualUPDATE += domainUPDATE.boundary['left,right'].integral( func @ ns, geometry=ns.x, degree=degree)
            residualEVAL1   = residualEVAL1 + residualUPDATE
            
            #Time dependant factors
            inertiaUPDATE  = domainUPDATE.integral('basis_n T rho c' @ns, geometry=ns.x, degree=degree)
            inertiaUPDATE += domainUPDATE.integral('rho L basis_n ph' @ns, geometry=ns.x, degree=degree)          
            inertiaTOTAL   = inertiaTOTAL + inertiaUPDATE
            
            # Create constrains
            consEVAL = domain[:,((i+1)*LayerResolution + 1 )  :].integral('T^2' @ ns, geometry=ns.x, degree=degree)               
            consEVAL = solver.optimize('lhs', consEVAL, droptol=1e-15)

        # Solver part that doesn't need to be calculated all the time
        res1 = - inertiaTOTAL / timestep 
        
        print('Creating res 1 =', time.clock()-starttime)
        starttime = time.clock()
        
        for itime in range(0,maxiter):       
            
            # Create time dependant boundary condition
            ns.q = absorptivity*laserpower/laserspotsize/1.77/timestep * function.exp( -((x0-0.3*width-0.3*width/maxiter*itime)**2)/(0.5*laserspotsize**2) )
            residualEVAL2 = domainEVAL.boundary['top'].integral('-basis_n q' @ ns, geometry=ns.x, degree=degree)
            
            #Updating the residual
            residualTOTAL  = residualSTART0 + residualEVAL1 + residualEVAL2 

            print('Updating residual with bc =', time.clock()-starttime)
            starttime = time.clock()

            # Non-linear solver (implicit backwards euler)        
            res0 = residualTOTAL + inertiaTOTAL / timestep
            res = res0 + res1.replace({target: function.Argument(target0, lhs.shape)})
            jac = res.derivative(target)
            print('prepare solver =', time.clock()-starttime)
            starttime = time.clock()
            
            lhs = solver.solve(solver.newton(target, residual=res, jacobian=jac, lhs0=lhs, constrain=consEVAL, arguments=collections.ChainMap(arguments or {}, {target0: lhs})), tol=newtontol)
            
            print('Solve for lhs =', time.clock()-starttime)
            starttime = time.clock()
            
            #Updating the solution
            nsplot = ns(lhs=lhs)
            
            if itime == 0 and i == 0:    
                ns.phend = nsplot.ph
            else:
                ns.phend = function.max(ns.phend,nsplot.ph)
            
            print('Update solution =', time.clock()-starttime)
            print('magnification factor of q = ',absorptivity*laserpower/laserspotsize/1.77)
            starttime = time.clock()
            
            #Plotting the variables
            points, Tvalues, PHvalues, PHFINAL = domain.elem_eval([geom, nsplot.T, nsplot.ph, ns.phend], ischeme='vertex1', separate=True)
            with plot.PyPlot('2. figures/temperature {} {}'.format(i, itime), dpi = 300) as plt:
                plt.title('t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                plt.mesh(points, Tvalues)
                plt.colorbar()
                plt.clim(0, 1000)
                #plt.show()
                
    
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
                
            print('Time for plotting =', time.clock()-starttime)

if __name__ == '__main__':
    starttime = time.clock()
    
    main()
    
    print('runtime =', time.clock()-starttime)
    winsound.Beep(2500, 1000)
    time.sleep(1)
    winsound.Beep(2500, 1000)
    
