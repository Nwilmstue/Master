#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver, util
import numpy, unittest, matplotlib, time, collections, itertools, functools, numbers, scipy, sys


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

## Build a constrain vector 
#   @ Parameter     ns = The namespace (Namspace class)
#   @ Parameter     ex1 = Amount of elements in x1 direction (integer)
#   @ Parameter     degree = The polynomial degree of the spline functions (integer)
#   @ Parameter     LayerResolution = The amount of elements a layer consists of (integer)
#   @ Parameter     n = The amount of Layers (integer)
#   @ Parameter     i = The incrementel layer wherefor the vector is build (integer)
#   @ return        cons = A constrainvector (object NaNVec)
def BuildConstrains(ns,ex1,LayerResolution,degree,i,n):
    lengthconsELEMENT = len(ns.basis)/(ex1+degree-1)
    Azero = (LayerResolution + degree - 1) * (n-i-1)    
    consELEMENT = util.NanVec(int(lengthconsELEMENT))
    consELEMENT[-Azero:] = 0
    
    cons = numpy.tile(consELEMENT,(ex1+degree-1))
    return cons

def testLHScons(ns,LayerResolution,domain,consEVAL,lhs, ischeme,degree):
    consEVALtest = domain[:,(LayerResolution + 1  )  :].project(0, onto=ns.basis, geometry=ns.x, ischeme=ischeme)
    numpy.testing.assert_array_equal(consEVAL,consEVALtest)
    print('Unittest constrains passed')
    lhs0test = domain.integral('(T)^2' @ ns, geometry=ns.x, degree=degree*2)
    lhs0test = solver.optimize('lhs', lhs0test, droptol=1e-15)
    lhstest = lhs0test
    print('Unittest lhs0 passed')

def main(
    nelems: 'number of elements' = 32,
    degree: 'polynomial degree' = 3,
    timescale: 'time scale (timproestep=timescale/nelems)' = 1.,
    tol: 'solver tolerance' = 1e-5,
    ndims: 'spatial dimension' = 1,
    endtime: 'end time' = 1,
    figures: 'create figures' = True,
    adiabatic: 'adiabatic conditions' = True,
    Utest: 'Unittest' = True,
    ):

    # Create namespace and add objects
    ns = function.Namespace()

    starttime1 = time.clock()
    solvertime = 0
    
    ## Variables values
    # Geometry and topology
    LayerResolution = 8                   #Amount of elements per layer
    n = 4                                 #Amount of layers
    ex1 = 96                              #Elements in x1 direction
    dx1 = 0.3                             #Distance in x1 direction in [m]
    dx2 = 0.1                             #Distance in x2 direction in [m]


    # Construct topology, geometry and basis
    ex2 = LayerResolution * n + 1   #Elements in x2 direction   
    verts1 = numpy.linspace(0, dx1, (ex1))
    verts2 = numpy.linspace(0, dx2, (ex2))
    domain, geom = mesh.rectilinear([verts1, verts2])
    

    ns.x = geom
    basistype = 'spline'
    ischeme = 'gauss4'

    knotm = BuildKnotMult(LayerResolution,n,degree)     
    ns.basis = domain.basis(basistype, degree=degree, knotmultiplicities=[None,knotm])

    #Define variables in the namespace
    x0,x1 =geom
    ns.T = 'basis_n ?lhs_n'
    ns.rho = 7820               # density
    ns.c = 250                  # Specific heat capicity
    ns.k = 500                  # Heat conductivity
    ns.L = 0                  # Latent heat generation
    ns.Tl = 5                # Heat generation 
    ns.Ts = 4
    ns.Ti = 0.                     #Initial Temperature
    ns.S = 4
    ns.Q = 0
    ns.h = 200.                  # Convective Heat Loss
    ns.Toutside = 0.	            # Temperature at infinity
    ns.qboundary = 0.
    ns.itime = 0 
    ns.ph = 0.5 * (function.tanh(ns.S * 2 / (ns.Tl - ns.Ts) * ( ns.T - (ns.Ts + ns.Tl) / 2 )) + 1)
    
    print('Time building domain and initializing =', time.clock()-starttime1)
    starttime = time.clock()    
    
    #Residual of entire Domain
    conductivitySTART0 = domain.integrate(ns.eval_ij('basis_i basis_j Q'), geometry=ns.x, ischeme=ischeme)

    #check if adiabatic
    if adiabatic:
        func = 'basis_n qboundary'
        print('Adibatic boundary conditions')
    else:
        func = '-basis_i h (Toutside - T)'
        print('convective boundary conditions')

    timestep  = 0.5
    maxiter   = 5
 
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
            conductivityUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i,k (k basis_j,k)'), geometry=ns.x, ischeme=ischeme)
            conductivityEVAL1   = conductivityUPDATE
            conductivityTOTAL   = conductivityEVAL1
            
            print('Constructing conductivity-matrix (K)  =', time.clock()-starttime)
            starttime = time.clock()
            
            loadUPDATE = domainUPDATE.boundary['left,right,bottom'].integrate(ns.eval_n(func), geometry=ns.x, ischeme=ischeme)
            loadEVAL1 = loadUPDATE
            
            print('Constructing load-vector (F) =', time.clock()-starttime)
            starttime = time.clock()

            
            #Time dependant factors
            capacitanceUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i basis_j rho c'), geometry=ns.x, ischeme=ischeme)
            #inertiaUPDATEb  = domainUPDATE.integrate(ns.eval_n('rho L basis_n ph'), geometry=ns.x, ischeme=ischeme)        
            capacitanceTOTAL    = capacitanceUPDATE#d + scipy.sparse.diags(inertiaUPDATEb)

            print('Constructing Capacitance-matrix (C) =', time.clock()-starttime)
            starttime = time.clock()
            
            # Initialization of the field
            lhs = numpy.zeros(len(ns.basis))
            
            print('Initialisation lhs =', time.clock()-starttime)
            starttime = time.clock()
            
            # Create constrains
            consEVAL = BuildConstrains(ns,ex1,LayerResolution,degree,i,n)

            if Utest:
                testLHScons(ns,LayerResolution,domain,consEVAL,lhs,ischeme,degree)
                consEVALtest = domain[:,(LayerResolution + 1  )  :].project(0, onto=ns.basis, geometry=ns.x, ischeme=ischeme)
                assert consEVALtest.all() == consEVAL.all()
                assert len(consEVALtest) == len(consEVAL)
                numpy.testing.assert_array_equal(consEVAL,consEVALtest)
                print('Unittest constrains passed')
                lhs0test = domain.integral('(T)^2' @ ns, geometry=ns.x, degree=degree*2)
                lhs0test = solver.optimize('lhs', lhs0test, droptol=1e-15)
                lhstest = lhs0test
                assert lhs.all() == lhstest.all()
                assert len(lhs) == len(lhstest)
                print('Unittest lhs0 passed')

            print('Constructing constrains =', time.clock()-starttime)
            starttime = time.clock()

        # Last layer
        elif i == n-1:
            #Update domain
            domainEVAL      = domain
            domainUPDATE    = domain[ : , i * LayerResolution : ]
            
            # Create residual
            conductivityUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i,k (k basis_j,k)'), geometry=ns.x, ischeme=ischeme)             
            conductivityEVAL1   = conductivityEVAL1 + conductivityUPDATE
            
            # Create Load
            loadUPDATE = domainUPDATE.boundary['left,right'].integrate(ns.eval_n(func), geometry=ns.x, ischeme=ischeme)
            loadEVAL1  = loadEVAL1 + loadUPDATE
            
            #Time dependant factors
            capacitanceUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i basis_j rho c'), geometry=ns.x, ischeme=ischeme)
            capacitanceTOTAL   = capacitanceTOTAL + capacitanceUPDATE
            
            #Creat constrains
            consEVAL = None
        
        # Rest of the layers
        else:
            domainEVAL      = domain[:, : (i+1)*LayerResolution]
            domainUPDATE    = domain[:,i*LayerResolution : (i+1)*LayerResolution]
            
            # Create residual
            conductivityUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i,k (k basis_j,k)'), geometry=ns.x, ischeme=ischeme)
            conductivityEVAL1   = conductivityEVAL1 + conductivityUPDATE
            
            # Create Load
            loadUPDATE = domainUPDATE.boundary['left,right'].integrate(ns.eval_n(func), geometry=ns.x, ischeme=ischeme)
            loadEVAL1  = loadEVAL1 + loadUPDATE
            
            #Time dependant factors
            capacitanceUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i basis_j rho c'), geometry=ns.x, ischeme=ischeme)
            capacitanceTOTAL   = capacitanceTOTAL + capacitanceUPDATE
            
            # Create constrains
            consEVAL = domain[:,((i+1)*LayerResolution + 1 )  :].project(0, onto=ns.basis, geometry=ns.x, ischeme=ischeme)               

        
        #Building matrices for the solver
        conductivityTOTAL  = conductivitySTART0 + conductivityEVAL1 

        B = (1 / timestep) * capacitanceTOTAL
        A = B + conductivityTOTAL        

        print('Constructing solvermatrices =', time.clock()-starttime)
        starttime = time.clock()
        
        for itime in range(0,maxiter):       
            
            # Create time dependant boundary condition
            ns.q = 1e6 * function.exp( -((x0-0.1-0.05*timestep*itime)**2)/(0.01**2) )
            loadEVAL2 = domainEVAL.boundary['top'].integrate(ns.eval_n('basis_n q'), geometry=ns.x, ischeme=ischeme)
            loadTOTAL = loadEVAL1 + loadEVAL2
            
            print('updating load with bc =', time.clock()-starttime)
            starttime = time.clock()
            
            #Solve the equation
            lhs = A.solve((loadTOTAL) + B.matvec(lhs), constrain=consEVAL)
            ns.lhs = lhs
            ns.T = ns.basis.dot(ns.lhs) 
            
            solvertime = solvertime + (time.clock()-starttime)
            print('solve for lhs =', time.clock()-starttime)
            starttime = time.clock()
            
            #Updating the solution
            nsplot = ns(lhs=lhs)
            
            if itime == 0 and i == 0:    
                ns.phend = nsplot.ph
            else:
                ns.phend = function.max(ns.phend,nsplot.ph)
            
            print('Update solution =', time.clock()-starttime)
            starttime = time.clock()
            
            if figures:
                #Plotting the variables
                points, Tvalues, PHvalues, PHFINAL = domain.elem_eval([geom, nsplot.T, nsplot.ph, ns.phend], ischeme='vertex1', separate=True)
                with plot.PyPlot('temperature {} {}'.format(i, itime), dpi = 300) as plt:
                    plt.title('Temperature at t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                    plt.mesh(points, Tvalues)
                    #im = plt.imshow(numpy.arange(200).reshape((1,3)))                 
                    plt.colorbar(orientation='horizontal')
                    plt.clim(0,10)
                    plt.ylabel('height in [m]')
                    plt.xlabel('Width in [m]')
                    plt.savefig('2. Figures/temperature {} {}'.format(i, itime),bbox_inches='tight')
        
                with plot.PyPlot('phase change {} {}'.format(i,itime), dpi = 300) as plt:
                    plt.title('Phase change at t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                    plt.mesh(points, PHvalues)
                    plt.colorbar(orientation='horizontal')
                    plt.clim(0, 1)
                    plt.ylabel('height in [m]')
                    plt.xlabel('Width in [m]')
                    plt.savefig('2. Figures/phase change {} {}'.format(i, itime),bbox_inches='tight')
        
                with plot.PyPlot('phase final {} {}'.format(i,itime), dpi = 300) as plt:
                    plt.title('Phase final at t={:5.2f}'.format(itime*timestep + i*maxiter*timestep))
                    plt.mesh(points, PHFINAL)
                    plt.ylabel('height in [m]')
                    plt.xlabel('Width in [m]')                    
                    plt.colorbar(orientation='horizontal')
                    plt.clim(0, 1)
                    plt.savefig('2. Figures/phase final {} {}'.format(i, itime),bbox_inches='tight')
                    
                print('Time for plotting =', time.clock()-starttime)
                starttime = time.clock()
                
                
                
    print('Time spend in solver=',solvertime)
    print('fraction spend solving=',solvertime/(time.clock()-starttime1))        
if __name__ == '__main__':
    starttime11 = time.clock() 
    
    cli.run(main)
    
    print('runtime =', time.clock()-starttime11)

    
