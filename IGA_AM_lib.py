#! /usr/bin/env python3

## @package IGA_AM_lib
# This is the library for the IGA and AM combined functions. 

from nutils import *
from pre_post_lib import * 
import numpy, unittest, time, collections, itertools, functools, numbers, scipy, sys,os,datetime, shutil, matplotlib

## Build a constrain vector
#   @param     ns = The namespace (Namspace class)
#   @param     prop =  All the properties of the simulation(Object properties)
#   @param     i = The incrementel layer wherefor the vector is build (integer)
#   @return    cons = A constrainvector (object NaNVec)
def BuildConstrains(ns,prop,i):
    lengthconsELEMENT = len(ns.basis)/(prop.ex1+prop.degree-1)
    Azero = (prop.LayerResolution + prop.degree - 1) * (prop.n-i-1)
    consELEMENT = util.NanVec(int(lengthconsELEMENT))
    if Azero != 0:
        consELEMENT[-Azero:] = prop.Ti
    cons = numpy.tile(consELEMENT,(prop.ex1+prop.degree-1))
    return cons

## Build a knotmultiplicity vector for every layer C0 continuous
#   @param         prop =  All the properties of the simulation(Object properties)
#   @return        knotm = A knotmultiplicity list (list)
def BuildKnotMult(prop):
    knotm = [1] * ((prop.LayerResolution*prop.n) + 1)
    for index in [0,-1]:
        knotm[index] = prop.degree + 1
    wbc = [prop.LayerResolution * (i + 1) for i in range(prop.n - 1)]
    for index in wbc:
        knotm[index] = prop.degree
    return knotm

## Test if the initial conditions and constrains are correct.
#   @param         ns = The namespace (Namspace class)
#   @param         prop = All the properties of the simulation (Object properties)
#   @param         domain = The domain of the problem (topology class)
#   @param         consEVAL = The intital constraints (NANVEC class)
#   @param         lhs = The intial left hand side (numpy array)
def testLHScons(ns,prop,domain,consEVAL,lhs):
    consEVALtest = domain[:,(prop.LayerResolution + 1  )  :].project(0, onto=ns.basis, geometry=ns.x, ischeme=prop.ischeme)
    numpy.testing.assert_array_equal(consEVAL,consEVALtest)
    print('Unittest constrains passed')
    lhs0test = domain.integral('(T)^2' @ ns, geometry=ns.x, degree=prop.degree)
    lhs0test = solver.optimize('lhs', lhs0test, droptol=1e-15)
    numpy.testing.assert_array_equal(lhs,lhs0test)
    print('Unittest lhs0 passed')

## Easy timingfunctions
class timeprint():
    ## Intitialize the time at the beginning
    def __init__(self):
        self.time1 = time.time()
        self.tottime = time.time()
    
    ## Print used time until last timeprint   
    def timeprint(self,statement):
        print(statement + '\t in' , round(time.time() - self.time1,1), 'second(s)'  )
        self.time1 = time.time()
## Solidification process   
# The entire solidification part of the simulation is calculated in this class#
class solidification():
    ## Initialization of the soldification simulation by temperature @brief 
    # In this part the Topology, geometry and basis are constructed, the variables are defined, the solidification function is defined, 
    # The conductivity, constrains, initial solution, and load are initialized and apropriate boundary functions are chosen     
    # @param    prop = properties of the complete simulation
    def __init__(self,prop):
        ## The general properties of the simulation
        self.prop         = prop
        ## Namespace containing the basis and variables used in the weak formulations
        self.ns           = function.Namespace()        
        
        # Construct topology, geometry and basis for solidification process
        ## The amount of elements in x2 direction (Y-direction)
        self.ex2          = prop.LayerResolution * prop.n + 1
        
        verts1       = numpy.linspace(0, prop.dx1, (prop.ex1))
        verts2       = numpy.linspace(0, prop.dx2, (self.ex2))
        ## The geometry of the bed.
        self.domain, self.geom = mesh.rectilinear([verts1, verts2])
        ## A knotvector for the x2 direction
        self.knotm        = BuildKnotMult(prop)
        self.ns.basis     = self.domain.basis(prop.basistype, degree=prop.degree, knotmultiplicities=[None,self.knotm])
        
        #Define variables that need to be in the namespace
        self.ns.rho          = prop.rho                            # density
        self.ns.c            = prop.c                              # Specific heat capicity
        self.ns.k            = prop.k                              # Heat conductivity
        self.ns.L            = prop.L                              # Latent heat generation
        self.ns.EQinitialize = prop.EQinitialize
        self.ns.h            = prop.h                              # Convective Heat Loss
        self.ns.Ti           = prop.Ti
        self.ns.Toutside     = prop.Toutside	                  # Temperature at infinity
        self.ns.qadiabatic   = prop.qadiabatic
        self.ns.itime        = prop.itime
        
        self.ns.x = self.geom
        self.x0 , self.x1 = self.geom
        self.ns.T = 'basis_n ?lhs_n'
        self.ns.ph = 0.5 * (function.tanh(self.prop.S * 2 / (self.prop.Tl - self.prop.Ts) * ( self.ns.T - (self.prop.Ts + self.prop.Tl) / 2 )) + 1)
        
        ## Conductivity matrix (for entire bed at the start)
        self.conductivitySTART0 = self.domain.integrate(self.ns.eval_ij('basis_i basis_j EQinitialize'), geometry=self.ns.x, ischeme=self.prop.ischeme)
        ## Constrains for the bottom of the domain (dirichlet)
        self.consBot  = self.domain.boundary['bottom'].project(self.prop.Ti,onto=self.ns.basis, geometry=self.ns.x, ischeme=self.prop.ischeme)        
        
        ## initial condition/solution
        self.lhs         = numpy.zeros(len(self.ns.basis)) + self.prop.Ti
        ## Initial load condition (homogenious)
        self.loadEVAL1   = numpy.zeros(len(self.ns.basis))
    
        #check if adiabatic
        if prop.adiabatic:
            ## Function that is used for the boundary condtion of the walls (adiabatic or convective flow)
            self.func = 'basis_n qadiabatic'
        else:
            self.func = '-basis_i h (Toutside - T)'        
        
        log.user("Initialization of temperature driven solidification simulation")
        
    ## Updating the initial conditions for the new layer. @brief 
    # Every layer the evaluated an updated domain are updated, including the capacitance, conductivity, load, constrains, and solver matrices     
    # @param    i = The layer that is evaluated            
    def update_layer(self,i):
        ## Temperature for all timesteps (list of objects)
        self.allT =[]
        ## Phase for all timesteps (list of objects)
        self.allph =[]
        ## end phase for all timesteps (list of objects)
        self.allphtotal = []
        ## End phase for all timesteps (list of arrays)
        self.allphtotaleval = []        
        
        ## Evaluated domain (bottom to layer of interest)
        self.domainEVAL   = self.domain[:,                          : (i + 1) * self.prop.LayerResolution]
        ## Domain that get's updated (domain of a layer)        
        self.domainUPDATE = self.domain[:, i*self.prop.LayerResolution   : (i + 1) * self.prop.LayerResolution]
        
        ## Conductivity of the domainupdate
        self.conductivityUPDATE  = self.domainUPDATE.integrate(self.ns.eval_ij('basis_i,k (k basis_j,k)'), geometry=self.ns.x, ischeme=self.prop.ischeme)

        ## Capacitance of the domain update
        self.capacitanceUPDATE  = self.domainUPDATE.integrate(self.ns.eval_ij('basis_i basis_j rho c'), geometry=self.ns.x, ischeme=self.prop.ischeme)

        if i == 0:
            ## Capacitance of the entire bed
            self.capacitanceTOTAL    = self.capacitanceUPDATE
            ## Conductivity of the entire bed
            self.conductivityTOTAL   = self.conductivityUPDATE + self.conductivitySTART0
            ## Boundary that is constrained by a neumann boundary condition (includes bottom when no dirichlet is there)
            self.bound = 'left,right'
        else:
            self.capacitanceTOTAL    += self.capacitanceUPDATE
            self.conductivityTOTAL   += self.conductivityUPDATE
            self.bound = 'left,right'

        # Build load
        ## Load of the part of the domainupdate
        self.loadUPDATE = self.domainUPDATE.boundary['left,right'].integrate(self.ns.eval_n(self.func), geometry=self.ns.x, ischeme=self.prop.ischeme)
        ## Load of the evaluated domain        
        self.loadEVAL1 += self.loadUPDATE

        #create constrains
        ## Constrains of the part of the layers that are not yet active
        self.consEVAL1 = BuildConstrains(self.ns,self.prop,i)
        
        ## Combined constrains
        self.consEVAL = self.consEVAL1 | self.consBot
        #Building matrices for the solver
        
        ## Solver matrix for implicit backward euler, loadcorrection >>(C/Dt + K)<< T = F - >>C/Dt<< T
        self.B = (1 / self.prop.timestep) * self.capacitanceTOTAL
        ## Solver matrix for implicit backward euler, >>(C/Dt + K)<< T = F - C/Dt T
        self.A = self.B + self.conductivityTOTAL

    ## Updating the load every timestep (laser in our case) @brief 
    # Every timestep a new load needs to be calculated, becuase of the moving laser.
    # @param    i = The layer that is evaluated   
    # @param    itime = The timestep that is evaluated          
    def update_load(self,itime,i):
        
        self.ns.q = self.prop.power * function.exp( -((self.x0-self.prop.dx1/5 - self.prop.dx1/5*3/self.prop.maxiter *itime)**2)/(0.001**2) )
        ## Load induced by laser (heat flux)        
        self.loadEVAL2 = self.domainEVAL.boundary['top'].integrate(self.ns.eval_n('basis_n q'), geometry=self.ns.x, ischeme=self.prop.ischeme)
        ## Complete Loadvector (laser and boundary condtions)        
        self.loadTOTAL = self.loadEVAL1 + self.loadEVAL2
        
    def solve(self,itime):
        #Solve the equation
        self.lhs = self.A.solve((self.loadTOTAL) + self.B.matvec(self.lhs), constrain=self.consEVAL)
        self.ns.lhs = self.lhs
        self.ns.T = self.ns.basis.dot(self.ns.lhs)
        
        #Updating the solution
        self.ns.ph = 0.5 * (function.tanh(self.prop.S * 2 / (self.prop.Tl - self.prop.Ts) * ( self.ns.T - (self.prop.Ts + self.prop.Tl) / 2 )) + 1)

        if itime == 0 and i == 0:
            self.ns.phend = self.ns.ph
            if self.prop.outputfile == 'vtk': 
                self.phend = self.domain.elem_eval(self.ns.ph, ischeme='vtk', separate=True)
                self.ns.phendeval = self.ns.ph
        else: 
            self.ns.phend = function.max(self.ns.phend,self.ns.ph)
            if self.prop.outputfile == 'vtk':
                self.ns.phendeval = function.max(self.ns.phendeval,self.ns.ph)            #Solve the equation
        
    def toprint(self,itime):
        if itime % round(self.prop.maxiter/self.prop.Afig) == 0:
            self.allT.append([self.ns.T])
            self.ns.phtotal = self.ns.phend + self.ns.ph - 1
            self.ns.phend = self.domain.elem_eval(function.max(self.ns.phend, self.ns.ph), ischeme='vertex1', asfunction=True)
            self.allphtotal.append([self.ns.phtotal])
            if self.prop.outputfile == 'vtk':
                self.phend = numpy.maximum(self.phend,self.domain.elem_eval(self.ns.phendeval, ischeme='vtk', separate=True))                  
                self.allphtotaleval.append(self.phend)       
                self.ns.phendeval = self.ns.ph
    
    def printt(self,i):
        if self.prop.figures:
            #figures(ns,prop,domain,[allT], ['Temperature'],[[200,2100]],  prop.dpi, i)
            if self.prop.outputfile == 'vtk':
                fig = figures(self.ns,self.prop,self.domain,[self.allT,self.allphtotaleval], ['Temperature','Phase'],[[200,2100] ,[ -1,1]],  self.prop.dpi, i)
            else:
                fig = figures(self.ns,self.prop,self.domain,[self.allT,self.allphtotal], ['Temperature','Phase'],[[200,2100] ,[ -1,1]],  self.prop.dpi, i)
            fig.printfig(self.prop.outputfile)
### Thermomechanical process
class mech():
    
    def __init__(self,prop,solidifyT):
        self.prop = prop
        self.maxrefine = 3   #????
        self.pointstest = solidifyT.domain.elem_eval(solidifyT.ns.x , ischeme='bezier2', separate=True)
        log.user("Initialization of thermo-machanical simulation")
#    def update_domain(self,nsT):    
#        if numpy.max(nsT.T) > self.prop.Tl:
#            levelset =nsT.ph
#            domainnew = nsT.domain.trim(levelset-0.5, maxrefine = self.maxrefine)
#            info.error('The domain is updated')
#        
#        
#    def plot(domain,ns,levelset):
#        maxrefine = 3
#
#        pointstest,valtest = domain.elem_eval([ns.x,levelset-0.5] , ischeme='bezier2', separate=True)
##    with plot.PyPlot('test_if values are plottalbe', ndigits=1) as plt:
##        plt.mesh(pointstest,valtest)
##        plt.colorbar(orientation = 'horizontal')
#        
#        if numpy.max(valtest) > 0:
#            domainnew = domain.trim(levelset-0.5, maxrefine = maxrefine)    
#            points = domainnew.simplex.elem_eval(ns.x, ischeme='bezier2', separate=True)
#            with plot.PyPlot('test_domaintrim', ndigits=1) as plt:
#                plt.mesh(pointstest)
#                plt.mesh(points, edgecolors='r' ,dpi = 600, mergetol=1e-6)
#            with plot.PyPlot('Zoomed in', ndigits=1) as plt:
#                plt.mesh(points, edgecolors='r' ,dpi = 600, mergetol=1e-6)

  #      plt.segments( tpoints , color='g', lw=2 )


