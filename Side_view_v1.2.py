#! /usr/bin/env python3

from nutils import *
import numpy, unittest, matplotlib, time, collections, itertools, functools, numbers, scipy, sys,os,datetime, shutil
from prettytable import PrettyTable
from beeprint import pp

## Build a knotmultiplicity vector for every layer C0 continuous
#   @ Parameter     L = The amount of elements a layer consists of (integer)
#   @ Parameter     n = The amount of Layers (integer)
#   @ Parameter     degree = The polynomial degree of the spline functions (integer)
#   @ return        knotm = A knotmultiplicity list (list)
def BuildKnotMult(prop):
    knotm = [1] * ((prop.LayerResolution*prop.n) + 1)
    for index in [0,-1]:
        knotm[index] = prop.degree + 1
    wbc = [prop.LayerResolution * (i + 1) for i in range(prop.n - 1)]
    for index in wbc:
        knotm[index] = prop.degree
    return knotm

## Build a constrain vector
#   @ Parameter     ns = The namespace (Namspace class)
#   @ Parameter     prop =  All the properties of the simulation(Object properties)
#   @ Parameter     i = The incrementel layer wherefor the vector is build (integer)
#   @ return        cons = A constrainvector (object NaNVec)
def BuildConstrains(ns,prop,i):
    lengthconsELEMENT = len(ns.basis)/(prop.ex1+prop.degree-1)
    Azero = (prop.LayerResolution + prop.degree - 1) * (prop.n-i-1)
    consELEMENT = util.NanVec(int(lengthconsELEMENT))
    if Azero != 0:
        consELEMENT[-Azero:] = prop.Ti
    cons = numpy.tile(consELEMENT,(prop.ex1+prop.degree-1))
    return cons
    
class timeprint():
    def __init__(self):
        self.time1 = time.time()
        self.tottime = time.time()
        
    def timeprint(self,statement):
        print(statement + '\t in' , round(time.time() - self.time1), 'second(s)'  )
        self.time1 = time.time()

## Test if the initial conditions and constrains are correct.
#   @ Parameter
def testLHScons(ns,LayerResolution,domain,consEVAL,lhs, ischeme,degree):
    consEVALtest = domain[:,(LayerResolution + 1  )  :].project(0, onto=ns.basis, geometry=ns.x, ischeme=ischeme)
    numpy.testing.assert_array_equal(consEVAL,consEVALtest)
    print('Unittest constrains passed')
    lhs0test = domain.integral('(T)^2' @ ns, geometry=ns.x, degree=degree*2)
    lhs0test = solver.optimize('lhs', lhs0test, droptol=1e-15)
    lhstest = lhs0test
    print('Unittest lhs0 passed')

## Plot figures in an efficient fashion
def figures(ns,prop,domain,variable,name,clim,dpi, ii):
    assert len(variable) == len(name), 'The length of the variables and name do not match'
    assert len(variable) == len(clim), 'The length of the variables and clim do not match'

    Dinput = [ns.x]
    for k in range(0,len(variable)):
        for i in range(0,len(variable[0])):
            Dinput.append(variable[k][i][0])
#    print('In element eval with', len(variable[0])*len(variable),' arrays of around a length of', len(ns.basis)*4)
   
    assert len(Dinput) <= 150, ' Trying to print to many timesteps in one go' 
    Total_eval = domain.elem_eval(Dinput, ischeme='vertex1', separate=False)
    with config(verbose = 1, nprocs = 8):
        for i in  parallel.pariter(range(0,len(variable[0])),8):
            for k in range(0,len(variable)):
                    with plot.PyPlot(name[k] + str(ii) + ' {}'.format(i)) as plt:
                        plt.title(name[k] + ' at t={:5.1f}'.format(i*prop.timestep))
                        plt.mesh(Total_eval[0], Total_eval[i+1+k*len(variable[0])])
                        plt.colorbar(orientation = 'horizontal')
                        plt.clim(clim[k][0],clim[k][1])
                        plt.ylabel('Height in [m]')
                        plt.xlabel('Width in [m]')
                        #plt.figure(figsize=(prop.ex1*10,(prop.LayerResolution * prop.n + 1 )*10))
                        plt.savefig(prop.dir + '2. Figures/' + name[k] + str(ii) + ' {}'.format(i),bbox_inches='tight', dpi = dpi)

class properties():

    def __init__(self,printprop = True,extended = True, EXIT = False):
        self.TOTAL = []
        #Choices
        self.TOTAL.append(['figures'                    ,True       ,'Create figures'                           , 'Simulation options'])
        self.TOTAL.append(['dpi'                        ,150        ,'Resolution of the figures'                , 'Simulation options'])
        self.TOTAL.append(['adiabatic'                  ,True       ,'Apply adiabatic boundary conditions'      , 'Simulation options'])
        self.TOTAL.append(['Utest'                      ,False      ,'Run Unittest for lhs and constrains'      , 'Simulation options'])
        self.TOTAL.append(['breakvalue'                 ,1000         ,'Until which layer is simulated'           , 'Simulation options'])
#        self.TOTAL.append(['animation'                  ,False      ,'Build GIF animation'                      , 'Simulation options'])

        #FEM options
        self.TOTAL.append(['ischeme'                    ,'gauss4'   ,'The scheme used for numerical integration'      , 'FEM options'])
        self.TOTAL.append(['basistype'                  ,'spline'   ,'The basisfunctions used over the domain. '      , 'FEM options'])
        self.TOTAL.append(['degree'                     ,3          ,'The polynomial degree of the basisfunctions'    , 'FEM options'])

        # Time solver options
        self.TOTAL.append(['timestep'                   ,1/500        ,'Timestel in [s]'    , 'Time options'])
        self.TOTAL.append(['maxiter'                    ,30           ,'amount of steps of the laser per layer '    , 'Time options'])


        # Material properties
        self.TOTAL.append(['c'                          ,520        ,'Specific heat capacity '                      , 'Material'])
        self.TOTAL.append(['rho'                        ,4510       ,'density '                                     , 'Material'])
        self.TOTAL.append(['k'                          ,16         ,'Heat conductivity '                           , 'Material'])
        self.TOTAL.append(['L'                          ,0          ,'Latent heat production '                      , 'Material'])
        self.TOTAL.append(['Tl'                         ,2000       ,'Temperature where mateirial is liquid '       , 'Material'])
        self.TOTAL.append(['Ts'                         ,1500       ,'Temperature where material is solid'          , 'Material'])
        self.TOTAL.append(['S'                          ,4          ,'The strength of TANH curve for T transistion' , 'Material'])
        self.TOTAL.append(['h'                          ,200        ,'Convective heat loss'                         , 'Material'])
        
        
        #Heat source (LASER) properties
        self.TOTAL.append(['power'                      ,1.3e8        ,'Laser power'                         , 'Laser properties'])
        
        #Boundary conditions
        self.TOTAL.append(['Toutside'                   ,0          ,'Temperature at infinity'                      , 'Boundary conditions'])
        self.TOTAL.append(['qadiabatic'                 ,0          ,'heat flux when adiabatic'                     , 'Boundary conditions'])

        #initial conditions
        self.TOTAL.append(['EQinitialize'               ,0          ,'Added heat from the start'                    , 'Initial conditions'])
        self.TOTAL.append(['Ti'                         ,200        ,'Intitial Temperature '                        , 'Initial conditions'])
        self.TOTAL.append(['itime'                      ,0          ,'Intitial timeset     '                        , 'Initial conditions'])

        #Geometry and topology variables
        self.TOTAL.append(['LayerResolution'            ,2           ,'Amount of elements per layer'         , 'Geometry and topology'])
        self.TOTAL.append(['n'                          ,20          ,'Amount of layers'                     , 'Geometry and topology'])
        self.TOTAL.append(['ex1'                        ,30         ,'Elements in x1 direction'             , 'Geometry and topology'])
        self.TOTAL.append(['dx1'                        ,0.015         ,'Distance in x1 direction in [m]'      , 'Geometry and topology'])
        self.TOTAL.append(['dx2'                        ,0.01         ,'Distance in x2 direction in [m]'      , 'Geometry and topology'])

        # Create the variables
        for item in range(0,len(self.TOTAL)):
            sol = self.TOTAL[item]
            setattr(self, sol[0], sol[1])

        if printprop:
            self.printprop(extended = True, EXIT = EXIT)
        
        # Create logfiles and backups
        today = datetime.date.today()
        self.dir = '/home/niki/Documents/1_Master/0_Master_thesis/3_code/Auto/'+str(today) + '/'
        self.srcfile = '/home/niki/Documents/1_Master/0_Master_thesis/3_code/'+str(os.path.basename(__file__))               
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            os.makedirs(self.dir + '2. Figures/')
        shutil.copy(self.srcfile, self.dir)
            


    def printprop(self,extended = True, EXIT = False):
        t = PrettyTable(['Variable','Value','Explanation', 'Variable type'])
        sol = self.TOTAL[0]
        typecheck = sol[3]
        for item in range(0,len(self.TOTAL)):
            sol = self.TOTAL[item]
            if not typecheck == sol[3]:
                t.add_row(['      ','      ','      ','      '])
            t.add_row([sol[0], sol[1] ,sol[2], sol[3]])
            typecheck = sol[3]
        print(t)

        if extended:
            t = PrettyTable(['What','Value','Explanation'])
            t.add_row(['Layerthickness', self.dx2/self.n ,'The thickness of the layer in [m]'])
            print(t)

        if EXIT:
            print('Exitted the script in printproperties' )
            sys.exit()



def main():
    # Create namespace and properties
    time = timeprint()
    ns = function.Namespace()
    prop = properties(printprop = True , extended = True , EXIT = False)
    #config(verbose=2, nprocs=8)

    # Construct topology, geometry and basis
    ex2          = prop.LayerResolution * prop.n + 1   #Elements in x2 direction
    verts1       = numpy.linspace(0, prop.dx1, (prop.ex1))
    verts2       = numpy.linspace(0, prop.dx2, (ex2))
    domain, geom = mesh.rectilinear([verts1, verts2])
    knotm        = BuildKnotMult(prop)
    ns.basis     = domain.basis(prop.basistype, degree=prop.degree, knotmultiplicities=[None,knotm])

    #Define variables that need to be in the namespace
    ns.rho          = prop.rho                            # density
    ns.c            = prop.c                              # Specific heat capicity
    ns.k            = prop.k                              # Heat conductivity
    ns.L            = prop.L                              # Latent heat generation
    ns.EQinitialize = prop.EQinitialize
    ns.h            = prop.h                              # Convective Heat Loss
    ns.Ti           = prop.Ti
    ns.Toutside     = prop.Toutside	                  # Temperature at infinity
    ns.qadiabatic   = prop.qadiabatic
    ns.itime        = prop.itime

    x0,x1 =geom
    ns.x = geom
    ns.T = 'basis_n ?lhs_n'
    ns.ph = 0.5 * (function.tanh(prop.S * 2 / (prop.Tl - prop.Ts) * ( ns.T - (prop.Ts + prop.Tl) / 2 )) + 1)
    
    #Initialization
    conductivitySTART0 = domain.integrate(ns.eval_ij('basis_i basis_j EQinitialize'), geometry=ns.x, ischeme=prop.ischeme)
    consBot  = domain.boundary['bottom']. project(prop.Ti,onto=ns.basis, geometry=ns.x, ischeme=prop.ischeme)        
    
    #initial condition
    lhs         = numpy.zeros(len(ns.basis)) + prop.Ti
    loadEVAL1   = numpy.zeros(len(ns.basis))

    #check if adiabatic
    if prop.adiabatic:
        func = 'basis_n qadiabatic'
    else:
        func = '-basis_i h (Toutside - T)'

    for i in range(prop.n):
        print('Layer', i+1, 'out of', prop.n , 'layers ','with',prop.maxiter,' timesteps.')
        allT =[]
        allph =[]
        allphtotal = []        
        
        domainEVAL   = domain[:,                          : (i + 1) * prop.LayerResolution]
        domainUPDATE = domain[:, i*prop.LayerResolution   : (i + 1) * prop.LayerResolution]

        conductivityUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i,k (k basis_j,k)'), geometry=ns.x, ischeme=prop.ischeme)

        # Capacitance
        capacitanceUPDATE  = domainUPDATE.integrate(ns.eval_ij('basis_i basis_j rho c'), geometry=ns.x, ischeme=prop.ischeme)

        if i == 0:
            capacitanceTOTAL    = capacitanceUPDATE
            conductivityTOTAL   = conductivityUPDATE + conductivitySTART0
            bound = 'left,right'
        else:
            capacitanceTOTAL    += capacitanceUPDATE
            conductivityTOTAL   += conductivityUPDATE
            bound = 'left,right'

        # Build load
        loadUPDATE = domainUPDATE.boundary['left,right'].integrate(ns.eval_n(func), geometry=ns.x, ischeme=prop.ischeme)
        loadEVAL1 += loadUPDATE

        #create constrains
        consEVAL1 = BuildConstrains(ns,prop,i)
        consEVAL = consEVAL1 | consBot
        #Building matrices for the solver

        B = (1 / prop.timestep) * capacitanceTOTAL
        A = B + conductivityTOTAL

        time.timeprint('Constructed')
        for itime in range(0,prop.maxiter):

            # Create time dependant boundary condition
            ns.q = prop.power * function.exp( -((x0-prop.dx1/5 - prop.dx1/5*3/prop.maxiter *itime)**2)/(0.001**2) )
            loadEVAL2 = domainEVAL.boundary['top'].integrate(ns.eval_n('basis_n q'), geometry=ns.x, ischeme=prop.ischeme)
            loadTOTAL = loadEVAL1 + loadEVAL2

            #Solve the equation
            lhs = A.solve((loadTOTAL) + B.matvec(lhs), constrain=consEVAL)
            ns.lhs = lhs
            ns.T = ns.basis.dot(ns.lhs)

            #Updating the solution
            ns.ph = 0.5 * (function.tanh(prop.S * 2 / (prop.Tl - prop.Ts) * ( ns.T - (prop.Ts + prop.Tl) / 2 )) + 1)

            if itime == 0 and i == 0:
                ns.phend = ns.ph
            else:
                ns.phend = function.max(ns.phend,ns.ph)

            ns.phtotal = ns.phend + ns.ph - 1
            if itime % 6 == 0:
                allT.append([ns.T])
                allphtotal.append([ns.phtotal])
            
        if i == prop.breakvalue:
            break
        time.timeprint('Solved \t   ')
        if prop.figures:
            #figures(ns,prop,domain,[allT], ['Temperature'],[[200,2100]],  prop.dpi, i)
            #figures(ns,prop,domain,[allT,allphtotal], ['Temperature','Phase'],[[200,2100] ,[ -1,1]],  prop.dpi, i)
            figures(ns,prop,domain,[allphtotal], ['Phase'],[[ -1,1]],  prop.dpi, i)
        time.timeprint('Printed  \t ')
    print( 'Total time is'  ,round(time.time() - time.tottime()))

if __name__ == '__main__':
    with config(verbose = 3, nprocs = 8):
        cli.run(main) 
        print('finished')
