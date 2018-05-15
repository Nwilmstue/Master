#! /usr/bin/env python3

## @package IGA_AM_lib
# This is the library for the IGA and AM combined functions. 

from nutils import util, plot,function, mesh #, config, cli, function, mesh, parallel,plot
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
        
#class solidi():
#    def __init__():
#
### mechanical solver
#class mech():
#    
#    def __init__():
#    
#
#    def update_domain(self):    
#        
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


