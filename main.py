#! /usr/bin/env python3

## @package Main
# This is the main code, from where the simulation is conducted. 
from nutils import *
from IGA_AM_lib import *
from pre_post_lib import * 
import numpy, unittest, time, collections, itertools, functools, numbers, scipy, sys,os,datetime, shutil, matplotlib


## Properties of the simulation
#

def main():
    # Create namespace and properties
    time1 = timeprint()
    backup()
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

        time1.timeprint('Constructed')
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
#                ns.phend = function.max(ns.phend,ns.ph)
                ns.phend = function.add(ns.phend, ns.ph)

            ns.phtotal = ns.phend + ns.ph - 1
            
            # Select pictures that needs to be printed
            if itime % 6 == 0:
                allT.append([ns.T])
                allphtotal.append([ns.ph])
            
        if i == prop.breakvalue:
            break
        time1.timeprint('Solved \t   ')
        if prop.figures:
            #figures(ns,prop,domain,[allT], ['Temperature'],[[200,2100]],  prop.dpi, i)
            figures(ns,prop,domain,[allT,allphtotal], ['Temperature','Phase'],[[200,2100] ,[ -1,1]],  prop.dpi, i)
            #figures(ns,prop,domain,[allphtotal], ['Phase'],[[ -1,1]],  prop.dpi, i)
        time1.timeprint('Printed  \t ')
    print( 'Total time is'  ,round(time.time() - time1.tottime),'seconds')

if __name__ == '__main__':
    with config(verbose = 3, nprocs = 8):
        cli.run(main) 
        print('finished')
