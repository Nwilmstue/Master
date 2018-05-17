#! /usr/bin/env python3

## @package Main
# This is the main code, from where the simulation is conducted. 
from nutils import *
from IGA_AM_lib import *
from pre_post_lib import * 
import numpy, unittest, time, collections, itertools, functools, numbers, scipy, sys,os,datetime, shutil, matplotlib

def main():
    #initialization
    prop     = properties( printprop = True , extended = True , EXIT = False )
    prop.backup()
    solidifyT = solidification(prop)
    mechan = mech(prop,solidifyT)

    #layerloop
    for i in range(prop.n):
        log.user('Layer', i+1, 'out of', prop.n , 'layers ','with',prop.maxiter,' timesteps.')
        solidifyT.update_layer(i)

    #timesteploop
        for itime in range(0,prop.maxiter):
            #   solidification proces
            solidifyT.update_load(itime,i)
            solidifyT.solve(itime,i)
            solidifyT.toprint(itime)
            #   Mechanical process      
#            mechan.update_domain(soldifyT.ns)
            
        solidifyT.printt(i)

if __name__ == '__main__':
    with config(verbose = 3, nprocs = 8, richoutput = True):
        cli.run(main)