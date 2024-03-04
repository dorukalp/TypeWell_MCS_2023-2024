# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:40:32 2023
store global variables
@author: Doruk Alp, PhD
updated: Dec. 2023
1st vers.: Jul. 2023 

    call stand-alone MC run file several times
    to repeat the MC experiment
    plot to see if mean EUR is reaching a threshold
    
    assume q/qi, normalized rate, so no CDF for qi
    for now assume const. D_lim for switch to exp. eqn.
    generate CDF for Di, b
    remember to truncate Di to range (0-1), b to [0 - +inf]
    random sample Di, b, get CDF for EUR
    plot DC for p10-50-90 of EUR 
        -> need to store Di, b and EUR in table
           & fn to trace back Di, b from EUR value


"""
import numpy as np
from icecream import ic

#import inspect
from inspect import getsourcefile          # DA: to get file path
from os.path import abspath, dirname       # DA: to get file path
from os import getppid, getpid

import sys

from numba import njit, jit
from numba.types import FunctionType

# Numba needs to know all the dependencies of a func
# at compile time. 
# Must move import statements outside the function,
# OR use the numba.extending.overload decorator
#  to define a custom implementation for the imported module.

# import multiprocessing as mp # do not leave here, gives err.


# *****************************************************************************
# constants, custom precision params. & vars. 
# to be set & remain accessible to all fns.
# To store global vars. 
# *****************************************************************************

class clss_glb:
    def __init__(self, srcf=None):
        # import sys; print(sys.version)
        
        # self.icp = int(99) # num. of decimal digits stored, for Decimal pack.
                     # 42 seems to be the limit, default is 28
                     # 32-bit is ~7 digits
                     # 64-bit is ~14 digits
        
        self.neg_one = -1.0
        self.zero = 0.0
        self.one = 1.0
        self.two = 2.0
        self.five = 5.0
        self.ten = 10.0
        self.hdrd = 100.0
        self.year = 365.0
        
        self.pert = 1.0e-5
        
        self.pinf = np.inf # define +ve infinity
        # self.ninf = -np.inf
        self.frmt = '%.6E'         # std. num. formatting
        self.frmtF = '%.15E'         # std. num. formatting
        #self.icall = 0

        if srcf is None:
            f = list(sys._current_frames().values())[-1]
            # print(f)
            # print("test: ", f.f_back.f_globals['__name__'])
            srcf = f.f_back.f_globals['__file__']

        # switched to sys._current_frames, below not needed for now:
        # srcfile, srcdir, srcpath = fn_srcpath(srcf)
        outf, srcfile, srcdir, srcpath = fn_outf(srcf)

        self.srcpath = srcf     # srcf = srcpath
        self.srcfile = srcfile
        self.srcdir = srcdir
        
        self.outf = outf
        
        self.sec = []
        self.secN = []
        
        # self.pool = pool_handler()
# NotImplementedError: 
#    pool objects cannot be passed between processes or pickled


# @jit or njit fails
def fn_srcpath(srcpath=None):
# DA: path of this .py file during run time.
# DA: deploy results into the same path

    # did not work as expected:
    # dirpath = dirname(abspath(getsourcefile(lambda:0)))

    # switched to sys._current_frames, below not needed for now:
    # srcpath = abspath(srcf) #, where srcf = getsourcefile(lambda:0) in the calling fn.

    if srcpath is None:
    # switched to sys._current_frames, below not needed for now:
        srcpath = abspath(getsourcefile(lambda:0))

    inds = [i for i, c in enumerate(srcpath) if c == '\\']
    LB = max(inds)+1
    
    inds = [i for i, c in enumerate(srcpath) if c == '.']        
    UB = max(inds)
    
    srcfile = srcpath[LB:UB]
    srcdir = srcpath[:LB]

    return srcfile, srcdir, srcpath

def fn_outf(srcf=None):

    # switched to sys._current_frames, below not needed for now:
    srcfile, srcdir, srcpath = fn_srcpath(srcf)
    
    str_outf = srcdir + srcfile + '_out.csv'
    txt = 'Output from ' + srcpath + ':\n'
   
    fn_write(txt, str_outf)
    
    # using with supposedly ensures file handler closed upon exit
    with open(str_outf, 'w', encoding="utf-8") as f:
        f.write(txt)

    #fN = open(str_outf, 'w', encoding="utf-8")
    # f.write(txt)
            
    return str_outf, srcfile, srcdir, srcpath

def fn_write(txt, outf):

    # using with supposedly ensures file handler closed upon exit
    with open(outf, 'a', encoding="utf-8") as f:
        f.write(txt)
            
    return

def fn_info_process():
    # txt += 'mod. name: ' + __name__ + ', '
    txt = 'ppid: ' + str(getppid()) + ', pid: ' + str(getpid())
    return txt, getppid(), getpid()

    # txt = 'ppid: ' + str(os.getppid()) + ', pid: ' + str(os.getpid())
    # return txt, os.getppid(), os.getpid()


# *****************************************************************************
# %% main fn.
# *****************************************************************************
def main():
# since using this file as var storage module
# nothing to write in this section.
    print('This is: md_glb.py ran standalone')
    return

if __name__ == '__main__':
    # if this file is exec. stand. alone
    # following statements are exec.
#    glb = clss_glb()
    main()
    raise SystemExit # place under __main__ only, else:
                     # if this file is called as a module,
                     # and it is executed, the run quits