# -*- coding: utf-8 -*-
"""
@author: Doruk Alp, PhD
updated: July, Aug, Oct, Dec 2023
updated: May 2021
1st vers: Jan. 2021 

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
#
# STEP 1. Import needed libraries.
# 
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic 

from numba import njit, jit

# #@njit fails with icecream

import sys # use print for numba + sys.stdout.flush() to print in pll
# neither stdout works with numba :(

import md_glb # access & change var stored in var_glb

# DA: re-import numpy etc in each new module, or 
# DA: Bec. "from var_glb import *" is discouraged 
# DA: can only access to modules within var_glb by providing full address:
# b = 0
# var_glb.ic(b)
# c = var_glb.np.zeros(3)

# *****************************************************************************
# %% Custom functions, before main code:
# *****************************************************************************

@njit
def fn_a(ai, b, t):
    # instantaneous decline at a given t
    a = ai / (1.0 + ai*b*t)
    return a

@njit
def fn_D_to_a (D, b, dt):
    # convert D (eff. decline) to a (nom. decline)
    if b < 0.0:
        print('b < 0.0, b = ', b)
    elif b == 0.0:
    # eqn. for exp. decline:
        a = - (np.log(1.0-D))/dt
    if b > 0.0:
    # eqn. for hyp. decline
        a = 1.0/b/dt * ( (1.0 - D)**(-1.0 * b) - 1.0 )
        if a <= 0.0: print('a <= 0.0, a = ', a)
        
    return a

@njit
def fn_Arps(qi, ai, b, t):
    # ai is initial nominal decline per unit time of choice, 
    # is it REALLY instantaneous, then ??
    if b <= 0.0: # Exp. Decline
        q = qi * np.exp(-ai*t)
    if b > 0.0: # Hyperbolic
        q = qi * (1.0 + ai*b*t)**(-1.0/b)
        
    if q < 0.0:
        print("q < 0: ", q, qi, ai, b, t)
    # if q == -0.0:
    #     print("q == -0.0: ", q, qi, ai, b, t)
    #     q = 0.0 # correct -ve zero to +ve
        
    return q

@njit
def fn_sqdt(qi, ai, b, t):
    # integral of hyperbolic eqn:
    if b < 0.0:
        sqdt = -981.0    
    elif b == 0.0: # Exp. Decline
        sqdt = qi *( 1.0 - np.exp(-ai*t))/ai        
        # sqdt = -990.0      
    elif b == 1.0:
        q = fn_Arps(qi, ai, b, t)
        sqdt = qi/ai * np.log(qi/q)
        #sqdt = -982.0
    elif b > 0.0: # Hyperbolic 
        sqdt = qi/ai/(b - 1.0) * ( (1.0 + b*ai*t)**((b - 1.0)/b) - 1.0 )
        
    if sqdt < 0.0:
        print("sqdt < 0: ", sqdt, qi, ai, b, t)
    # if sqdt == -0.0:
    #     print("sqdt == -0.0: ", sqdt, qi, ai, b, t)  
    #     sqdt = 0.0 # correct -ve zero to +ve
        
    return sqdt

# *****************************************************************************
# %% Multi-segment eqns.
# *****************************************************************************

@njit
def fn_tExp(ai, b, at):
    # time of switch to exp. decline
    # check below eqn:
    if b <= 0.0:
        tExp = 0.0
    else:
        tExp = (1.0/at - 1.0/ai)/b
    
    # if np.isnan(tExp):
    #     ic(ai, b, at)
    
    return tExp

@njit
def fn_Arps_2seg(ai, b, t, ts, aexp, bexp=None, qi=None):
    # combine hyperbolic and exponential
    if bexp is None: bexp = 0.0 
    if qi is None: qi = 1.0        

    qN = fn_Arps(qi, ai, b, ts)
    
    if t <= ts:
        q = fn_Arps(qi, ai, b, t)
    if t > ts:        
        dt = t - ts
        q = fn_Arps(qN, aexp, bexp, dt)
    return q

@njit
def fn_Arps_2seg_multt(ai, b, arrt, ts, aexp, bexp=None, qi=None):
    # combine hyperbolic and exponential
    if bexp is None: bexp = 0.0 
    if qi is None: qi = 1.0          
     
    c = arrt.shape[0] # = len(arrt)
    q = np.zeros(c)

    qN = fn_Arps(qi, ai, b, ts) # big mistake to leave in the loop below 
    
    for i in range(0,c):
        t = arrt[i]
        
        if t <= ts:
            q[i] = fn_Arps(qi, ai, b, t)
        
        if t > ts:        
            dt = t - ts
            q[i] = fn_Arps(qN, aexp, bexp, dt)

        # txt = 'qi: ' + str(qi) + ', t: ' + str(t) + ', i: ' + str(i) + ', q: ' + str('%.6E' % q[i]) + '\n'
        
        # f = var_glb.outf
        # with open(var_glb.outf.name,"a") as f:
        #     f.write(txt)

    return q

@njit
def fn_sqdt_2seg(ai, b, t, ts, aexp, bexp=None, qi=None):
    # cum. Arps. in 2 parts
    # does not work on arrays due to if statements
    # either vectorize or replace if statements with what??
    if bexp is None: bexp = 0.0
    if qi is None: qi = 1.0
    
    qN = fn_Arps(qi, ai, b, ts)    
    sqdtN = fn_sqdt(qi, ai, b, ts)
        
    if t <= ts:
        sqdt = fn_sqdt(qi, ai, b, t)
    if t > ts:        
        dt = t - ts
        sqdt = sqdtN + fn_sqdt(qN, aexp, bexp, dt)

    # ic(ai, b, t, ts, aexp)
    # ic(sqdt)

    return sqdt

#
# ER fn.
#
@njit # jit gives error
def fn_ER(tmax, val, bexp=None, qi=None):
    # calc. ER for given values
    if bexp is None: bexp = 0.0
    if qi is None: qi = 1.0 

    Di, b, Dexp = val
    
    ai = fn_D_to_a(Di, b, 12.0)
    aexp = fn_D_to_a(Dexp, b, 12.0) # should use b not bexp here!!
        
    # find time of switch to exp. flow
    tExp = fn_tExp(ai, b, aexp)

    ER = fn_sqdt_2seg(ai, b, tmax, tExp, aexp)

    if ai <= 0.0: print(ai) # ic(ai)
    if ER <= 0.0: print(ER) # ic(ER)

    return ai, tExp, aexp, ER



# *****************************************************************************
# %% main fn.
# *****************************************************************************

def main():
    return

if __name__ == '__main__':
    main()