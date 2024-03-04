# -*- coding: utf-8 -*-
"""
@author: Doruk Alp, PhD
updated: Dec 2023, Jan. 2024
updated: Jul 2023:

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

updated: July, Aug, Oct, Dec 2023
updated: May 2021
1st vers: Jan. 2021

    Copyright (C) 2024 Doruk Alp, PhD
    dorukalp.edu at gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the MIT License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    MIT License for more details.

    You should have received a copy of the MIT License
    along with this program.  If not, see <https://mit-license.org/>.


"""
# *****************************************************************************
# %% Std. libraries: 
# STEP 1. Import needed libraries.
# *****************************************************************************

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

#import scipy.stats as st 
# vs from scipy import stats as st
from scipy.stats import qmc, norm, truncnorm, mode, gaussian_kde
# vs from scipy.stats import qmc as st_qmc

# from statsmodels.distributions.empirical_distribution import ECDF

# import seaborn as sns
# sns.set_theme(style="ticks")

# import pandas as pd
    
from icecream import ic
import time
import pickle

from numba import njit, jit
# from numba.types import FunctionType

import multiprocessing as mp

import sys

# Own libraries::
import md_glb # access & change var stored in var_glb
import md_Arps
import md_plt


# *****************************************************************************
# %% Funcs. - Stand Alone 
# *****************************************************************************

@njit
def eqn_pdf_distNorm(mu, sig, b):
# probability density function
# not utilized, only for checking lib. results.
# gives normal distribution based on eqn. 
# mu = mean, sig = stdev, b = probability
    calc = 1.0/(sig * np.sqrt(2.0 * np.pi)) * np.exp( - (b - mu)**2.0 / (2.0 * sig**2.0) )
    return calc


def fn_sample_distNorm_numpy(dist, ndat, cfg, sseed=None):
    # self.arr[:,i] = fn_distNorm_numpy(i, self.dist, self.ndat, \
        # self.mult, self.sseed)
        
    # gen. norm. dist. for params., trunc. if mult > 1.0
    # store randomly sampled values from ditributions of n param.
    # sample 10% more than asked, remove -ves, then return required range

    # ndat: num. of samples to pick,
    # mult: multiply ndat to pick more than required, 
    # bec. will remove -ves  

    mu, sig, LB, UB = dist.mu, dist.sig, dist.LB, dist.UB
    
    # must reset seed value each time, to obtain same distribution
    # Fix random state for reproducibility.
    # reasonable to recall RNG here each time, so that can provide 
    # the same sequence of seeds, for testing purposes
    # Description on this page no longer seems to work: 
    # https://www.sharpsightlabs.com/blog/numpy-random-seed/

    # move RNG outside, pass as arg.
    # if seed is given, if block below will utilize, else will run fast.
    # else, no need to repeat
    # RNG = np.random.default_rng()

    RNG = cfg.RNG
    if sseed is not None: 
        RNG = np.random.default_rng(sseed)
    
    mult = 1.0
    if dist.boo_trunc: mult = cfg.mult
                            
    r = round(ndat*mult)
    a = RNG.normal(mu, sig, r)
    
    if mult > 1.0:
        # truncate normal dist. via rejecting -ves.
        # truncate based on Di = (0-1], based on b = (1 to +inf)
        # remove -ves, only keep +ves
        
        a = a[a[:]>LB]       
        a = a[a[:]<=UB]    

    a = a[:ndat]

    return a


def fn_sample_distNorm_LHS(dist, ndat, cfg, sseed=None):
    # Latin Hypercube Sampling
    # structure is different from regular implementation above
    # just to show it could be done differently.
           
    # Number of samples, find prime num. closest to ndat
    # this is already done in main seciton. So, no need anymore:
# added below line to reject -ves. Yet, perhps did not work.
# thus, using truncnorm afterwards
#    r = fn_find_prime_sq(ndat*mult) # after eliminating 
#    n = fn_find_prime_sq(ndat)

    n = ndat
    tbl_samples = np.zeros((n,2))

    # consider sseed+1 to diff. b sampling base
    
    # Generate a Latin hypercube sample 
    # strength=2 for orthogonal sampling
    # then number of samples must be the square of a prime number.
    # Considered moving this to out of fn, to the main section
    # yet, then we would not be able to give a fixed sequence of seeds
    
    sampler = cfg.LH_samplr
    if sseed is not None:
        sampler = qmc.LatinHypercube(d=2, strength=2, optimization="random-cd", seed = sseed)
    
    tbl = sampler.random(n)
    r, c = tbl.shape
       
    # Dexp = 0.05
    # LBs = [Dexp, 2.0]
    # UBs = [1.0, np.inf]
    # sample_scaled = qmc.scale(sample, LBs, UBs)

    for i in range(0,2):
        
        # 1st loop for Di, 2nd for b. 3 to 4 are non-truncated dists. not utilized here
        xmu = dist[i].mu
        xsig = dist[i].sig
        xUB = dist[i].UB
        xLB = dist[i].LB
 
        if dist[i].boo_trunc:
            a, b = (xLB - xmu) / xsig, (xUB - xmu) / xsig
            tbl_samples[:,i] = truncnorm(a, b, loc=xmu, scale=xsig).ppf(tbl[:,i])
        else:
            tbl_samples[:,i] = norm(loc=xmu, scale=xsig).ppf(tbl[:,i])
            
    return tbl_samples

@njit
def fn_sample_distNorm_stats(dist, ndat, mult, sseed=None):
    mu, sig, LB, UB = dist.mu, dist.sig, dist.LB, dist.UB
    r = round(ndat*mult)
    
    c = len(mu)
    a = np.zeros((r, c)) 
    
    for i in range(0, c):
        # instantiate an object w/ 4 param.: lower, upper, mu, and sigma
        # scipy.stats just uses numpy.random to generate its random numbers (2015)
        # Fix random state for reproducibility
        if sseed is not None: np.random.seed(sseed) # this would not work, see above.  

        obj = truncnorm((LB[i] - mu[i])/sig[i], (UB[i] - mu[i])/sig[i], loc=mu[i], scale=sig[i])
        
        # sample n times
        samples = obj.rvs(r)    
        a[:,i] = samples #[:,0]
    return a

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def fn_find_prime_sq(psq):
    
    check = False
    while not check:
        p = int(np.sqrt(psq))
        check = is_prime(p)
        psq -= 1        

    return p**2

@njit
def fn_idx_pcent_by_rank(tbl):
    # find row index of pcentiles for specific ER col.
    # uses previously stored rank values
    pcent = [10, 50, 90]
    ip = [0, 0, 0, 0] # stores row numbers

    r, c = tbl.shape
    
    j = 7 # index of rank column, ER ranks were stored earlier

    for i in range(0, 3):
        rank = round(r * pcent[i]/100.0)        
        ip[i] = np.where(tbl[:, j] == rank)[0][0]
        # rowi = tbl[tbl[:, j] == rank] # requires storing row nums in a column.
        # ip[i] = int(rowi[0,0])

    j = 6 # EUR column        
    mean = np.mean(tbl[:,j])
    ip[3] = np.abs(tbl[:,j]-mean).argmin()
    i = ip[3]
    err = (mean - tbl[i,j])/tbl[i,j]
    
    return ip, mean, err

#@njit
def fn_idx_stats(arr):
    # find row index of approx. pcentiles
    # for specified col.
    # or pass 1D array only
    # uses package functions
    
    # stores row numbers for 3 pcentiles, mean and mode
    ip = [0, 0, 0, 0, 0]
    ipN = [0, 0, 0, 0, 0]    
    ipK = [0, 0, 0, 0, 0] 
    
    ranks = arr.argsort().argsort() + 1
    rank_max = np.max(ranks) 
    pcents = ranks / rank_max # this already yields real (floating) value
    r = len(arr)

    pcent = [10, 50, 90]
    for i in range(0, 3):
        # I found this rank opt. to be more reliable 
        # than the other 2 opt. afterwards
        rank = round(r * pcent[i]/100.0)        
        ip_check = np.where(ranks == rank)[0][0]
        ip[i] = ip_check
        
        pcentval = np.percentile(arr, pcent[i])
        ipK[i] = np.abs(arr - pcentval).argmin()
                
        # # check for interpolate arg., got strange results:
        # ip_check = np.abs(arr - np.percentile(arr, pcent[i], interpolation='nearest')).argmin()
        # ipN[i] = ip_check
        #ic(i, pcentval, ip[i], ip_check[i]) #, ip_checkN[i])

    meanval = np.mean(arr)
    
    kde = gaussian_kde(arr)
    modeval = arr[np.argmax(kde(arr))]
    
    # modeval, intval = mode(arr) # did not seem to work

    ip[3] = np.abs(arr-meanval).argmin()
    ip[4] = np.abs(arr-modeval).argmin()
    
    return ip

# *****************************************************************************
# %% Classes
# vars. to be set & remain accessible to all fns. 
# *****************************************************************************

class clss_dist_Norm:
    def __init__(self, mu, sig, boo_trunc=None, LB=None, UB=None, name=None):
        # Store param. for norm. dist.: "Di, [-]", "b, [-]"
        # both tmax and aexp can be defined as distributions!!
        # these define "tuples", not lists or arrays
        self.name = "" if name is None else name
        self.mu   = mu
        self.sig  = sig
        
        self.boo_trunc = False if boo_trunc is None else boo_trunc
        
        # for truncating norm. dists.
        self.LB = LB 
        self.UB = UB 

class clss_MCTW_cfg:
    # Monte Carlo based type-well settings
    def __init__(self, tmax=None, tstp=None, Dexp=None):

        # time interval, num. of months
        self.tmax = (30.0 * 12.0 + 1) if tmax is None else tmax
        self.tstp = 1.0 if tstp is None else tstp

        # array of time values, for calc & plot:
        self.arrt = np.arange(0.0, self.tmax, self.tstp)
        self.nstp = len(self.arrt)

        # D to switch to exp. decline
        self.Dexp = 0.05  if Dexp is None else Dexp    
        
        self.ncol = 9   # num. of columns of resuls from single MC exp  
        
        # test switch: if provided, start random num. gen. with seed = 1
        # then increment +1 for each exp. This way can conduct a controlled exp.
        # compare serial & pll runs
        self.boo_seed = False 
        self.boo_seed = True

        self.pll = False
        self.pll = True
        
        self.mult = 1.5
        # Random Num. gen.
        self.RNG = np.random.default_rng()
        self.LH_samplr = qmc.LatinHypercube(d=2, strength=2, optimization="random-cd", seed = None)
        
class clss_MCTW:
    # Monte Carlo based type-well class
    # def __init__(self, cfg, dist, ndat, nexp, LHS):
    def __init__(self, cfg, ndat, nexp, boo_LHS):
                
        # ndat is the # of samples, or wells, to use in calc. & ranking
        #   ndat * mult > ndat, to compensate for -ve samples
        #   remove -ves, only keep ndat out of (ndat * mult) to cont. process
        # should have same # of samples for each parameter??
        
        # self.ndat = 999 if ndat is None else ndat
        # self.mult = 1.5 if mult is None else mult       

        # # numpy random seed ctrl, usage: np.random.seed(19680801)
        # self.sseed = sseed
        
        # store parameters for each fake well after each MC exp.:
        # col 0, 1: sampled Di, b, 
        # col 2 - 6: Dexp*, calc. ai, tExp, aexp, ER
        # * is const.
        # col 7: rank
        # col 8: actual calc. pcentile, approximating P10-50-90
        # add. cols. for multiple MC exps.
        # col 9: orig. row number of current row in the temp table after indv. MC exp.
        # col 10: pcentile index: P10-50-90-mean = 0-1-2-3
        # col 11: exp. count
        
        # results to be stored in 2 tables:
        # tbl: stores cfg & EUR data from MC exp., overwritten by each MC exp.
        # this should be reset for each MC exp, else no point for repeating MC

        # tblp: stores pcentile data for last exp, append to results table for each MC exp.     
        # store EUR p10-50-90-mean so we can reconstruct for plotting.

        # tblq: stores type-well (P10-50-90-mean) rates, for the last MC. exp.
        # tble: stores pcentile data for all exp, & mean EURs w/r/t Nexp. 
        # tblf: stores mode or mean of pcentiles from all MC exp, for type-well gen.
        # tblfq: stores type-well (P10-50-90-mean) rates for mean/mode of all MC exps.

        # also plot pdfs of ERs from a single MC, and pdf of pcentile ERs from all exp.
        # then decide if mean or mode is more representative.
        
        # results array dimensions, c=column
        
        r = ndat
        c = cfg.ncol # 9 # self.tbl_ncol        
        self.tbl = np.zeros((r,c))
        # self.tbl[:,0] = range(r) # not needed, kept for convenience
        
        r = 4        
        self.tblp = np.zeros((r,c+2)) # temp. store P10-50-90-mean values
                                      # +1) 0 to 3 idx for P10-50-90-mean, to filter main table 
                                      # +2) orig. idx from tbl                                  
        r = cfg.nstp
        self.tblq = np.zeros((r,4)) # temp. store rates for pcentile wells, why??


        # section below is perm. storage, separate from above.

        r = nexp*4     
        self.tble = np.zeros((r,c+3)) # +3) store exp count


        self.tblf = np.zeros((4,c+3)) # +3) store mean or mode of pcentiles of all exp.

        r = cfg.nstp
        self.tblfq = np.zeros((r,4)) # store rates for pcentile wells, final, based on tblf

        r = nexp     
        self.tblm = np.zeros((r,2)) # +1) cum. avg. of meanER from nexp
                                    # +2) err_meanER for each exp.

        self.boo_LHS = boo_LHS              # if TRUE, then do Latin hypcube sampling
        # 1) exp num., no need to store.
    
        # self.err_MeanER = 0.0 # %err in mean of ERs and ER of well closest to mean
        #                       # No new fake well for mean ER, for now.
        #                       # Just pick among current fakes with mean ER closest


# *****************************************************************************
# %% Funcs. - using classes & modules
# *****************************************************************************

#@njit
def fn_set_MCTW_objs(nexp, ndat, xL_dist, xcfg):
    # set fixed parameters block used to be here
    # ready data sets for MC exps:
    # r, c = ndat.shape #.shape[0] neither works if 1D array
    # r = ndat.shape    # this returns a tuple, have to revert to len() for 1D array
    r = len(ndat)
   
    xL_MCTW = []
    
    for i in range(r):
        if i < 2:  # 1st set is for Latin Hypercube
            # xL_MCTW.append(clss_MCTW(xcfg, xL_dist, ndat[k], nexp, LHS=True))            
            xL_MCTW.append(clss_MCTW(xcfg, ndat[i], nexp, boo_LHS=True)) 
        else:       # others naive/simple MC
            # xL_MCTW.append(clss_MCTW(xcfg, xL_dist, ndat[k], nexp, LHS=False))
            xL_MCTW.append(clss_MCTW(xcfg, ndat[i], nexp, boo_LHS=False))
        
        # below no longer applies as tbl is gen. locally within fn_MCTW
        # and re-assigned to xL_MCTW obj at the end of run.
        # xL_MCTW[i].tbl[:,2] = xcfg.Dexp

    return xL_MCTW


def fn_loop_MCTW_exp(nexp, ndat, xL_dist, xcfg, xL_MCTW):

    # r, c = ndat.shape[0] # does not work for 1D arrays

    r = len(ndat) # == nset = len(xL_MCTW)
    c = xcfg.ncol

    # set args, needed for mp pll runs.
    # may consume memory unneccessarily, consider different options
    # xL_MCTW[i] was intended as a shared mem. access, 
    # thus, does not work well with pll.
    # separate temp. tables/arrays from xL_MCTW[i] as new clss
    # find a way to construct nargs with this new class only
    # then simply copy results from pll run to proper row section
   
    
    nargs = []
    for i in range(r):          # loop for data set    
        for j in range(nexp):   # loop for exp.
            # add data set index to arg. to trace & id pll runs easily
            # And, to set predetermined seed values for rand. num. gen., to test
            # for instance, same seed across 4 data sets, updated every exp.
            # this allows for perfect comparison across sample sizes
            # as only diff. between sets will be sampling tech. & sample count
            nargs.append([i, r, xL_dist, xL_MCTW[i].boo_LHS, \
                          ndat[i], j, nexp, xcfg])
 
            # iset, nset, dist, boo_LHS, ndat, iexp, nexp, cfg    

    frmt = '%.4E'
    
    # only pass iexp & iset and base cfg to pll, rest of the table should be
    # completed in each pll, then pll_output should be stored in the main MCTW clss 
    
    if xcfg.pll:
        k = 0   # arg. num.
        with mp.Pool(processes = mp.cpu_count() - 2) as pool:
            for result in pool.map(fn_MCTW, nargs):
                # inside this loop is exec.
                # after each time 1 of pll runs completed by the fn_xxx !??
                # NO!, results from pll runs are stored in mem.
                # and then below section is repeated over each result
                # this consumes significant amount of memory, 
                # ensuing in mem. error for  large num. of exp. & ndat
                
                ip, meanER, err_meanER, tblp = result[0:4]
                
                i = nargs[k][0]    # this arg is the data set number
                iexp = nargs[k][5] # this arg is the exp number
                
                # add to cum. exp. table for permanent storage:
                # for each exp. we make 4 entries: P10-50-90 & mean
                j = iexp * 4    # row number of tble (exp. res. table)
                xL_MCTW[i].tble[j:j+4,:c+2] = tblp
                xL_MCTW[i].tble[j:j+4,c+2] = iexp

                # find mean of ER & store permanenetly, for nexp vs mean ER plot     
                # filter current table of permanent results
                tbl = xL_MCTW[i].tble[xL_MCTW[i].tble[:,c+1]==3]
                     
                xL_MCTW[i].tblm[iexp,0] = np.mean(tbl[:,6])
                xL_MCTW[i].tblm[iexp,1] = err_meanER 

                iset = nargs[k][0]                
                txt = 'set: ' + str(iset) \
                        + ', exp: ' + str(iexp) + ', ndat: ' +  str(ndat[i])
                sys.stdout.flush()   
                print("Completed: ", txt, '>> fn_MCTW_pll', flush=True)           
                sys.stdout.flush()
                
                if iexp == nexp-1:
                    # store data of last exp:
                    tbl_temp = result[4]
                    xL_MCTW[i].tbl = tbl_temp
                
                k += 1
                                
    else:
        for i in range(r):
            # ii = 0
            for j in range(nexp):
                k = (i * nexp) + j
                
                result = fn_MCTW(nargs[k])
                
                ip, meanER, err_meanER, tblp = result[0:4]

                
                i = nargs[k][0]    # this arg is the data set number
                iexp = nargs[k][5] # this arg is the exp number
                
                # add to cum. exp. table for permanent storage:
                # for each exp. we make 4 entries: P10-50-90 & mean
                j = iexp * 4    # row number of tble (exp. res. table)
                xL_MCTW[i].tble[j:j+4,:c+2] = tblp
                xL_MCTW[i].tble[j:j+4,c+2] = iexp

                # find mean of ER & store permanenetly, for nexp vs mean ER plot     
                # filter current table of permanent results
                tbl = xL_MCTW[i].tble[xL_MCTW[i].tble[:,c+1]==3]
                     
                xL_MCTW[i].tblm[iexp,0] = np.mean(tbl[:,6])
                xL_MCTW[i].tblm[iexp,1] = err_meanER 
                
                if iexp == nexp-1:
                    # store data of last exp, for plotting purposes:
                    tbl_temp = result[4] 
                    xL_MCTW[i].tbl = tbl_temp
                
                
                # iset, nset, dist, boo_LHS, ndat, iexp, nexp, cfg
                txt = 'set: ' + str(nargs[k][0]) \
                        + ', exp: ' + str(nargs[k][5]) + ', ndat: ' +  str(nargs[k][4]) 
                print("Completed: ", txt, '>> fn_loop_MCTW_exp', flush=True)  

                # res.append(fn_MCTW(nargs[i]))
            
        txt = 'complete arg: ' + str(i)
        ic(txt) 
       
    return xL_MCTW


def fn_MCTW(nargs):
    # single MC exp.
    # use iset to assign a unique seed for random num. gen, for testing purposes 
    iset, nset, dist, boo_LHS, ndat, iexp, nexp, cfg = nargs
    
    r = ndat
    c = cfg.ncol # 9 # self.tbl_ncol 
    tbl = np.zeros((r,c))
    tblp = np.zeros((4,c+2))

    # this block is no longer needed as we moved RNG and LH_samplr under cfg
    # Now, a fixed seed can be entered at the cfg definition
    # NO!! we need it here so that seed increments with iexp

    nseed = None    
    if cfg.boo_seed:
        # use same seed across data set for each exp.
        # update seed for each new exp.
        nseed = iexp # if = 1, all MC exp. will return same values

    # sample distributions, every time.
    tbl[:,0:2] = fn_sample(dist, ndat, boo_LHS, cfg, nseed)
    tbl[:,2] = cfg.Dexp
    
    for i in range(0, r):
        # loop over sample sets (each fake well), calc & store EUR
        # fn_ER returns .tbl[:,3,7] = ai, tExp, aexp, ER
        tbl[i,3:7] = md_Arps.fn_ER(cfg.tmax, tbl[i,0:3])

    # sort & calc. rank of ER30 values for quick retrieval: 
    # duplicated in fn_idx_stats, remove one 
    ranks = tbl[:,6].argsort().argsort() + 1
    rank_max = np.max(ranks) 
    pcent = ranks / rank_max # this already yields real (floating) value

    tbl[:,7] = ranks
    tbl[:,8] = pcent

    # find indices of percentiles & mean ER for this MC exp.
    # then re-construct corresponding type-well 

    ip, meanER, err_meanER = fn_idx_pcent_by_rank(tbl)

    # indices of pcentiles from above fn. should be matching fn. below!?
    # both methods yield close results, diff. due to rounding and abs. value.
    # will rely on above method for now, in 1 instance ip_check gave wrong result    
    #ip_check = fn_idx_stats(xMCTW.tbl[:,6:8])

    
    # to be flattened and accelerated:        
    for i in range(0, 4):
        k = ip[i]
        # collect pcentile wells in the temp. table
        tblp[i,:c] = tbl[k,:]
        tblp[i,c] = k
        tblp[i,c+1] = i
        
    # calc. flow rates for type-wells from single MC: - why do we need this ???
    # Previously, the block below was needed at this level, for single MC exp.
    # for multiple MC exp. case, not needed here.
    # we shall calc. rates corresp. to Di & b for the mean of pcentiles from all exp.
    # once all exp. are completed, outside this fn.
    # At this level, it only increases comp. time., thus c/o for now.
    #     # get rates for pcentile wells:
    #     # fn_ER returns .tbl[:,3,7] = ai, tExp, aexp, ER
    #     # def fn_Arps_2seg_multt(ai, b, arrt, ts, aexp, bexp=None, qi=None)
    #     xMCTW.tblq[:,i] = md_Arps.fn_Arps_2seg_multt(xMCTW.tbl[k,3], xMCTW.tbl[k,1], 
    #                               cfg.arrt, xMCTW.tbl[k,4], xMCTW.tbl[k,5])
    
    txt = 'set: ' + str(iset) \
            + ', exp: ' + str(iexp) + ', ndat: ' +  str(ndat)
    sys.stdout.flush()   
    print("Completed: ", txt, '>> fn_MCTW', flush=True)           
    sys.stdout.flush()
    
    # should return xMCTW for pll to work!!?? - check
    # how to reconcile output xMCTW with input xL_MCTW[i] ???
    
    arg_out = [ip, meanER, err_meanER, tblp]
    if iexp == nexp-1:
        # only last exp. run should return MC table
        # else no memory is left
        arg_out = [ip, meanER, err_meanER, tblp, tbl] 
        
    return arg_out

#@njit
def fn_sample(dist, ndat, boo_LHS, cfg, sseed=None):

    # obtain Di & b 2ndary dists, i.e. only +ves
    # a mult. of 1.2 ended with truncated_ndat < orig_ndat
    # assign distributions to 1st 2 cols. of tbl
    # for some reason, perhaps just for demo, I assign dist. 2 different ways
    # LHS has the loop embedded, NRS has the loop here
    # we use 1st two dist. (truncated versions)
    
    tbl = np.zeros((ndat,2))
    
    if boo_LHS == True:
        tbl[:,0:2] = fn_sample_distNorm_LHS(dist, ndat, cfg, sseed)
    else:
        for i in range(0,2):
            # 1st loop for Di, 2nd for b. 3 to 4 are non-truncated dists. not utilized here
            # consider sseed+1 to diff. b sampling base
            tbl[:,i] = fn_sample_distNorm_numpy(dist[i], ndat, cfg, sseed)

        # ic('NRS sampled')

    return tbl



# *****************************************************************************
# %% main fn.: 
# *****************************************************************************
def main():

    # set fixed params. & define xcfg etc. here 
    # so that all fn. & clss can access it:
    xcfg = clss_MCTW_cfg()

    # define distributions
    xL_dist = []
                                 # mu, sig, boo_trunc, LB, UB, name
    xL_dist.append(clss_dist_Norm(0.5, 0.3, True, xcfg.Dexp, 0.9999, "Di_truncated"))
    xL_dist.append(clss_dist_Norm(3.0, 1.0, True, 0.0, np.inf, "b_truncated"))    
    # xL_dist.append(clss_dist_Norm(3.0, 1.0, 0.0, 12, "b")) 
    
    xL_dist.append(clss_dist_Norm(0.5, 0.3, name="Di"))
    xL_dist.append(clss_dist_Norm(3.0, 1.0, name="b"))    
 
    # MC exp. settings
    #ndat = np.array([300, 999, 2000, 9999])
    ndat = np.array([3000, 9999, 99999, 999999])

    ndat[0] = fn_find_prime_sq(ndat[0]) # needed for Latin Hyper-Cube
    ndat[1] = fn_find_prime_sq(ndat[1]) # needed for Latin Hyper-Cube

    #nset = ndat.shape[0]
    nexp = 300

    # set MC exp. objs.
    xL_MCTW = fn_set_MCTW_objs(nexp, ndat, xL_dist, xcfg)
   

# %% Repeat MC exp. multiple times = nexp
    start = time.time()
    fn_loop_MCTW_exp(nexp, ndat, xL_dist, xcfg, xL_MCTW)

    end = time.time()
    elapsed_time = end - start
    ic(elapsed_time/60, ' mins')

# %% save results as binary file & exit   
    xglb = md_glb.clss_glb()    # needed for saving files


    
    txt = '_Nexp' + str(nexp) + '_nLHS' + str(ndat[1]) 
    if xcfg.pll: txt = txt + '_pll'
    pklf = xglb.srcdir + xglb.srcfile + txt + '.pkl'
    
    # Open a file and use dump()
    # A new file will be created
    with open(pklf, 'wb') as file: 
        pickle.dump(xL_dist, file)        
        pickle.dump(xL_MCTW, file)
        pickle.dump(xcfg, file)
        pickle.dump(xglb, file)

    ic('Results stored to: ' + pklf)

    print('Call to main function')
    return

if __name__ == '__main__':
    main()

    raise SystemExit # also removes vars from mem.?

# %% APPENDICES
    # for k in range(n):
    #     tbl[:,k] = xL_MCTW[k].tblm[:,0]
    
    # these did not work::
    # ic(xdist[0:4].name)
    # txt = np.array(xdist[:].name) 
    # txt = []
    # txt.append(xdist[:].name)

    

    
    
