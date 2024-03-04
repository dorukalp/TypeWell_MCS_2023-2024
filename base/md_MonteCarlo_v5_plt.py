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
# *****************************************************************************
# %% Std. libraries: 
# STEP 1. Import needed libraries.
# *****************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

#import scipy.stats as st 
# vs from scipy import stats as st
from scipy.stats import qmc, norm, truncnorm, mode, gaussian_kde
# vs from scipy.stats import qmc as st_qmc

from statsmodels.distributions.empirical_distribution import ECDF

import seaborn as sns
sns.set_theme(style="ticks")

from datetime import datetime

import pandas as pd
    
from icecream import ic
import time
import pickle

from numba import njit, jit
# from numba.types import FunctionType

# Own libraries::
import md_glb # access & change var stored in var_glb
import md_Arps
import md_plt


# *****************************************************************************
# %% Funcs. Stand Alone 
# *****************************************************************************

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

    return ip #, ipN, ipK

#@njit
def plt_CDF(a):
    # get x,y data for CDF plot
    # this may not redundant, 
    # seaborn or other packages may have a 1-liner for this
        
    dims = a.shape
    ndim = len(dims)
    if ndim > 1:
        a = np.reshape(a, (dims(0), 0))
        
    myECDF = ECDF(a)
    plt_x = myECDF.x
    plt_y = myECDF.y
    return plt_x, plt_y

#@njit
def plt_QQ(x1, lbl1, x2=None, lbl2=None, name=None, axN=None):

    import statsmodels.api as sm    # to create Q-Q plot
    
    # dist=stats.norm, # add to qqplot args.
    alf = 1.0 # md_plt.alf
    
    if axN is None:
        fig, axN = plt.subplots()
    
    sm.qqplot(x1, fit=True, line='45', \
                marker=md_plt.Lmark[0], \
                markersize = md_plt.mark_size/2, \
                markerfacecolor='b', markeredgecolor='b', \
                fillstyle='none', \
                alpha=alf, \
                label=lbl1, \
                ax = axN)
        
    if x2 is not None:
        sm.qqplot(x2, fit=True, line='45', marker=md_plt.Lmark[2], \
                    markersize = md_plt.mark_size/3, \
                    markerfacecolor='k', markeredgecolor='k', \
                    fillstyle='none', \
                    alpha=alf, \
                    label=lbl2, \
                    ax = axN)
        
        axN.relim()                 # recalc the ax.dataLim
        axN.autoscale()             # update ax.viewLim using new dataLim
        # axN.autoscale_view()      # update ax.viewLim using new dataLim
        # axN.set_aspect('equal')   # made a box, i.e. scaled the axes did not change limits
        # axN.axis('equal')         # ??
                
    if name is None: 
        txt  = 'Q-Q Plot of unknown entity'
    else:
        txt = name
        
    md_plt.plt_format(axN, 'Theoretical Quantiles', 'Sample Quantiles', txt)

    return


def plt_nexp_vs_meanER(xL_MCTW, str_legend):
    
    str_title = "Num. of experiments vs mean EUR"
        
    ylabel = 'mean EUR [vol. unit]'
    xlabel = "Number of MC experiments"

    pltx = np.arange(0,nexp,1)          
    fig, axs = plt.subplots()
    
    n = len(xL_MCTW)
    
    for k in range(n):
        plty =xL_MCTW[k].tblm[:,0]
 
        axs.plot(pltx, plty, label=str_legend[k], \
                              marker = md_plt.Lmark[k], \
                              markersize = md_plt.mark_size/3, \
                              linewidth = 2, \
                              linestyle='None', \
                              fillstyle='none')

    md_plt.plt_format(axs, xlabel, ylabel, str_title)
    plt.draw()
    # ic(md_plt.colorsN)

    return

def plt_contourER(cfg, vals):
    # contour map of ERs for the Di & b range
    
    n = len(vals)
 
    str_title = 'Contour Plot of dimensionless ER values'
    xlabel = 'Di values' 
    ylabel = 'b values'
    
    cntr_levels = [1,5,10,20,30,40,60,80,100,120,160]
    # cntr_levels = 10 # if on, x-axis may be put on log scale     
    
    val_x = vals[0] # Di values 

    fig, axs = plt.subplots(n-1, 1, constrained_layout=True, figsize=(8,6))

    for ii in range(1, n):   
        val_y = vals[ii] # b values

        # Create 2D grid points from initial values  
        [plt_x, plt_y] = np.meshgrid(val_x, val_y) 
    
        ny, nx = plt_x.shape 
        # nx = plt_x.shape[0] 
        # ny = plt_x.shape[1]
    
        plt_z = np.zeros((ny,nx)) # ER values

        # ER = np.zeros(n)
        # cMCTW.tbl[:,0] = 
        # plt_x = cMCTW.tbl[:,0]
        # plt_y = cMCTW.tbl[:,1]
        # n = ndat.shape[0]
        # r, c = cMCTW.tbl.shape
        
        for j in range(0, nx):
            for i in range(0, ny):
                # loop over sample sets (each fake well), calc & store EUR
                valp = [plt_x[i,j], plt_y[i,j], cfg.Dexp]
                ai, tExp, aexp, plt_z[i,j] = md_Arps.fn_ER(cfg.tmax, valp)

        # changing limits do not help, contours/plot goes out of screen # !!!
        # one soln. is to plot 2 regions as subplot
        # this requires splitting the data
        # ax.set_ylim(0.0, 0.2)  
        cntr = axs[2-ii].contour(plt_x, plt_y, plt_z, cntr_levels, colors='k') 
        # ax.set_yscale('log')
        
        md_plt.plt_format(axs[2-ii], '', '', '')
        axs[2-ii].get_legend().remove()
        axs[2-ii].clabel(cntr, inline=True, fontsize=8)
        # if 2-ii == 0:
        # #     axs[2-ii].set_xlabel=None
        #      axs[2-ii].set_xticklabels=[]


    # axs[0].set_xlabel=None
    axs[0].set_xticklabels([])
            
    fig.suptitle(str_title) # single title for subplots
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    plt.draw()
  
    return #plt_x, plt_y



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
# %% Funcs. using classes & modules
# *****************************************************************************

def fn_compare_dist_trunc_QQ(xL_dist, ndat, cfg):

    n = len(xL_dist)
    nprm = fn_find_prime_sq(ndat) # num. of samples
    arr = np.zeros((nprm, n,2))   # array to hold samples, x2 for LHS

    sseed = 15  # fix sseed for proper comparison. 

    #str_label = np.array([xL_dist[i].name for i in range(n)])
    str_label = [xL_dist[i].name for i in range(n)]
    # QQ plot does not require axis labels
    # ylabel = 'Frequency'
    # xlabel = "Dimensionless ER30 values [vol. units]"
    str_title = [None] * 4

    for i in range(0,2):
        # i = 1 for Di, = 2 for b
        # i+2 used not truncated dist.
        str_title[i] = 'Naive Random Sampling ' + str_label[i] + " vs " + str_label[i+2]
        str_title[i+2] = 'Latin Hypercube Sampling ' + str_label[i] + " vs " + str_label[i+2]

    for i in range(0,2):    # this loop uses NRS
        # i = 1 for Di, = 2 for b
        # i+2 used not truncated dist.
                
        arr[:,i,0] = fn_sample_distNorm_numpy(xL_dist[i], nprm, cfg, sseed) 
        arr[:,i+2,0] = fn_sample_distNorm_numpy(xL_dist[i+2], nprm, cfg, sseed)
        #plt_QQ(arr[:,i], str_label[i], arr[:,i+2], str_label[i+2], name=title)

    for ii in range(0,2):    # this loop uses LHS
    # due to my implementation, LHS can not handle single xL_MCTW obj.
    # will submit 2 dist. at once.
    # 1st loop submits trunc. Di & b together, 2nd loop submits non-truncs
        i = ii * 2
        tbl = fn_sample_distNorm_LHS(xL_dist[i:i+2], nprm, cfg, sseed)
        arr[:,i,1] = tbl[:,0] # samples of Di
        arr[:,i+1,1] = tbl[:,1] # samples of b
        

    nrow = 2
    ncol = 2
    nplot = nrow * ncol
        
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, 
                        constrained_layout=True, figsize=(8, 6))
    ii = 0
    ij = 0
    k = 0   # index for NRS
    for i in range(0,4):    # 4 subplots

        ii = i
        if i > 1:
            ii = i - 2
            ij = 1        
            k = 1   # after 2nd plot, swithc to LHS    

        # lw = 2.0
        # if j == 0: lw = 3.0
        
        # => to plot on a previously gen. axis
        # 'QQ Plot of '
        title = md_plt.Lsplt[i] + str_title[i]
        plt_QQ(arr[:,ii,k], str_label[ii], arr[:,ii+2,k], str_label[ii+2], \
               name=title, axN = axs[ii,ij])
            
        #md_plt.plt_format(axs[ii, ij], xlabel, ylabel, str_title)
                    

    fig.set_constrained_layout_pads(w_pad = 0., h_pad = 0., 
                                    hspace = 0.05, wspace = 0.05)
    plt.draw()
    return


def fn_sample(cMCTW, dist, ndat, mult=None, sseed=None):

    # obtain Di & b 2ndary dists, i.e. only +ves
    # a mult. of 1.2 ended with truncated_ndat < orig_ndat
    # assign distributions to 1st 2 cols. of tbl:
    
    if cMCTW.LHS == True:
        cMCTW.tbl[:,0:2] = fn_sample_distNorm_LHS(dist, ndat, mult, sseed)
    else:
        for i in range(0,2):    
            cMCTW.tbl[:,i] = fn_sample_distNorm_numpy(dist[i], ndat, mult, sseed)

    return


# *****************************************************************************
# %% main fn.: 
# *****************************************************************************

def main():
    # the plots in this section are sort of independent of generated
    # MC data.
    
    # contour plot of dless ER
    val = []
    val.append(np.arange(xL_dist[0].LB, xL_dist[0].UB, 0.01)) # Di values
    
    vali = 0.004
    val.append(np.arange(xL_dist[1].LB, vali, 0.002)) # b values
    val.append(np.arange(vali, 12, 0.1))   # b values
    
    # x-axs: Di, y-axs: b, contours: ER
    # shows that inverse problem (given ER, find Di & b)
    # has no unique solution. So, need to keep track of Di & b per each ER
    plt_contourER(xcfg, val) 

    # shows impact of rejecting -ves on QQ plot
    # 1E5 is too many, slows down but not sure if sampling or plotting
    a = md_plt.Lmark
    fn_compare_dist_trunc_QQ(xL_dist, 999, xcfg)
       
    return

if __name__ == '__main__':
# %% load result. var. from prev. run: 
    # Load prev. saved workspace by importing the pickle file 
    # Must keep class definitions on this file as well
    pklf = r"D:\Alp\myDrives\OneDrive\PNG\research\00_DCA_TypeWell_own"
    pklf = pklf + r"\python_actual\11_MC\md_MonteCarlo_v5_test_Nexp300_nLHS9409_pll.pkl"
    
    with open(pklf, 'rb') as file:
        ic(file)
        xL_dist = pickle.load(file)
        xL_MCTW = pickle.load(file)
        xcfg = pickle.load(file) 
        xglb = pickle.load(file) 
   
# %% base plots independent of generated MC data.

    # Suppress displaying plots, 
    # i.e.set backend to non-interactive (silent mode)
    # matplotlib.use('Agg') => did not work.  
    
    main()
        
# %% set series labels/legend

    #ndat = np.zeros(nset)
    ndat = []   
    for obj in xL_MCTW:
        r, c = obj.tbl.shape
        # ndat[i] = r
        ndat.append(r)
    
    ndat = np.array(ndat)
    nset = ndat.shape[0] # nset = len(xL_MCTW)

    r, c = xL_MCTW[0].tble.shape
    
    nexp = r/4

    str_legend = []   
    for i in range(0, nset):
        if i < nset/2:
            str_legend.append('LHS, samples = ' + str(ndat[i]))
        else:
            str_legend.append('NRS, samples = ' + str(ndat[i]))
            


# %% joint plot of single MC    
   # replicates seaborn joint plot
   # not needed, does not serve much of a purpose for now
#   plt_contourER_joint(tbl)

# %% joint plots of LHS & NRS with ndat = max.
# to compare LHS & naive MC.
# shows that too many samples required for NRS (naive random sampling)
# to replicate proposal distributions smoothly
 
    n_LHS, ncols = xL_MCTW[0].tbl[:,0:2].shape
    n_MC, ncols = xL_MCTW[2].tbl[:,0:2].shape
    
    str_title = ['Joint plot of Latin Hypercube Sampling, sample count = ' + str(n_LHS),\
                 'Joint plot of naive Random Sampling, sample count = ' + str(n_MC)]

    labelx = 'Di values'
    labely = 'b values'
    
    nrow = 1
    ncol = 2
    nplot = nrow * ncol
    
    # fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    # fig = plt.figure(figsize=(8, 6))
    # gs = GridSpec(nrow, ncol)
    #g = []
    for i in range(0,1):
        j = i
        if i == 1: j = 2
        
        # g = plt_snsjoint(xL_MCTW[j].tbl[:,0:2], labelx, labely, str_title[i])
        # md_plt.plt_format(axs[i], labelx, labely, str_title[i])
        # mg = SeabornFig2Grid(g, fig, gs[i])

        df = pd.DataFrame(xL_MCTW[j].tbl[:,0:2])
        df.rename(columns={0: labelx, 1: labely}, inplace=True)
    
        g = sns.jointplot(data=df, x=labelx, y=labely, kind="kde", color='k')
        g.plot_joint(sns.scatterplot, color="b", legend=False, s=12)
        g.fig.suptitle(str_title[i], fontsize=14, fontweight='bold')
        g.fig.subplots_adjust(top=0.93) # to leave space for title
        plt.draw()

    # gs.tight_layout(fig)
    #plt.draw() # 99999 points is too much for plotting

# # %% joint plots of LHS & MC with ndat = 999
# # to compare LHS & naive MC:
#     for i in range(0,2):
#         if i == 1: i = 3
#         plt_snsjoint(xL_MCTW[i].tbl[:,0:2])

# %% plot nexp vs mean ER, show change with repetition of MC 
    plt_nexp_vs_meanER(xL_MCTW, str_legend)
    

# %% Set tblf, Plot PDF %ER30s from all MC exp.
# shows that variation narrows down with increasing sample count & LHS 

    str_label = ['P10','P50','P90','Mean']
    ylabel = 'Frequency'
    xlabel = "Dimensionless ER30 values [vol. units]"

    nrow = 2
    ncol = 2
    nplot = nrow * ncol
        
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, 
                        constrained_layout=True, figsize=(8, 6))
    ii = 0
    ij = 0
    for i in range(0,4):    # 4 subplots

        ii = i
        if i > 1:
            ii = i - 2
            ij = 1        
        
        for j in range(0,4): # 4 data sets: LHS 1,2, NRS 1, 2
            # filter MC exp. results for each pcentile
            # plot pcentile ERs hist. from all MC exps. 
            # find index of mean or mode, and store data (Di, b etc) for plotting
            
            # filter the table by pcentile, store as temp. tbl:
            tblf = xL_MCTW[j].tble[xL_MCTW[j].tble[:,10]==i]

            # find the indices of rows corresponding to mean & pcentiles,
            # send filtered table to pertinent function:
            ip = fn_idx_stats(tblf[:,6])
            # store mean of pcentile values from all exp.
            xL_MCTW[j].tblf[i] = tblf[ip[3],:]
    
            # plot pdf of filtered pcentile values:
            df = pd.DataFrame(tblf)
            
            lw = 2.0
            if j == 0: lw = 3.0
            
            # => to plot on a previously gen. axis
            sns.kdeplot(df[6], ax = axs[ii, ij], \
                        color = md_plt.Lcolors[5-j], \
                        linestyle = md_plt.LstyLine[3-j], \
                        linewidth = lw, \
                        label=str_legend[j])
            
        str_title = md_plt.Lsplt[i] \
            + 'PDF of ' + str_label[i] + ' ER30 values from all MC experiments'
        md_plt.plt_format(axs[ii, ij], xlabel, ylabel, str_title)
                    

    fig.set_constrained_layout_pads(w_pad = 0., h_pad = 0., 
                                    hspace = 0.05, wspace = 0.05)

    plt.draw()

# %% Final q tbl corresp. to mean of pcents from all MC exp.
# for LHS case only

    # calc. q at t values for data sets from final tbl:         
    for j in range(0,4):        # loop for data set
        for i in range(0,4):    # loop for pcentiles
            ai = xL_MCTW[j].tblf[i,3]
            b = xL_MCTW[j].tblf[i,1]
            ts = xL_MCTW[j].tblf[i,4]
            aexp = xL_MCTW[j].tblf[i,5]        
            
            xL_MCTW[j].tblfq[:,i] = md_Arps.fn_Arps_2seg_multt(ai, b, xcfg.arrt, ts, aexp, bexp=None, qi=None)


# %% Plot type-wells corresp. to Final q tbl
    str_title = "Type-wells based on means of percentiles from MC experiments"
    ylabel = 'Dimensionless Flow Rate q [v.u./t.u.]'
    xlabel = "Time [t.u.]"
                
    pltx = xcfg.arrt
          
    fig, axs = plt.subplots()
    
    for i in range(4):
        plty =xL_MCTW[1].tblfq[:,i]
 
        axs.plot(pltx, plty, label=str_label[i], \
                              marker = md_plt.Lmark[i], \
                              markersize = md_plt.mark_size/5, \
                              linewidth = 2, \
                              linestyle='None', \
                              fillstyle='none')

    md_plt.plt_format(axs, xlabel, ylabel, str_title)
    axs.set_yscale('log')
    plt.draw()



# %% save figs as files
    str_stamp = str(datetime.now())
    str_stamp = str_stamp.replace(":", "")
    str_stamp = str_stamp.split(".", 1)[0]
    
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'Fig_{i}_{str_stamp}.svg', format="svg")
      
    raise SystemExit # also removes vars from mem.?

