# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:54:03 2021
@author: Doruk Alp, PhD
updated: Feb. 2024
updated: July, Aug, Oct, Dec 2023
updated: May 2021
1st vers: Jan. 2021 

    Format matplotlib plots
    
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
#
# STEP 1. Import needed libs:
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mp_lines 
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#
# font settings
#
font_size = int(8)
title_size = int(10)
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

plt.rc('font', size=font_size)          # controls default text sizes
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#
# List of colors:
#
# did not work::
# colrmap = plt.get_cmap('rainbow', 10) 
# colors = colrmap.colors 
# plt.get_cmap(None, None)

Lcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#colorsN = plt.get_cmap('rainbow')(np.linspace(0, 1, 8))
#colorsN = list(colors._colors_full_map.values())
#
# transparency settings
#
alf = 0.5 # sets transparency to 50%

#
# List of markers:
#
from matplotlib.lines import Line2D
Dmark = Line2D.markers
#Lmark = list(Dmark.values())
Lmark = list(Dmark.keys())

# Lmark.insert(1, '.')
# Lmark.insert(1, Lmark.pop(17))
# Lmark.insert(4, Lmark.pop(18))

for i in range(0, 2): Lmark.pop(0)
for i in range(0, 5): Lmark.pop(5)
for i in range(0, 4): Lmark.pop(30)

Lmark.insert(0, Lmark.pop(11))
Lmark.insert(0, Lmark.pop(11))



mark_size = 14 # mark_size for plot is ~1/3 of scatter


LstyLine = list(mp_lines.lineStyles.keys())
for i in range(0, 3): LstyLine.pop(4)


#
# List of subplot designations:
#
Lsplt = '(a) ', '(b) ', '(c) ', '(d) ', '(e) ', '(f) ', '(g) ', '(h) '

#
# default formatting for plots
#

def plt_format(axs, xlabel, ylabel, title=None, ylabel2=None):

    if title is not None: axs.set_title(title, size=title_size, weight='bold')
    if ylabel2 is None: 
        axs.set_ylabel(ylabel, size=font_size)
        axs.set_xlabel(xlabel, size=font_size)
    axs.minorticks_on()
    axs.tick_params(axis='both', which='both', labelsize=font_size)
    axs.grid(True, which='both', axis='both')
    axs.grid(True, which='minor', axis='both', ls=':')
    axs.legend(loc='best', fontsize= font_size)

# axs.yaxis.set_major_locator(MultipleLocator(10))
# axs.legend(bbox_to_anchor = (1.01, 0.2))

    # ax.set_title('data_qq',fontsize=15)
    # ax.xaxis.get_label().set_fontsize(12)
    # ax.yaxis.get_label().set_fontsize(12)
  
    return

def main():
    
    plt.rcParams["figure.figsize"]
    
    return

if __name__ == '__main__':
    main()


#
# %% APPENDICES
#

# character description
# '-'       solid line style
# '--'      dashed line style
# '-.'      dash-dot line style
# ':'       dotted line style
# '.'       point marker
# ','       pixel marker
# 'o'       circle marker
# 'v'       triangle_down marker
# '^'       triangle_up marker
# '<'       triangle_left marker
# '>'       triangle_right marker
# '1'       tri_down marker
# '2'       tri_up marker
# '3'       tri_left marker
# '4'       tri_right marker
# 's'       square marker
# 'p'       pentagon marker
# '*'       star marker
# 'h'       hexagon1 marker
# 'H'       hexagon2 marker
# '+'       plus marker
# 'x'       x marker
# 'D'       diamond marker
# 'd'       thin_diamond marker
# '|'       vline marker
# '_'       hline marker
