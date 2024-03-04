# Type-well generation tool for evaluating unconventional oil & gas fields

Currently, the code is left as-is, i.e. as a script/module, rather than a package.

There are 2 main files to run:

1) md_MonteCarlo_v5_test.py

Main file to edit & run. Distribution types, ranges, sample & experiment counts are specified within this file. Change these parameters and run the file. Results are saved in an external pickle file (.pkl), in the same directory as the .py file.

2) md_MonteCarlo_v5_plt.py

Loads the results from .pkl file (should be in the same directory), and generates various plots. Recommended use is to execute each code block ("cell") separately, as necessary. Cells are usually independent.
.pkl file path may need to be corrected within the file.

NB: I mostly ran the code within an IDE (Spyder). Hence, my plots were created within the IDE. If you ran the code from commandline, you may need to close pop-up figures, for the code to continue execution of next block.     

There are couple other dependency modules:

a) md_glb: a weak implementation of global variable space. 

b) md_Arps: Arps (1945) rate, cumulative prod., trapezoidal rule equations, and their 2 segment application. 

## Description

For some background on the purpose of the code please see:

Alp, D. (2023). Impact of Numerics on Monte-Carlo Based Type-Well Generation for Unconventional Fields. 21th International Petroleum and Natural Gas Congress and Exhibition of Turkiye. IPETGAS Held in Ankara, Turkiye. September 27-29, 2023, p40-49. https://www.ipetgas.org/IPETGAS-K%C4%B0TAP.pdf

I have drafted an expanded version ready for submission to a journal. I shall add reference to the final publication here in due time. 

## Getting Started

### Dependencies & LICENSE info
#### md_MonteCarlo_v5_test.py
|     Package     |                        License                        |
| :-------------: | :---------------------------------------------------: |
| multiprocessing | No additional license, possibly Python built-in module |
|      numpy      |                      BSD-3-Clause                     |
|      pickle     | No additional license, possibly Python built-in module |
|       sys       | No additional license, possibly Python built-in module |
|       time      | No additional license, possibly Python built-in module |

#### md_MonteCarlo_v5_plt.py
|   Package   |                        License                        |
|-------------|-------------------------------------------------------|
|  matplotlib |                          PSF                          |
|    numpy    |                      BSD-3-Clause                     |
|    pandas   |                  BSD 3-Clause License                 |
|    pickle   | No additional license, possibly Python built-in module |
|   seaborn   | No additional license, possibly Python built-in module |
| statsmodels |                      BSD License                      |
|     time    | No additional license, possibly Python built-in module |

### Installing

* You will need to download files from this repository. It is assumed that to-be-user is somewhat familiar with running python code on local PC.
* Any modifications needed to be made to files/folders: None I can remember.

### Executing program

* How to run the program: I recommend within an IDE, but works from cmdline as well.

## Help

It is a basic code, but if you wish to get in touch welcome to email me at dorukalp.edu@gmail.com. Please note, it may be a few days before I can respond.

## Authors

Doruk Alp, PhD

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

The template to this README:
[DomPizzie](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)

Other templates for an expanded README file:
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)

