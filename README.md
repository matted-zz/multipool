Multipool 0.10.1
==============

[![Build Status](https://travis-ci.org/matted/multipool.svg)](https://travis-ci.org/matted/multipool) [![Coverage Status](https://coveralls.io/repos/matted/multipool/badge.svg)](https://coveralls.io/r/matted/multipool)

See the [wiki page](https://github.com/matted/multipool/wiki) for more
details, including usage examples and installation instructions.


usage: mp_inference.py [-h] -n N [-m {replicates,contrast}] [-r RES] [-c CM]
                       [-t FILTER] [-np] [-o OUTFILE] [-v]
                       countfile [countfile ...]

Multipool: Efficient multi-locus genetic mapping with pooled sequencing,
version 0.10. See http://cgs.csail.mit.edu/multipool/ for more details.

positional arguments:
  countfile             Input file[s] of allele counts

optional arguments:
  -h, --help            show this help message and exit
  -n N, --individuals N
                        Individuals in each pool (required)
  -m {replicates,contrast}, --mode {replicates,contrast}
                        Mode for statistical testing. Default: replicates
  -r RES, --resolution RES
                        Bin size for discrete model. Default: 100 bp
  -c CM, --centimorgan CM
                        Length of a centimorgan, in base pairs. Default: 3300
                        (yeast average)
  -t FILTER, --truncate FILTER
                        Truncate possibly fixated (erroneous) markers.
                        Default: true
  -np, --noPlot         Turn off plotting output.. Default: false
  -o OUTFILE, --output OUTFILE
                        Output file for bin-level statistics
  -v, --version         show program's version number and exit


Count file format: 

A whitespace delimited file with a row for each marker (SNP or small
indel).  The first column reports the locus position in base pairs
(used with the --centimorgan parameter to compute crossover
probabilities).  The second column reports the number of sequencing
reads from the first analyzed strain and the third column reports the
read count from the second strain.
