# WARNING : Forked version of Xpure -- work in progress before merging
Please download the main version of Xpure on https://gitlab.in2p3.fr/xpure/Xpure

We extend the pure pseudo-power-spectrum formalism proposed recently in the context of the cosmic microwave background polarized power spectra estimation by Smith (2006) to incorporate cross-spectra computed for multiple maps of the same sky area. We present an implementation of such a technique, paying particular attention to a calculation of the relevant window functions and mixing (mode-coupling) matrices. We discuss the relevance and treatment of the residual E/B leakage for a number of considered sky apodizations as well as compromises and assumptions involved in an optimization of the resulting power spectrum uncertainty. In particular, we investigate the importance of a pixelization scheme, patch geometry, and sky signal priors used in apodization optimization procedures. In addition, we also present results derived for more realistic sky scans as motivated by the proposed balloon-borne experiment EBEX. We conclude that the presented formalism, thanks to its speed and efficiency, can provide an interesting alternative to the CMB polarized power spectra estimators based on the optimal methods at least on angular scales smaller than ~10. In this regime, we find that it is capable of suppressing the total variance of the estimated B-mode spectrum to within a factor of ~2 of the variance due to only the sampling and noise uncertainty of the B modes alone, as derived from the Fisher matrix approach.



Developpers
-----------
* G. Fabbian
* J. Grain
* J. Peloton
* R. Stompor
* M. Tristram


Publications
------------

* Ferté, Peloton, Grain, Stompor, Phys. Rev. D 92, 083510 (2015)
* Ferté, Grain, Tristram, Stompor, Phys. Rev. D 88, 023524 (2013)
* Grain, Tristram, Stompor, Phys. Rev. D 86, 076005 (2012)
* Grain, Tristram, Stompor, Phys. Rev. D 79, 123515 (2009)


Prerequisites
-------------

* Healpix C libraries (http://healpix.jpl.nasa.gov)
* CFITSIO (http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
* pureS2HAT (http://www.apc.univ-paris7.fr/APC_CS/Recherche/Adamis/MIDAS09/software/pures2hat/pureS2HAT.html)
* FFTW (http://fftw.org)
* SPRNG (http://sprng.org. Compatible with version < 4.0, recommended version 2.0)


Installing
----------

With **CMAKE**, you can install Xpure using the following commands :


```
cmake -S . -B build --install-prefix <path>
cmake --build build
cmake --install build
```

or

```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<prefix>
cmake --build build
cmake --install build
```

having exported in your environment the paths to s2hat and healpix if needed


Or, you can as well run the following commands for the configure :

* CC=mpicc ./waf configure --s2hat=${PATH_S2HAT} --healpix=${PATH_HEALPIX}
* CC=mpicc ./waf build
* ./waf install



Examples
--------

On ADAMIS cluster: 

* CC=/usr/bin/mpicc ./waf configure --s2hat=/usr/local/s2hat --healpix=/home/lejeune/soft/healpy-0.10.2/hpbeta/
* CC=/usr/bin/mpicc ./waf build

On Hopper: 

* module swap PrgEnv-pgi PrgEnv-gnu 
* module load fftw/3.3.0.0
* module load acml
* ./waf configure --platform=hopper --s2hat=/global/homes/r/radek/s2hat/lib/hopper/v2.55/gnu --healpix=/project/projectdirs/cmb/modules/hopper/gnu/cmb/2.5.1/healpix_2.15a-2.5.1 --cfitsio=/project/projectdirs/cmb/modules/hopper/gnu/cmb/2.5.1/cfitsio_3.25-2.5.1 --sprng=/usr/common/usg/sprng/2.0/gnu --fftw=/usr/common/usg/fftw/3.3.0.0/x86_64

   

On Edison: 

* module swap PrgEnv-intel PrgEnv-gnu 
* module load fftw cfitsio
* module load sprng gsl
* ./waf configure --platform=edison --s2hat=/global/homes/r/radek/s2hat/lib/edison/gnu --healpix=/project/projectdirs/cmb/modules/edison/hpcports_gnu/healpix-2.15a --cfitsio=/project/projectdirs/cmb/modules/edison/hpcports_gnu/cfitsio-3.31 --sprng=/project/projectdirs/cmb/modules/hopper/gnu/cmb/2.5.1/sprng_2.0-2.5.1 --gsl=/usr/common/usg/gsl/1.15/gnu
