################################################################################################
# Makefile to compile SCALAR2SPIN@CORI and XPURE@CORI. 
# You need to load the following module: 
# >module swap <your-env> PrgEnv-intel
# >module load cfitsio
# >module load cray-fftw
# >module load gsl
# Be sure also that you have a INTEL environment (PrgEnv-intel) before loading modules!
# If your 
# Note that you may need to add your HEALPIX/lib and CFITSIO_DIR/lib path to your 
# LD_LIBRARY_PATH environment before running the binaries.
# Author: Giulio Fabbian, Julien Peloton 06.06.2021
# Comments: FabbianG@cardiff.ac.uk, peloton@lal.in2p3.fr
################################################################################################

################################################################################################
# Uncomment it if you do not have a custom version of healpix and/or CFITSIO installed
#HEALPIX=/global/u1/f/fabbian/software/Healpix_3.50
#CFITSIO=/global/homes/f/fabbian/software/cfitsio/cori/intel

HEALPIX=/global/homes/m/mag/software/Healpix_3.70

S2HAT=/global/homes/f/fabbian/software/s2hat_v2.55_beta
CFITSIO=/global/homes/m/mag/software/cfitsio-4.1.0
SPRNG=/global/homes/f/fabbian/software/sprng2.0
INC_DIR = -I${GSL_DIR}/include -I${S2HAT}/include -I${HEALPIX}/include -I${CFITSIO}/include -I${SPRNG}/include
LIB_DIR = -L${GSL_DIR}/lib -L${HEALPIX}/lib -L${S2HAT}/lib/cori/intel/ -L${CFITSIO}/lib -L${SPRNG}/lib

CC  = cc -g -O3
LL  = ftn -lifcore -nofor_main 
F77 = ftn 

# # you need to define you linear algebra library. NERSC supports multiple libraries
# # Uncomment the relavant part.
# # libsci/gsl/mkl are supported by NERSC at the time of writing this script
# # but acml require custom installation
LINALG_LIB =  -lgsl -lgslcblas 
LINALG_FLAG = -DGSL

#LINALG_LIB = -lsci_intel
#LINALG_FLAG = -DLIBSCI -marc=native

#LINALG_LIB = -mkl  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
#LINALG_FLAG = -march=native -DMKL

# require custom installation
#LINALG_LIB = -lacml
#LINALG_FLAG =-march=native -DACML

# if you compiled sprng with gmp link -lgmp. If you compiled everything with gnu link also -lgfortran 
LIB = -ls2hat_std -lhealpix -lchealpix  -lsprng -lcfitsio -lifcore -lm ${LINALG_LIB}

FLAG  = -DDEBUG -O3 -m64 -static -fPIC ${LINALG_FLAG} -DFITS -DHEALPIXDATA=\"${HEALPIX}/share/healpix/\"
FFLAG = -DDEBUG
LFLAG = -DDEBUG
################################################################################################



OBJFILE = wig3j_f.o xpure_tools.o xpure_io.o compute_all_xls.o s2hat_map2purealm.o

all :
		gmake -f ${MAKEFILE_LIST} ${OBJFILE}
		gmake -f ${MAKEFILE_LIST} xpure xpure_create_mll scalar2spin optimalmasks_PCG
		mkdir -p ../build/cori
		mv scalar2spin ../build/cori
		mv xpure xpure_create_mll ../build/cori
		mv optimalmasks_PCG ../build/cori
		chgrp -R mp107 ../
		chmod -R g+rwX ../
		

obj :
		gmake -f ${MAKEFILE_LIST} ${OBJFILE}

clean :
		rm -f ${OBJFILE} *.o

%.o: %.c
		${CC} ${FLAG} ${INC_DIR} -c -o $@ $<

wig3j_f.o: wig3j_f.f
		${F77} ${FFLAG} -c wig3j_f.f

scalar2spin: ${OBJFILE} scalar2spin.o
		${CC} ${LFLAG} ${OBJFILE} scalar2spin.o ${LIB_DIR} ${LIB} -o scalar2spin

optimalmasks_PCG : ${OBJFILE} optimalmasks_PCG.o
		${CC} ${LFLAG} ${OBJFILE} optimalmasks_PCG.o ${LIB_DIR} ${LIB} -o optimalmasks_PCG

xpure: ${OBJFILE} xpure.o
		${CC} ${LFLAG} ${OBJFILE} xpure.o ${LIB_DIR} ${LIB} -o xpure

xpure_create_mll: ${OBJFILE} xpure_create_mll.o
		${CC} ${LFLAG} ${OBJFILE} xpure_create_mll.o ${LIB_DIR} ${LIB} -o xpure_create_mll

s2hat_pure: ./s2hat_pure.h
