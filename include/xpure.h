#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mpi.h"
#include "s2hat.h"
#include "s2hat_pure.h"

#ifdef CNAG
#include <nag.h>
#include <nag_stdlib.h>
#include <nagf07.h>
#include <nagf06.h>
#elif CBLAS
#include <cblas.h>
#include <clapack.h>
#elif ACML
/* #include <acml.h> */
#elif ESSL
#include <essl.h>
#elif MKL
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#elif GSL
#include <gsl/gsl_blas.h>
#include <gsl/gsl_blas_types.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_permutation.h>
#elif LIBSCI
#include <cblas.h>
#include <lapacke.h>
#endif


//Table of contents
//-----------------
//    - Xpol maker
//    - format tools
//    - alms tools
//    - FITS tools
//    - Mll tools
//    - error bars

#ifndef UNDEF_HEALPIX_VALUE
#define UNDEF_HEALPIX_VALUE ( (const double)(-1.6375e30) )
#endif

#ifndef DBL_MAX
#define DBL_MAX            1.79769313486231470e+308
#endif

#ifndef MIN
#define MIN(x,y) (x<y ? x : y) /* Minimum of two arguments */
#endif
#ifndef MAX
#define MAX(x,y) (x>y ? x : y) /* Maximum of two arguments */
#endif
#ifndef SQR
#define SQR(a) ((a)*(a))
#endif
// int fullsky;
// int rank;
// int root, gangroot;


#define HEAL_UNDEF(a) ( fabs(a-UNDEF_HEALPIX_VALUE) < 1e-5*fabs(UNDEF_HEALPIX_VALUE) )



s2hat_int4 compute_all_xls( s2hat_int4 nmaps, s2hat_int4 noStokes, s2hat_int4 nlmax, s2hat_int4 nmvals, s2hat_int4 *mvals, s2hat_int4 lda,
			    s2hat_dcomplex *local_alm, 
			    s2hat_int4 nxtype, s2hat_flt8 *xcl, s2hat_int4 *frstSpec, s2hat_int4 *scndSpec,
			    s2hat_int4 gangSize, s2hat_int4 n_gangs, s2hat_int4 my_gang_no, s2hat_int4 my_gang_rank, s2hat_int4 gang_root, MPI_Comm my_gang_comm, MPI_Comm global_comm);







//=========================================================================
//General tools (xpol_tools_s2hat)
//=========================================================================
void PCl( int nele, int nbin, double *matp, double *pseudocl, double *pseudocb);
void PMQ( int nbin, int nele, double *P, double *V, double *Q, double *M);
double SolveSystem( int nele, double *mat, double *cell);

void BinShifts( int nbins, char* mask_list, int *binshift, int *nbin_per_mask, int rank);
void GatherCell( int nbins, int bshift, int npm, double *cell, MPI_Comm root_comm);
void GatherMll( int nbins, int bshift, int npm, double *mbb, MPI_Comm root_comm);


//=========================================================================
//IO tools
//=========================================================================
int get_parameter(const char *fname, const char *nom, char *value);
int get_vect_size( char *infile, int lmax);
int read_fits_vect( char *infile, double *vect, int lmax, int colnum, int def);
void read_fits_map( int nside, double *map, char *infile, int col);
void read_TQU_maps( int nside, double *map, char *infile, int nstokes);

void read_distribute_map( s2hat_pixeltype cpixelization, s2hat_int4 nmaps, s2hat_int4 mapnum, int ncomp, 
			  char *mapname, s2hat_int4 first_ring, s2hat_int4 last_ring, s2hat_int4 map_size, 
			  double *local_map, s2hat_int4 myid, s2hat_int4 numprocs, s2hat_int4 root, MPI_Comm comm);

void write_fits_vect( int nele, double *vect, char *outfile, int hdu);
void write_fits_map( int nstokes, int npix, double *map, char *outfile);

void collect_write_map( s2hat_pixeltype cpixelization, s2hat_int4 nmaps, s2hat_int4 mapnum, s2hat_int4 nstokes,
                        char *mapname, s2hat_int4 first_ring, s2hat_int4 last_ring, s2hat_int4 map_size,
                        s2hat_flt8 *local_map, s2hat_int4 myid, s2hat_int4 numprocs, s2hat_int4 root, MPI_Comm comm);
void collect_write_mll( int nlmax, char *mllfile, double *local_mll, s2hat_int4 myid, s2hat_int4 root, MPI_Comm comm);
