/*****************************************************
 *                                                   *
 *  COMPUTE OPTIMAL WINDOW FUNCTION WITHIN EACH BIN  *
 *                                                   *
 *****************************************************/

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
/*#include <stdint.h>*/
#include <math.h>
#include <malloc.h>
#include <sys/stat.h>

#include "chealpix.h"
#include "fitsio.h"
#include "s2hat.h"

#define HEALPIX_undef -1.6375e30 /* Healpix convention */
#define SQR(a,b) ((a)*(b))
#define MIN(x,y) (x<y ? x : y) /* Minimum of two arguments */
#define MAX(x,y) (x>y ? x : y) /* Maximum of two arguments */

struct distrib_prop
{
  int nmvals;
  int firstringvalue;
  int lastringvalue;
  int mapsizevalue;
  long long int nplmvalue;
};

int get_parameter(const char *fname, const char *nom, char *value);
void read_fits_map( int nside, double *map, char *infile, int col);
void write_fits_map( int nstokes, int npix, double *map, char *outfile);

//WIGNER COMPUTATION
////////////////////

int wig3j( double l2, double l3, double m2, double m3, double *THRCOF);
void wig3j_( double *L2, double *L3, double *M2, double *M3,
	       double *L1MIN, double *L1MAX, double *THRCOF, int *NDIM, int *IER);

//IN THE SPIN_WEIGHTED APPROACH (RELAXED CONDITIONS)
////////////////////////////////////////////////////

void compute_spin_wlm( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
		       s2hat_pixeltype pixelisation, s2hat_scandef scan, 
		       s2hat_pixparameters pixparam, int pixchoice,
		       int nlmax, int nmmax, int *mvals,
		       struct distrib_prop disprop, 
		       double *local_w8ring,
		       double *local_window_scal, double *local_window_vect, double *local_window_tens,
		       s2hat_dcomplex *local_wlm_scal, s2hat_dcomplex *local_wlm_vect, s2hat_dcomplex *local_wlm_tens);
void compute_spin_window( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			  s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			  s2hat_pixparameters pixparam, int pixchoice,
			  int nlmax, int nmmax, int *mvals,
			  struct distrib_prop disprop,
			  double *local_mask, 
			  s2hat_dcomplex *local_wlm_scal, s2hat_dcomplex *local_wlm_vect, s2hat_dcomplex *local_wlm_tens,
			  double *local_win_sig_scal, double *local_win_sig_vect, double *local_win_sig_tens);
void spin_precompute_matrix( int nlmax, int nmmax, int *mvals,
			     struct distrib_prop disprop, 
			     int llow, int lhigh,
			     double *local_Cl, double fwhm,
			     double *local_S00, double *local_S11, double *local_S22, double *local_S01, double *local_S02, double *local_S12);
void spin_wlmXsignal( int nlmax, int nmmax, int *mvals,
		      struct distrib_prop disprop,
		      double *local_S00, double *local_S11, double *local_S22, double *local_S01, double *local_S02, double *local_S12,
		      s2hat_dcomplex *local_wlm_scal_in, s2hat_dcomplex *local_wlm_vect_in, s2hat_dcomplex *local_wlm_tens_in, 
		      s2hat_dcomplex *local_wlm_scal_out, s2hat_dcomplex *local_wlm_vect_out, s2hat_dcomplex *local_wlm_tens_out);
void spin_windowXnoise( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			s2hat_pixparameters pixparam, int pixchoice,
			struct distrib_prop disprop, 
			int llow, int lhigh,
			double *local_noise,
			double *local_win_sig_scal, double *local_win_sig_vect, double *local_win_sig_tens,
			double *local_window_scal_in, double *local_window_vect_in, double *local_window_tens_in,
			double *local_window_scal_out, double *local_window_vect_out, double *local_window_tens_out);
void compute_spin_weight( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			  s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			  s2hat_pixparameters pixparam, int pixchoice,
			  int nlmax, int nmmax, int *mvals,
			  struct distrib_prop disprop,
			  int llow, int lhigh,
			  double *local_Cl, double fwhm,
			  double *weight0, double *weight1, double *weight2);
void spin_windowXinverse( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			  s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			  s2hat_pixparameters pixparam, int pixchoice,
			  int nlmax, int nmmax, int *mvals,
			  struct distrib_prop disprop,
			  double weight0, double weight1, double weight2, 
			  int llow, int lhigh,
			  double *local_noise,
			  double *local_window_scal_in, double *local_window_vect_in, double *local_window_tens_in,
			  double *local_window_scal_out, double *local_window_vect_out, double *local_window_tens_out);
void spin_PCG( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
	       s2hat_pixeltype pixelisation, s2hat_scandef scan, 
	       s2hat_pixparameters pixparam, int pixchoice,
	       int nlmax, int nmmax, int *mvals,
	       double *local_w8ring,
	       struct distrib_prop disprop,
	       double *local_mask,
	       int llow, int lhigh, double fsky,
	       double *local_Cl, double *local_noise, double fwhm,
	       int max_it, double error,
	       double *local_B_scal,
	       double *local_W_scal, double *local_W_vect, double *local_W_tens);


//SCALAR PRODUCT
////////////////

double scalar_product( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
		       s2hat_pixeltype pixelisation, s2hat_scandef scan, 
		       s2hat_pixparameters pixparam, int pixchoice,
		       int nlmax, int nmmax, int *mvals,
		       struct distrib_prop disprop,
		       double *local_V1, double *local_V2);
double spin_product( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
		     s2hat_pixeltype pixelisation, s2hat_scandef scan,
		     s2hat_pixparameters pixparam, int pixchoice,
		     int nlmax, int nmmax, int *mvals,
		     struct distrib_prop disprop,
		     double *local_V1, double *local_V2);
double regularization( int nlmax, double lc, double l);


/////////////
/////////////
//         //
//MAIN CODE//
//         //
/////////////
/////////////

int main( int argc, char* argv[])
{
  int my_rank;
  int my_procnum;
  int root=0;

  /***********/
  /*start MPI*/
  /***********/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&my_procnum);

  /*********************/
  /*read parameter file*/
  /*********************/
  char *pfile;

  int lmax;
  int nbin;
  int nside;
  double fwhm;
  int it_max;
  double error;
  char val[1024];

  char binfile[200];
  char maskfilewindow[200], cmbfile[200], noisefile[200];
  char output_spin0[200], output_spin1[200], output_spin2[200];

  pfile = argv[1];
  
  get_parameter( pfile, "lmax", val);
  lmax = atoi( val);
  get_parameter( pfile, "nbin", val);
  nbin = atoi( val);
 
  get_parameter( pfile, "nside", val);
  nside = atoi( val);
  
  get_parameter( pfile, "fwhm", val);
  fwhm = atoi( val);
 
  get_parameter( pfile, "iteration_max", val);
  it_max = atoi( val);
  get_parameter( pfile, "accuracy", val);
  error = atoi( val) * 0.0000000001;

  get_parameter( pfile, "BinFile", binfile);

  get_parameter( pfile, "maskBinary", maskfilewindow);
  get_parameter( pfile, "cell", cmbfile);
  get_parameter( pfile, "noise", noisefile);

  get_parameter( pfile, "output_spin0", output_spin0);
  get_parameter( pfile, "output_spin1", output_spin1);
  get_parameter( pfile, "output_spin2", output_spin2);

  /******************************/
  /*create pixelization/scanning*/
  /******************************/
  if( my_rank == root ) {printf("-SET PIXELIZATION AND SCANNING\n");}
  int i_pix, length;

  int pixchoice;
  long long int Npixtot;
  long long int Nringsall;

  s2hat_pixparameters pixparam;
  s2hat_pixeltype pixelisation;
  s2hat_scandef scan;
  
  double *mask;
  int *maskscan;
  FILE *f;
  
  pixchoice=PIXCHOICE_HEALPIX;
  pixparam.par1=nside;
  pixparam.par2=0;
  set_pixelization( pixchoice, pixparam, &pixelisation);
  
  Npixtot=pixelisation.npixsall;
  Nringsall=pixelisation.nringsall;
  
  if (my_rank==root)
    {
      mask=(double *)calloc( (int)Npixtot, sizeof(double));      
      read_fits_map( nside, mask, maskfilewindow, 1);            
  
      maskscan=(int *)calloc( (int)Npixtot, sizeof(int));      
      for( i_pix=0; i_pix<(int)Npixtot; i_pix++) {maskscan[i_pix]=(int)(mask[i_pix]);}
      mask2scan( maskscan, pixelisation, &scan);

      free(maskscan);
      /*f=fopen( fmask, "r");
	length=fread( maskscan, sizeof(int), Npixtot, f);
	fclose(f);*/
    }
  
  MPI_scanBcast( pixelisation, &scan, root, my_rank, MPI_COMM_WORLD);

  /****************************************************/
  /*get local parameters for parallelized computations*/
  /****************************************************/
  if( my_rank == root ) {printf("-GET LOCAL CHARACTERISTICS FOR MPI\n");}
  struct distrib_prop disprop;

  int nmvals;
  int first_ring;
  int last_ring;
  int map_size;
  
  long long int nplm;

  int nlmax=lmax;
  int nmmax=lmax;
  int *mvals;

  int lda=nlmax;

  get_local_data_sizes( 0, pixelisation, scan, nlmax, nmmax, 
			my_rank, my_procnum, 
			&nmvals, &first_ring, &last_ring, &map_size, &nplm,
  			root, MPI_COMM_WORLD);

  (disprop).nmvals=nmvals;
  (disprop).firstringvalue=first_ring;
  (disprop).lastringvalue=last_ring;
  (disprop).mapsizevalue=map_size;
  (disprop).nplmvalue=nplm;
  
  mvals=(int *)calloc( nmvals, sizeof(int));
  find_mvalues( my_rank, my_procnum, 
		nmmax, nmvals, 
		mvals);

  /*************************************************/
  /*distribute mask and uncorrelated noise variance*/
  /*         read signal power spectrum            */
  /*************************************************/
  if( my_rank == root ) {printf("-DISTRIBUTE MASK, NOISE MAP AND READ POWER SPECTRUM \n");}
  int i_ring;

  double *noisemap;
  
  double *local_mask;
  double *local_noise;
  double *local_Cl;

  double *local_w8ring;

  double fsky=0.;

  local_Cl=(double *)calloc( 2*lmax+2, sizeof(double));
  f=fopen( cmbfile, "r");
  length=fread( local_Cl, sizeof(double), 2*lmax+2, f);
  fclose(f);

  if (my_rank==root)
  {
	int ell;
	for(ell=0;ell<2*lmax+2;ell++) {printf("ell=%d and cell=%e\n",ell,local_Cl[ell]);}	
  }
 
  local_w8ring=(double *)calloc( 2.0*(disprop.lastringvalue-disprop.firstringvalue+1), sizeof(double));
  for( i_ring=0; i_ring<2.0*(disprop.lastringvalue-disprop.firstringvalue+1); i_ring++) {
    local_w8ring[i_ring]=1.;
  }

  if (my_rank==root)
    {
      for( i_pix=0; i_pix<Npixtot; i_pix++) { 
	fsky += mask[i_pix];
      }
      fsky /= (double)Npixtot;
      printf("Npix obs= %e and fsky=%e\n", fsky*(double)Npixtot, fsky);
      
      noisemap=(double *)calloc( (int)Npixtot, sizeof(double));
      read_fits_map( nside, noisemap, noisefile, 1);
   }
 
  MPI_Bcast( &fsky, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
 
  local_mask=(double *)calloc( disprop.mapsizevalue, sizeof(double));
  distribute_map( pixelisation, 
		  1, 0, 1, 
		  disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		  local_mask, mask, 
		  my_rank, my_procnum, root, MPI_COMM_WORLD);


  local_noise=(double *)calloc( disprop.mapsizevalue, sizeof(double));
  distribute_map( pixelisation, 
		  1, 0, 1, 
		  disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		  local_noise, noisemap, 
		  my_rank, my_procnum, root, MPI_COMM_WORLD);
  
  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++) {
  	if ( local_mask[i_pix] < 0.5 ) {local_noise[i_pix]=1000000.;}
  }
  
  if (my_rank==root)
    {
      free(mask);
      free(noisemap);
    }

  /*******************/
  /*compute bin range*/
  /*******************/
  if( my_rank == root ) {printf("-COMPUTE BIN RANGE\n");}
  int i_bin;
  int *band_ell;
  band_ell=(int *)calloc( nbin+1, sizeof(int));
  if ( my_rank == root)  {
    f = fopen( binfile, "r");
    for ( i_bin=0; i_bin<=nbin; i_bin++) { fscanf(f, "%d", &band_ell[i_bin]);}
    fclose(f);

    printf("          -Lower L value=%d, Higher L value=%d, Total Bin Number=%d\n", band_ell[0], band_ell[nbin], nbin);
  }

  MPI_Bcast( band_ell, nbin+1, MPI_INT, root, MPI_COMM_WORLD);



  /***********/
  /*start PCG*/
  /***********/
  if( my_rank == root ) {printf("-COMPUTE OPTIMAL WINDOW VIA P.C.G\n");}

  //SPIN WEIGHTED APPROACH
  ////////////////////////
  if( my_rank == root ) {printf("          -Compute in the Spin-Weighted approach\n");}
  char filename[200];
  
  double *W_scal;
  double *W_vect;
  double *W_tens;
  
  double *local_B_scal;
   
  double *local_W_scal; 
  double *local_W_vect;
  double *local_W_tens;
  
  if( my_rank==root) {
    W_scal=(double *)calloc( 2*(int)Npixtot, sizeof(double));
    W_vect=(double *)calloc( 2*(int)Npixtot, sizeof(double));
    W_tens=(double *)calloc( 2*(int)Npixtot, sizeof(double));
  }
  
  local_B_scal=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  
  local_W_scal=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  local_W_vect=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  local_W_tens=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
   
  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++) {local_B_scal[i_pix]=local_mask[i_pix];}
  
  for( i_bin=0; i_bin<nbin; i_bin++)
    {
      if( my_rank == root ) {printf("          -Start %d-th bin\n", i_bin+1);}
      if( my_rank == root ) {printf("           Lmin=%d and Lmax=%d\n", band_ell[i_bin], band_ell[i_bin+1]);}
      
      spin_PCG( my_rank, my_procnum, root, MPI_COMM_WORLD,
		pixelisation, scan, 
		pixparam, pixchoice,
		nlmax, nmmax, mvals,
		local_w8ring,
		disprop,
		local_mask,
		band_ell[i_bin], band_ell[i_bin+1], fsky,
		local_Cl, local_noise, fwhm,
		it_max, error,
		local_B_scal,
		local_W_scal, local_W_vect, local_W_tens);
      
      if( my_rank == root ) {printf("          -Collect and write map\n");}
      collect_map( pixelisation, 1, 0, 2, 
		   W_scal, 
		   disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		   local_W_scal, 
		   my_rank, my_procnum, root, MPI_COMM_WORLD);
      collect_map( pixelisation, 1, 0, 2, 
		   W_vect, 
		   disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		   local_W_vect, 
		   my_rank, my_procnum, root, MPI_COMM_WORLD);
      collect_map( pixelisation, 1, 0, 2, 
		   W_tens, 
		   disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		   local_W_tens, 
		   my_rank, my_procnum, root, MPI_COMM_WORLD);
      
 
      if( my_rank==root) {
	/*write in fits file*/
	/********************/
	/*write scalar window*/
 	sprintf( filename,"%s_bin%d.fits", output_spin0, i_bin+1);
        write_fits_map( 1, (int)Npixtot, W_scal, filename);
	
	/*write vector window*/
	sprintf( filename,"%s_bin%d.fits", output_spin1, i_bin+1);
        write_fits_map( 2, (int)Npixtot, W_vect, filename);

        /*write tensor window*/
 	sprintf( filename,"%s_bin%d.fits", output_spin2, i_bin+1);
        write_fits_map( 2, (int)Npixtot, W_tens, filename);

      }
    }
  
  free(band_ell);
  free(local_Cl);
  
  free(local_mask);
  free(local_noise);

  free(local_W_scal);
  free(local_W_vect);
  free(local_W_tens);
  
  free(local_B_scal);
   
  if( my_rank==root) {
    free(W_scal);
    free(W_vect);
    free(W_tens);
  }
  

  /*********/
  /*end MPI*/
  /*********/
  MPI_Finalize();
  return(0);

}

///////////////
///////////////
//           //
//SUBROUTINES//
//           //
///////////////
///////////////

//SPIN WEIGHTED SUBROUTINES
///////////////////////////
void compute_spin_wlm( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
		       s2hat_pixeltype pixelisation, s2hat_scandef scan, 
		       s2hat_pixparameters pixparam, int pixchoice,
		       int nlmax, int nmmax, int *mvals,
		       struct distrib_prop disprop, 
		       double *local_w8ring,
		       double *local_window_scal, double *local_window_vect, double *local_window_tens,
		       s2hat_dcomplex *local_wlm_scal, s2hat_dcomplex *local_wlm_vect, s2hat_dcomplex *local_wlm_tens)
{
  int s;

  /*compute spin-0 wlm*/
  s=0;
  s2hat_map2alm_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, local_w8ring, disprop.mapsizevalue, 
		      local_window_scal, nlmax, local_wlm_scal, 
		      my_procnum, my_rank, world_comm);
  
  /*compute spin-1 wlm*/
  s=1;
  s2hat_map2alm_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, local_w8ring, disprop.mapsizevalue, 
		      local_window_vect, nlmax, local_wlm_vect, 
		      my_procnum, my_rank, world_comm);

  /*compute spin-2 wlm*/
  s=2;
  s2hat_map2alm_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, local_w8ring, disprop.mapsizevalue, 
		      local_window_tens, nlmax, local_wlm_tens, 
		      my_procnum, my_rank, world_comm);
}


void compute_spin_window( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			  s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			  s2hat_pixparameters pixparam, int pixchoice,
			  int nlmax, int nmmax, int *mvals,
			  struct distrib_prop disprop,
			  double *local_mask, 
			  s2hat_dcomplex *local_wlm_scal, s2hat_dcomplex *local_wlm_vect, s2hat_dcomplex *local_wlm_tens,
			  double *local_win_sig_scal, double *local_win_sig_vect, double *local_win_sig_tens)
{
  int i_pix;
  int s;

  int lda=nlmax;

  /*scalar window*/
  s=0;
  s2hat_alm2map_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		      local_win_sig_scal, lda, local_wlm_scal,
		      my_procnum, my_rank, world_comm);

  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++)
    {
      local_win_sig_scal[i_pix] = local_mask[i_pix]*local_win_sig_scal[i_pix];
      local_win_sig_scal[disprop.mapsizevalue+i_pix] = 0.;
      /*local_win_sig_scal[disprop.mapsizevalue+i_pix] = local_mask[i_pix]*local_win_sig_scal[disprop.mapsizevalue+i_pix];*/
    }

  /*vector window*/
  s=1;
  s2hat_alm2map_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		      local_win_sig_vect, lda, local_wlm_vect,
		      my_procnum, my_rank, world_comm);

  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++)
    {
      local_win_sig_vect[i_pix] = local_mask[i_pix]*local_win_sig_vect[i_pix];
      local_win_sig_vect[disprop.mapsizevalue+i_pix] = local_mask[i_pix]*local_win_sig_vect[disprop.mapsizevalue+i_pix];
    }
  
  /*tensor window*/
  s=2;
  s2hat_alm2map_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		      local_win_sig_tens, lda, local_wlm_tens,
		      my_procnum, my_rank, world_comm);

  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++)
    {
      local_win_sig_tens[i_pix] = local_mask[i_pix]*local_win_sig_tens[i_pix];
      local_win_sig_tens[disprop.mapsizevalue+i_pix] = local_mask[i_pix]*local_win_sig_tens[disprop.mapsizevalue+i_pix];
    }

}

void spin_precompute_matrix( int nlmax, int nmmax, int *mvals,
			     struct distrib_prop disprop, 
			     int llow, int lhigh,
			     double *local_Cl, double fwhm,
			     double *local_S00, double *local_S11, double *local_S22, double *local_S01, double *local_S02, double *local_S12)
{
  int l, l1, l2;
  double ll, ll2, ll1;
  
  int limit, limit_inf; 
  int l1i;
  
  int status0, status1, status2;
  double *wigner0, *wigner1, *wigner2;

  double beam;
  double N_l1, N_l2;
  double shape;

  for( l=0; l<=nlmax; l++) {
    ll=(double)l;
  
    local_S00[l] = 0.;
    local_S11[l] = 0.;
    local_S22[l] = 0.;
    local_S01[l] = 0.;
    local_S02[l] = 0.;
    local_S12[l] = 0.;

    local_S00[nlmax+1+l] = 0.;
    local_S11[nlmax+1+l] = 0.;
    local_S22[nlmax+1+l] = 0.;
    local_S01[nlmax+1+l] = 0.;
    local_S02[nlmax+1+l] = 0.;
    local_S12[nlmax+1+l] = 0.;

    for( l2=llow; l2<lhigh; l2++) {
      ll2=(double)l2;

      N_l1 = ll2*(ll2+1.);
      N_l2 = (ll2-1.)*ll2*(ll2+1.)*(ll2+2.);
      shape = ll2*(ll2+1.)/2./M_PI;

      limit = MIN( (l+l2), (int)nlmax);
      limit_inf = MAX( 2, abs(l-l2));

      wigner0 = (double *)calloc( l+l2-limit_inf+1, sizeof(double));
      wigner1 = (double *)calloc( l+l2-limit_inf+1, sizeof(double));
      wigner2 = (double *)calloc( l+l2-limit_inf+1, sizeof(double));
      
      status0 = wig3j( (double)l, (double)l2, 0., -2., wigner0);
      status1 = wig3j( (double)l, (double)l2, -1., -1., wigner1);
      status2 = wig3j( (double)l, (double)l2,  -2., 0., wigner2);
      
      for( l1=limit_inf; l1<=limit; l1++) {
	l1i=l1-limit_inf;
	
	ll1=(double)l1;
	beam = exp(-2.*ll1*(ll1+1.)*fwhm*fwhm*M_PI*M_PI/(60.*60.*180.*180.)/8./0.693/2.);

	if( (l1+l2+l)%2 == 0) {
	  if( l > 1 ) {
	    
	    local_S00[l] += SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*beam;
	    local_S11[l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	    local_S22[l] += SQR(2*wigner2[l1i],2*wigner2[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)/N_l2*beam;
	    
	    local_S01[l] += SQR(2*wigner0[l1i],2*wigner1[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*2.*sqrt(N_l1/N_l2)*beam;
	    local_S02[l] += SQR(2*wigner0[l1i],2*wigner2[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)/sqrt(N_l2)*beam;
	    local_S12[l] += SQR(2*wigner1[l1i],2*wigner2[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*2.*sqrt(N_l1)/N_l2*beam;
	  
	    local_S11[nlmax+1+l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
            local_S22[nlmax+1+l] += SQR(2*wigner2[l1i],2*wigner2[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)/N_l2*beam;
	    local_S12[nlmax+1+l] += SQR(2*wigner1[l1i],2*wigner2[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*2.*sqrt(N_l1)/N_l2*beam;	  
	  }
	  
	  if( l == 1 ) {
	    
	    local_S00[l] += SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*beam;
	    local_S11[l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;

	    local_S01[l] += SQR(2*wigner0[l1i],2*wigner1[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*2.*sqrt(N_l1/N_l2)*beam;

	    local_S11[nlmax+1+l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	  }
	  
	  if( l == 0 ) {
	    
	    local_S00[l] += SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*beam;
	  }
	} else {
	  if( l > 1 ) {
	    
	    local_S00[l] += SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*beam;
	    local_S11[l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	    local_S22[l] += SQR(2*wigner2[l1i],2*wigner2[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)/N_l2*beam;
	    
	    local_S01[l] += SQR(2*wigner0[l1i],2*wigner1[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*2.*sqrt(N_l1/N_l2)*beam;
	    local_S02[l] += SQR(2*wigner0[l1i],2*wigner2[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)/sqrt(N_l2)*beam;
	    local_S12[l] += SQR(2*wigner1[l1i],2*wigner2[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*2.*sqrt(N_l1)/N_l2*beam;

	    local_S11[nlmax+1+l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
            local_S22[nlmax+1+l] += SQR(2*wigner2[l1i],2*wigner2[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)/N_l2*beam;
            local_S12[nlmax+1+l] += SQR(2*wigner1[l1i],2*wigner2[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*2.*sqrt(N_l1)/N_l2*beam;
	  }
	  
	  if( l == 1 ) {
	    
	    local_S00[l] += SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*beam;
	    local_S11[l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;

	    local_S01[l] += SQR(2*wigner0[l1i],2*wigner1[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*2.*sqrt(N_l1/N_l2)*beam;
	  
            local_S11[nlmax+1+l] += SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	  }
	  
	  if( l == 0 ) {
	    
	    local_S00[l] += SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*beam;
	  }
	}
	
      } //end loop l1
      
      free(wigner0);
      free(wigner1);
      free(wigner2);
      
    } //end loop l2
    
    local_S00[l] /= (16.*M_PI);
    local_S11[l] /= (16.*M_PI);
    local_S22[l] /= (16.*M_PI);
    
    local_S01[l] /= (16.*M_PI);
    local_S02[l] /= (16.*M_PI);
    local_S12[l] /= (16.*M_PI);
    
    local_S00[nlmax+1+l] /= (16.*M_PI);
    local_S11[nlmax+1+l] /= (16.*M_PI);
    local_S22[nlmax+1+l] /= (16.*M_PI);

    local_S01[nlmax+1+l] /= (16.*M_PI);
    local_S02[nlmax+1+l] /= (16.*M_PI);
    local_S12[nlmax+1+l] /= (16.*M_PI);

  } //end loop l
  
}

void spin_wlmXsignal( int nlmax, int nmmax, int *mvals,
		      struct distrib_prop disprop,
		      double *local_S00, double *local_S11, double *local_S22, double *local_S01, double *local_S02, double *local_S12,
		      s2hat_dcomplex *local_wlm_scal_in, s2hat_dcomplex *local_wlm_vect_in, s2hat_dcomplex *local_wlm_tens_in, 
		      s2hat_dcomplex *local_wlm_scal_out, s2hat_dcomplex *local_wlm_vect_out, s2hat_dcomplex *local_wlm_tens_out)
{
  int m;
  int l;
 
  for( m=0; m<disprop.nmvals; m++) {
    l=0;
    /*scalar wlm only*/
    (local_wlm_scal_out[m*(nlmax+1)+l]).re = local_S00[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).re;
    (local_wlm_scal_out[m*(nlmax+1)+l]).re = local_S00[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).im;
    
    (local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = 0.0;
    (local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = 0.0;
    
    l=1;
    /*scalar and vector wlm only*/
    (local_wlm_scal_out[m*(nlmax+1)+l]).re = local_S00[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).re+local_S01[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).re;
    (local_wlm_scal_out[m*(nlmax+1)+l]).re = local_S00[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).im+local_S01[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).im;
    
    (local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = 0.0;
    (local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = 0.0;
    
    (local_wlm_vect_out[m*(nlmax+1)+l]).re = local_S01[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).re+local_S11[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).re;
    (local_wlm_vect_out[m*(nlmax+1)+l]).im = local_S01[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).im+local_S11[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).im;
    
    (local_wlm_vect_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = local_S11[nlmax+1+l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re;
    (local_wlm_vect_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = local_S11[nlmax+1+l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im;
    
    for( l=2; l<=nlmax; l++) {
      /*scalar wlm*/
      (local_wlm_scal_out[m*(nlmax+1)+l]).re = local_S00[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).re+local_S01[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).re+local_S02[l]*(local_wlm_tens_in[m*(nlmax+1)+l]).re;
      (local_wlm_scal_out[m*(nlmax+1)+l]).im = local_S00[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).im+local_S01[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).im+local_S02[l]*(local_wlm_tens_in[m*(nlmax+1)+l]).im;
      
      (local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = 0.0;
      (local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = 0.0;
      
      //(local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = local_S00[l]*(local_wlm_scal_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S01[l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S02[l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re;
      //(local_wlm_scal_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im =  local_S00[l]*(local_wlm_scal_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S01[l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S02[l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im;
      
      /*vector wlm*/
      (local_wlm_vect_out[m*(nlmax+1)+l]).re = local_S01[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).re+local_S11[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).re+local_S12[l]*(local_wlm_tens_in[m*(nlmax+1)+l]).re;
      (local_wlm_vect_out[m*(nlmax+1)+l]).im = local_S01[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).im+local_S11[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).im+local_S12[l]*(local_wlm_tens_in[m*(nlmax+1)+l]).im;
      
      (local_wlm_vect_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = local_S11[nlmax+1+l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S12[nlmax+1+l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re;
      (local_wlm_vect_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = local_S11[nlmax+1+l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S12[nlmax+1+l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im;
      
      //(local_wlm_vect_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = local_S01[l]*(local_wlm_scal_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S11[l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S12[l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re;
      //(local_wlm_vect_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im =  local_S01[l]*(local_wlm_scal_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S11[l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S12[l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im;
      
      /*tensor wlm*/
      (local_wlm_tens_out[m*(nlmax+1)+l]).re = local_S02[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).re+local_S12[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).re+local_S22[l]*(local_wlm_tens_in[m*(nlmax+1)+l]).re;
      (local_wlm_tens_out[m*(nlmax+1)+l]).im = local_S02[l]*(local_wlm_scal_in[m*(nlmax+1)+l]).im+local_S12[l]*(local_wlm_vect_in[m*(nlmax+1)+l]).im+local_S22[l]*(local_wlm_tens_in[m*(nlmax+1)+l]).im;
      
      (local_wlm_tens_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = local_S12[nlmax+1+l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S22[nlmax+1+l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re;
      (local_wlm_tens_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = local_S12[nlmax+1+l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S22[nlmax+1+l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im;
      
      //(local_wlm_tens_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = local_S02[l]*(local_wlm_scal_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S12[l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re+local_S22[l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re;
      //(local_wlm_tens_out[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im =  local_S02[l]*(local_wlm_scal_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S12[l]*(local_wlm_vect_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im+local_S22[l]*(local_wlm_tens_in[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im;
      
    } //end loop l    
  } //end loop m
  
}

void spin_windowXnoise( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			s2hat_pixparameters pixparam, int pixchoice,
			struct distrib_prop disprop, 
			int llow, int lhigh,
			double *local_noise,
			double *local_win_sig_scal, double *local_win_sig_vect, double *local_win_sig_tens,
			double *local_window_scal_in, double *local_window_vect_in, double *local_window_tens_in,
			double *local_window_scal_out, double *local_window_vect_out, double *local_window_tens_out)
{
  int i_pix;
  
  int l;
  double ll;

  double factor0, factor1, factor2;

  factor0=0.0;
  factor1=0.0;
  factor2=0.0;
  for( l=llow; l<lhigh; l++)
    {
      ll=(double)l;

      factor0 += ll*(ll+1.);
      factor1 += 4.*ll*(ll+1.)/(ll+2.)/(ll-1.);    
      factor2 += ll*(ll+1.)/(ll-1.)/ll/(ll+1.)/(ll+2.);
    }
  factor0 /= (8.0*M_PI*M_PI);
  factor1 /= (8.0*M_PI*M_PI);
  factor2 /= (8.0*M_PI*M_PI);

  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++)
    {
      local_window_scal_out[i_pix] = factor0*local_noise[i_pix]*local_window_scal_in[i_pix]+local_win_sig_scal[i_pix];
      local_window_vect_out[i_pix] = factor1*local_noise[i_pix]*local_window_vect_in[i_pix]+local_win_sig_vect[i_pix];
      local_window_tens_out[i_pix] = factor2*local_noise[i_pix]*local_window_tens_in[i_pix]+local_win_sig_tens[i_pix];

      local_window_scal_out[disprop.mapsizevalue+i_pix]=  0.;
      //local_window_scal_out[disprop.mapsizevalue+i_pix] = factor0*local_noise[i_pix]*local_window_scal_in[disprop.mapsizevalue+i_pix]+local_win_sig_scal[disprop.mapsizevalue+i_pix];
      local_window_vect_out[disprop.mapsizevalue+i_pix] = factor1*local_noise[i_pix]*local_window_vect_in[disprop.mapsizevalue+i_pix]+local_win_sig_vect[disprop.mapsizevalue+i_pix];
      local_window_tens_out[disprop.mapsizevalue+i_pix] = factor2*local_noise[i_pix]*local_window_tens_in[disprop.mapsizevalue+i_pix]+local_win_sig_tens[disprop.mapsizevalue+i_pix];
    }
}      

void compute_spin_weight( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			  s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			  s2hat_pixparameters pixparam, int pixchoice,
			  int nlmax, int nmmax, int *mvals,
			  struct distrib_prop disprop,
			  int llow, int lhigh,
			  double *local_Cl, double fwhm,
			  double *weight0, double *weight1, double *weight2)
{
  int l, l1, l2;
  double ll, ll2, ll1;
  
  int limit, limit_inf; 
  int l1i;
  
  int status0, status1, status2;
  double *wigner0, *wigner1, *wigner2;

  double beam;
  double N_l1, N_l2;
  double shape;

  *weight0=0.;
  *weight1=0.;
  *weight2=0.;
  for( l=0; l<=nlmax; l++) {
    ll=(double)l;

    for( l2=llow; l2<lhigh; l2++) {
      ll2=(double)l2;

      N_l1 = ll2*(ll2+1.);
      N_l2 = (ll2-1.)*ll2*(ll2+1.)*(ll2+2.);
      shape = ll2*(ll2+1.)/2./M_PI;

      limit = MIN( (l+l2), (int)nlmax);
      limit_inf = MAX( 2, abs(l-l2));

      wigner0 = (double *)calloc( l+l2-limit_inf+1, sizeof(double));
      wigner1 = (double *)calloc( l+l2-limit_inf+1, sizeof(double));
      wigner2 = (double *)calloc( l+l2-limit_inf+1, sizeof(double));
      
      status0 = wig3j( (double)l, (double)l2, 0., -2., wigner0);
      status1 = wig3j( (double)l, (double)l2, -1., -1., wigner1);
      status2 = wig3j( (double)l, (double)l2,  -2., 0., wigner2);
      
      for( l1=limit_inf; l1<=limit; l1++) {
	l1i=l1-limit_inf;
	
	ll1=(double)l1;
	beam = exp(-2.*ll1*(ll1+1.)*fwhm*fwhm*M_PI*M_PI/(60.*60.*180.*180.)/8./0.693/2.);

	if( (l1+l2+l)%2 == 0) {
	  if( l > 1 ) {
	    
	    *weight0 += (2.*ll+1.)*SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*beam;
	    *weight1 += (2.*ll+1.)*SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*(local_Cl[nlmax+1+l1]+local_Cl[l1])*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	    *weight2 += (2.*ll+1.)*SQR(2*wigner2[l1i],2*wigner2[l1i])*shape*(local_Cl[nlmax+1+l1]+local_Cl[l1])*(2.*ll1+1.)/N_l2*beam;
	  }
	  
	  if( l == 1 ) {
	    
	    *weight0 += (2.*ll+1.)*SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*beam;
	    *weight1 += (2.*ll+1.)*SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*(local_Cl[nlmax+1+l1]+local_Cl[l1])*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	  }
	  
	  if( l == 0 ) {
	    
	    *weight0 += (2.*ll+1.)*SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[nlmax+1+l1]*(2.*ll1+1.)*beam;
	  }
	} else {
	  if( l > 1 ) {
	    
	    *weight0 += (2.*ll+1.)*SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*beam;
	    *weight1 += (2.*ll+1.)*SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*(local_Cl[l1]+local_Cl[nlmax+1+l1])*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	    *weight2 += (2.*ll+1.)*SQR(2*wigner2[l1i],2*wigner2[l1i])*shape*(local_Cl[l1]+local_Cl[nlmax+1+l1])*(2.*ll1+1.)/N_l2*beam;
	    	  }
	  
	  if( l == 1 ) {
	    
	    *weight0 += (2.*ll+1.)*SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*beam;
	    *weight1 += (2.*ll+1.)*SQR(2*wigner1[l1i],2*wigner1[l1i])*shape*(local_Cl[l1]+local_Cl[nlmax+1+l1])*(2.*ll1+1.)*4.*N_l1/N_l2*beam;
	  }
	  
	  if( l == 0 ) {
	    
	    *weight0 += (2.*ll+1.)*SQR(2*wigner0[l1i],2*wigner0[l1i])*shape*local_Cl[l1]*(2.*ll1+1.)*beam;
	  }
	}
	
      } //end loop l1
      
      free(wigner0);
      free(wigner1);
      free(wigner2);
      
    } //end loop l2
  } //end loop l
  
  *weight0 /= (16.*M_PI*4.*M_PI);
  *weight1 /= (16.*M_PI*4.*M_PI);
  *weight2 /= (16.*M_PI*4.*M_PI);
  /*if(my_rank==0) {cout<<"weight0="<<*weight0<<", weight1="<<*weight1<<", weight2="<<*weight2<<endl;}*/

}
 
void spin_windowXinverse( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
			  s2hat_pixeltype pixelisation, s2hat_scandef scan, 
			  s2hat_pixparameters pixparam, int pixchoice,
			  int nlmax, int nmmax, int *mvals,
			  struct distrib_prop disprop,
			  double weight0, double weight1, double weight2, 
			  int llow, int lhigh,
			  double *local_noise,
			  double *local_window_scal_in, double *local_window_vect_in, double *local_window_tens_in,
			  double *local_window_scal_out, double *local_window_vect_out, double *local_window_tens_out)
{
  int i_pix;
  
  int l;
  double ll;

  double factor0, factor1, factor2;

  factor0=0.0;
  factor1=0.0;
  factor2=0.0;
  for( l=llow; l<lhigh; l++)
    {
      ll=(double)l;

      factor0 += ll*(ll+1.);
      factor1 += 4.*ll*(ll+1.)/(ll+2.)/(ll-1.);    
      factor2 += ll*(ll+1.)/(ll-1.)/ll/(ll+1.)/(ll+2.);
    }
  factor0 /= (8.0*M_PI*M_PI);
  factor1 /= (8.0*M_PI*M_PI);
  factor2 /= (8.0*M_PI*M_PI);

  for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++)
    {
      local_window_scal_out[i_pix] = local_window_scal_in[i_pix]/(weight0+factor0*local_noise[i_pix]);
      local_window_vect_out[i_pix] = local_window_vect_in[i_pix]/(weight1+factor1*local_noise[i_pix]);
      local_window_tens_out[i_pix] = local_window_tens_in[i_pix]/(weight2+factor2*local_noise[i_pix]);
      
      local_window_scal_out[disprop.mapsizevalue+i_pix] = 0.;
      /*local_window_scal_out[disprop.mapsizevalue+i_pix] = local_window_scal_in[disprop.mapsizevalue+i_pix]/(weight0+factor0*local_noise[i_pix]);*/
      local_window_vect_out[disprop.mapsizevalue+i_pix] = local_window_vect_in[disprop.mapsizevalue+i_pix]/(weight1+factor1*local_noise[i_pix]);
      local_window_tens_out[disprop.mapsizevalue+i_pix] = local_window_tens_in[disprop.mapsizevalue+i_pix]/(weight2+factor2*local_noise[i_pix]);
    }
}
 
 
void spin_PCG( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
	       s2hat_pixeltype pixelisation, s2hat_scandef scan, 
	       s2hat_pixparameters pixparam, int pixchoice,
	       int nlmax, int nmmax, int *mvals,
	       double *local_w8ring,
	       struct distrib_prop disprop,
	       double *local_mask,
	       int llow, int lhigh, double fsky,
	       double *local_Cl, double *local_noise, double fwhm,
	       int max_it, double error,
	       double *local_B_scal,
	       double *local_W_scal, double *local_W_vect, double *local_W_tens)
{
  int i, it;
  int itmax;

  double norm, znorm, znorm1;
  double bknum, bkden, bk;
  double aknum, akden, ak;

  double epsilon;
  double err;

  double *local_S00, *local_S11, *local_S22, *local_S01, *local_S02, *local_S12;

  double *local_p, *local_pp;
  double *local_r, *local_rr;
  double *local_z, *local_zz;
  
  itmax=max_it;
  epsilon=error;

  local_p = (double *)calloc( 6*disprop.mapsizevalue, sizeof(double));
  local_z = (double *)calloc( 6*disprop.mapsizevalue, sizeof(double));
  local_r = (double *)calloc( 6*disprop.mapsizevalue, sizeof(double));

  local_pp = (double *)calloc( 6*disprop.mapsizevalue, sizeof(double));
  local_zz = (double *)calloc( 6*disprop.mapsizevalue, sizeof(double));
  local_rr = (double *)calloc( 6*disprop.mapsizevalue, sizeof(double));

  /*initialize all vectors*/
  for ( i=0; i<6*disprop.mapsizevalue; i++)
    {
      local_p[i]=0.0;
      local_r[i]=0.0;
      local_z[i]=0.0;

      local_pp[i]=0.0;
      local_rr[i]=0.0;
      local_zz[i]=0.0;
    }

  it=0;

  /*preconditionner*/
  double weight0, weight1, weight2;
  compute_spin_weight( my_rank, my_procnum, root, world_comm,
		       pixelisation, scan, 
		       pixparam, pixchoice,
		       nlmax, nmmax, mvals,
		       disprop,
		       llow, lhigh,
		       local_Cl, fwhm,
		       &weight0, &weight1, &weight2);
  spin_windowXinverse( my_rank, my_procnum, root, world_comm,
		       pixelisation, scan, 
		       pixparam, pixchoice,
		       nlmax, nmmax, mvals,
		       disprop,
		       weight0, weight1, weight2,
		       llow, lhigh,
		       local_noise,
		       local_B_scal, local_W_vect, local_W_tens,
		       local_W_scal, local_W_vect, local_W_tens);
  
  /*precompute matrix elements*/
  local_S00=(double *)calloc( 2*(nlmax+1), sizeof(double));
  local_S11=(double *)calloc( 2*(nlmax+1), sizeof(double));
  local_S22=(double *)calloc( 2*(nlmax+1), sizeof(double));
  local_S01=(double *)calloc( 2*(nlmax+1), sizeof(double));
  local_S02=(double *)calloc( 2*(nlmax+1), sizeof(double));
  local_S12=(double *)calloc( 2*(nlmax+1), sizeof(double));
  spin_precompute_matrix( nlmax, nmmax, mvals,
			  disprop, 
			  llow, lhigh,
			  local_Cl, fwhm,
			  local_S00, local_S11, local_S22, local_S01, local_S02, local_S12);
    
  /*compute initial residual and check convergence criterion*/
  s2hat_dcomplex *local_alm0_in;
  s2hat_dcomplex *local_alm1_in;
  s2hat_dcomplex *local_alm2_in;
  s2hat_dcomplex *local_alm0_out;
  s2hat_dcomplex *local_alm1_out;
  s2hat_dcomplex *local_alm2_out;
  local_alm0_in=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  local_alm1_in=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  local_alm2_in=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  local_alm0_out=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  local_alm1_out=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  local_alm2_out=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  double *local_sig_scal;
  double *local_sig_vect;
  double *local_sig_tens;
  local_sig_scal=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  local_sig_vect=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  local_sig_tens=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  compute_spin_wlm( my_rank, my_procnum, root, world_comm,
		    pixelisation, scan, 
		    pixparam, pixchoice,
		    nlmax, nmmax, mvals,
		    disprop, 
		    local_w8ring,
		    local_W_scal, local_W_vect, local_W_tens,
		    local_alm0_in, local_alm1_in, local_alm2_in);
  spin_wlmXsignal( nlmax, nmmax, mvals,
		   disprop, 
		   local_S00, local_S11, local_S22, local_S01, local_S02, local_S12,
		   local_alm0_in, local_alm1_in, local_alm2_in,
		   local_alm0_out, local_alm1_out, local_alm2_out);
  compute_spin_window( my_rank, my_procnum, root, world_comm,
		       pixelisation, scan, 
		       pixparam, pixchoice,
		       nlmax, nmmax, mvals,
		       disprop,
		       local_mask, 
		       local_alm0_out, local_alm1_out, local_alm2_out,
		       local_sig_scal, local_sig_vect, local_sig_tens);
  spin_windowXnoise( my_rank, my_procnum, root, world_comm,
		     pixelisation, scan, 
		     pixparam, pixchoice,
		     disprop,
		     llow, lhigh,
		     local_noise,
		     local_sig_scal, local_sig_vect, local_sig_tens,
		     local_W_scal, local_W_vect, local_W_tens,
		     &local_r[0], &local_r[2*disprop.mapsizevalue], &local_r[4*disprop.mapsizevalue]);
  for( i=0; i<2*disprop.mapsizevalue; i++)
    {
      local_r[i]=local_B_scal[i]-local_r[i];
      local_rr[i]=local_r[i];
 
      local_r[2*disprop.mapsizevalue+i]=-local_r[2*disprop.mapsizevalue+i];
      local_rr[2*disprop.mapsizevalue+i]=local_r[2*disprop.mapsizevalue+i];
      
      local_r[4*disprop.mapsizevalue+i]=-local_r[4*disprop.mapsizevalue+i];
      local_rr[4*disprop.mapsizevalue+i]=local_r[4*disprop.mapsizevalue+i];
    }
  norm = scalar_product( my_rank, my_procnum, root, world_comm,
			 pixelisation, scan, 
			 pixparam, pixchoice,
			 nlmax, nmmax, mvals,
			 disprop,
			 local_B_scal, local_B_scal);
  err=spin_product( my_rank, my_procnum, root, world_comm,
		    pixelisation, scan, 
		    pixparam, pixchoice,
		    nlmax, nmmax, mvals,
		    disprop,
		    local_r, local_r);
  if(my_rank==root) {printf("err=%e and it=-1\n", err);}
  err=sqrt(err/norm);
  if(my_rank==root) {printf("err=%e and it=-1\n", err);}

  /*main loop*/
  spin_windowXinverse( my_rank, my_procnum, root, world_comm,
		       pixelisation, scan, 
		       pixparam, pixchoice,
		       nlmax, nmmax, mvals,
		       disprop,
		       weight0, weight1, weight2,
		       llow, lhigh,
		       local_noise,
		       &local_r[0], &local_r[2*disprop.mapsizevalue], &local_r[4*disprop.mapsizevalue],
		       &local_z[0], &local_z[2*disprop.mapsizevalue], &local_z[4*disprop.mapsizevalue]);
  while ( it<=itmax)
    {
      it++;

      spin_windowXinverse( my_rank, my_procnum, root, world_comm,
			   pixelisation, scan, 
			   pixparam, pixchoice,
			   nlmax, nmmax, mvals,
			   disprop,
			   weight0, weight1, weight2,
			   llow, lhigh,
			   local_noise,
			   &local_rr[0], &local_rr[2*disprop.mapsizevalue], &local_rr[4*disprop.mapsizevalue],
			   &local_zz[0], &local_zz[2*disprop.mapsizevalue], &local_zz[4*disprop.mapsizevalue]);
      /*compute coefficent and directions*/
      bknum=spin_product( my_rank, my_procnum, root, world_comm,
			  pixelisation, scan, 
			  pixparam, pixchoice,
			  nlmax, nmmax, mvals,
			  disprop,
			  local_z, local_rr);
     if ( it==1)
       {
	 for ( i=0; i<6*disprop.mapsizevalue; i++)
	   {
	     local_p[i]=local_z[i];
	     local_pp[i]=local_zz[i];
	   }
	}
      else
	{
	  bk=bknum/bkden;
	  for ( i=0; i<6*disprop.mapsizevalue; i++)
	    {
	      local_p[i]=bk*local_p[i]+local_z[i];
	      local_pp[i]=bk*local_pp[i]+local_zz[i];
	    }
	}
      bkden=bknum;
      
      /*compute alpha-coefficient, new iterate and new residuals*/
      compute_spin_wlm( my_rank, my_procnum, root, world_comm,
			pixelisation, scan, 
			pixparam, pixchoice,
			nlmax, nmmax, mvals,
			disprop, 
			local_w8ring,
			&local_p[0], &local_p[2*disprop.mapsizevalue], &local_p[4*disprop.mapsizevalue],
			local_alm0_in, local_alm1_in, local_alm2_in);
      spin_wlmXsignal( nlmax, nmmax, mvals,
		       disprop, 
		       local_S00, local_S11, local_S22, local_S01, local_S02, local_S12,
		       local_alm0_in, local_alm1_in, local_alm2_in, 
		       local_alm0_out, local_alm1_out, local_alm2_out);
      compute_spin_window( my_rank, my_procnum, root, world_comm,
			   pixelisation, scan, 
			   pixparam, pixchoice,
			   nlmax, nmmax, mvals,
			   disprop,
			   local_mask, 
			   local_alm0_out, local_alm1_out, local_alm2_out,
			   local_sig_scal, local_sig_vect, local_sig_tens);
      spin_windowXnoise( my_rank, my_procnum, root, world_comm,
			 pixelisation, scan, 
			 pixparam, pixchoice,
			 disprop,
			 llow, lhigh,
			 local_noise,
			 local_sig_scal, local_sig_vect, local_sig_tens,
			 &local_p[0], &local_p[2*disprop.mapsizevalue], &local_p[4*disprop.mapsizevalue],
			 &local_z[0], &local_z[2*disprop.mapsizevalue], &local_z[4*disprop.mapsizevalue]);

      akden=spin_product( my_rank, my_procnum, root, world_comm,
			  pixelisation, scan, 
			  pixparam, pixchoice,
			  nlmax, nmmax, mvals,
			  disprop,
			  local_z, local_pp);
      ak=bknum/akden;
      
      /*compute directions and new W*/
      compute_spin_wlm( my_rank, my_procnum, root, world_comm,
			pixelisation, scan, 
			pixparam, pixchoice,
			nlmax, nmmax, mvals,
			disprop, 
			local_w8ring,
			&local_pp[0], &local_pp[2*disprop.mapsizevalue], &local_pp[4*disprop.mapsizevalue],
			local_alm0_in, local_alm1_in, local_alm2_in);
      spin_wlmXsignal( nlmax, nmmax, mvals,
		       disprop,
		       local_S00, local_S11, local_S22, local_S01, local_S02, local_S12,
		       local_alm0_in, local_alm1_in, local_alm2_in, 
		       local_alm0_out, local_alm1_out, local_alm2_out);
      compute_spin_window( my_rank, my_procnum, root, world_comm,
			   pixelisation, scan, 
			   pixparam, pixchoice,
			   nlmax, nmmax, mvals,
			   disprop,
			   local_mask, 
			   local_alm0_out, local_alm1_out, local_alm2_out,
			   local_sig_scal, local_sig_vect, local_sig_tens);
      spin_windowXnoise( my_rank, my_procnum, root, world_comm,
			 pixelisation, scan, 
			 pixparam, pixchoice,
			 disprop,
			 llow, lhigh,
			 local_noise,
			 local_sig_scal, local_sig_vect, local_sig_tens,
			 &local_pp[0], &local_pp[2*disprop.mapsizevalue], &local_pp[4*disprop.mapsizevalue],
			 &local_zz[0], &local_zz[2*disprop.mapsizevalue], &local_zz[4*disprop.mapsizevalue]);
     for ( i=0; i<disprop.mapsizevalue; i++)
	{
	  local_W_scal[i] += ak*local_p[i];
	  local_W_vect[i] += ak*local_p[2*disprop.mapsizevalue+i];
	  local_W_tens[i] += ak*local_p[4*disprop.mapsizevalue+i];

	  local_W_scal[disprop.mapsizevalue+i] += ak*local_p[disprop.mapsizevalue+i];
	  local_W_vect[disprop.mapsizevalue+i] += ak*local_p[3*disprop.mapsizevalue+i];
	  local_W_tens[disprop.mapsizevalue+i] += ak*local_p[5*disprop.mapsizevalue+i];
	}
     for ( i=0; i<disprop.mapsizevalue; i++)
       {
	 local_r[i] -= ak*local_z[i];
	 local_rr[i] -= ak*local_zz[i];

	 local_r[disprop.mapsizevalue+i] -= ak*local_z[disprop.mapsizevalue+i];
	 local_rr[disprop.mapsizevalue+i] -= ak*local_zz[disprop.mapsizevalue+i];

	 local_r[2*disprop.mapsizevalue+i] -= ak*local_z[2*disprop.mapsizevalue+i];
	 local_rr[2*disprop.mapsizevalue+i] -= ak*local_zz[2*disprop.mapsizevalue+i];

	 local_r[3*disprop.mapsizevalue+i] -= ak*local_z[3*disprop.mapsizevalue+i];
	 local_rr[3*disprop.mapsizevalue+i] -= ak*local_zz[3*disprop.mapsizevalue+i];

	 local_r[4*disprop.mapsizevalue+i] -= ak*local_z[4*disprop.mapsizevalue+i];
	 local_rr[4*disprop.mapsizevalue+i] -= ak*local_zz[4*disprop.mapsizevalue+i];

	 local_r[5*disprop.mapsizevalue+i] -= ak*local_z[5*disprop.mapsizevalue+i];
	 local_rr[5*disprop.mapsizevalue+i] -= ak*local_zz[5*disprop.mapsizevalue+i];
	}
      
      /*check stopping criterion*/
      spin_windowXinverse( my_rank, my_procnum, root, world_comm,
			   pixelisation, scan, 
			   pixparam, pixchoice,
			   nlmax, nmmax, mvals,
			   disprop,
			   weight0, weight1, weight2,
			   llow, lhigh,
			   local_noise,
			   &local_r[0], &local_r[2*disprop.mapsizevalue], &local_r[4*disprop.mapsizevalue],
			   &local_z[0], &local_z[2*disprop.mapsizevalue], &local_z[4*disprop.mapsizevalue]);

      err=spin_product( my_rank, my_procnum, root, world_comm,
			pixelisation, scan, 
			pixparam, pixchoice,
			nlmax, nmmax, mvals,
			disprop,
			local_r, local_r);
      err=sqrt(err/norm);
      /*if(my_rank==root) {printf("err=%e and it=%d\n", err, it);}*/
      if (err<=epsilon) break;
    }
	
   if ( my_rank == root ) {printf("err=%e and it=%d\n", err, it);}

  /*normalized window function*/
  znorm=scalar_product( my_rank, my_procnum, root, world_comm,
			pixelisation, scan, 
			pixparam, pixchoice,
			nlmax, nmmax, mvals,
			disprop,
			local_W_scal, local_W_scal);
  for( i=0; i<2*disprop.mapsizevalue; i++)
    {
      local_W_scal[i] *= sqrt(fsky*(double)(pixelisation.npixsall)/znorm);
      local_W_vect[i] *= sqrt(fsky*(double)(pixelisation.npixsall)/znorm);
      local_W_tens[i] *= sqrt(fsky*(double)(pixelisation.npixsall)/znorm);
    }

  free(local_alm0_in);
  free(local_alm1_in);
  free(local_alm2_in);
  free(local_alm0_out);
  free(local_alm1_out);
  free(local_alm2_out);

  free(local_sig_scal);
  free(local_sig_vect);
  free(local_sig_tens);

  free(local_r);
  free(local_p);
  free(local_z);

  free(local_rr);
  free(local_pp);
  free(local_zz);

  free(local_S00);
  free(local_S11);
  free(local_S22);
  free(local_S01);
  free(local_S02);
  free(local_S12);

}

//SCALAR PRODUCT
////////////////
double scalar_product( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
		       s2hat_pixeltype pixelisation, s2hat_scandef scan, 
		       s2hat_pixparameters pixparam, int pixchoice,
		       int nlmax, int nmmax, int *mvals,
		       struct distrib_prop disprop,
		       double *local_V1, double *local_V2)
{
  int i_pix;

  double scalar;
  double local_scalar;

  scalar=0.;
  local_scalar=0.;

  for( i_pix=0; i_pix<2*disprop.mapsizevalue; i_pix++) {
    local_scalar += local_V1[i_pix]*local_V2[i_pix];
  }

  MPI_Allreduce( &local_scalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, world_comm);
  return(scalar);
}
  
double spin_product( int my_rank, int my_procnum, int root, MPI_Comm world_comm,
		     s2hat_pixeltype pixelisation, s2hat_scandef scan,
		     s2hat_pixparameters pixparam, int pixchoice,
		     int nlmax, int nmmax, int *mvals,
		     struct distrib_prop disprop,
		     double *local_V1, double *local_V2)
{
  int i_pix;
  
  double scalar;
  double local_scalar;
  
  scalar=0.;
  local_scalar=0.;

  for( i_pix=0; i_pix<6*disprop.mapsizevalue; i_pix++) {
    local_scalar += local_V1[i_pix]*local_V2[i_pix];
  }
  
  MPI_Allreduce( &local_scalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, world_comm);
  return(scalar);
}
  
double regularization( int nlmax, double lc, double l)
{
  if( (int)l < (int)lc) { 
    return (1);
  } else {
    return (exp( (l-lc)*(l-lc+1.)/2./((double)nlmax-lc)/((double)nlmax-lc)));
  }
}

//WIGNER SUBROUTINES
////////////////////
int wig3j( double l2, double l3, double m2, double m3, double *THRCOF)
{

  double l1min, l1max;
  int ndim, ier=0, l;

  l1min = MAX( fabs(l2-l3), fabs(m2+m3));
  l1max = l2 + l3;
  ndim = (int)(l1max-l1min+1);
  
  if( l2<fabs(m2) || l3<fabs(m3) || (l1max-l1min)<0 ) for( l=0; l<ndim; l++) THRCOF[l] = 0.;
  else wig3j_( &l2, &l3, &m2, &m3, &l1min, &l1max, THRCOF, &ndim, &ier);
  
  if( ier) {
    for( l=0; l<ndim; l++) THRCOF[l] = 0.;
    printf( "err=%d  l2=%d, l3=%d, m2=%d, m3=%d : ", ier, (int)l2, (int)l3, (int)m2, (int)m3);
    switch( ier)
    {
    case 1 : printf( "Either L2.LT.ABS(M2) or L3.LT.ABS(M3)\n"); break;
    case 2 : printf( "Either L2+ABS(M2) or L3+ABS(M3) non-integer\n"); break;
    case 3 : printf( "L1MAX-L1MIN not an integer (l1min=%d, l1max=%d)\n", (int)l1min, (int)l1max); break;
    case 4 : printf( "L1MAX less than L1MIN (l1min=%d, l1max=%d)\n", (int)l1min, (int)l1max); break;
    case 5 : printf( "NDIM less than L1MAX-L1MIN+1 (ndim=%d)\n", (int)ndim); break;
    }
    fflush(stdout);
  }

  return( ier);

}

