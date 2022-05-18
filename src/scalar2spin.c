/*******************************************************
 *                                                     *
 *  COMPUTE VECTOR AND TENSOR WNIDOWS FROM SCALAR ONE  *
 *                                                     *
 *******************************************************/
#include "xpure.h"
#include "chealpix.h"



/////////////////////////////
//                         //
//DECLARATION OF STRUCTURES//
//                         //
/////////////////////////////
struct distrib_prop
{
  int nmvals;
  int firstringvalue;
  int lastringvalue;
  int mapsizevalue;
  long long int nplmvalue;
};



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
 char val[1024];

  char maskfilewindow[1024];
  char scalarfilewindow[1024], invnoisefile[1024];
  char output_spin0[1024], output_spin1[1024], output_spin2[1024];
  int lmax, nside;	
	
  pfile=argv[1];
	
  get_parameter( pfile, "nside", val);
  nside = atoi( val);
  get_parameter( pfile, "lmax", val);
  lmax = atoi( val);

  get_parameter( pfile, "maskBinary", maskfilewindow);
  get_parameter( pfile, "window_spin0", scalarfilewindow);
  get_parameter( pfile, "inverseNoise", invnoisefile);

  get_parameter( pfile, "output_spin0", output_spin0);
  get_parameter( pfile, "output_spin1", output_spin1);
  get_parameter( pfile, "output_spin2", output_spin2);
   
  /******************************/
  /*create pixelization/scanning*/
  /******************************/
  if( my_rank == root ) {printf("-SET PIXELIZATION AND SCANNING\n");}
 
  int i;
  int pixchoice;
  long long int Npixtot;
  long long int Nringsall;

  s2hat_pixparameters pixparam;
  s2hat_pixeltype pixelisation;
  s2hat_scandef scan;
  
  double *invsigma;
 
  int *maskscan;
  double *maskd;
  double *wread, *w_scal;
  
  pixchoice=PIXCHOICE_HEALPIX;
  pixparam.par1=nside;
  pixparam.par2=0;
  set_pixelization( pixchoice, pixparam, &pixelisation);
  
  Npixtot=pixelisation.npixsall;
  Nringsall=pixelisation.nringsall;

  if (my_rank==root)
    {  
     wread=(double *)calloc( (int)Npixtot, sizeof(double));
     maskd=(double *)calloc( (int)Npixtot, sizeof(double));
     invsigma=(double *)calloc( (int)Npixtot, sizeof(double));
   
     read_fits_map( nside, maskd, maskfilewindow, 1);
     read_fits_map( nside, wread, scalarfilewindow, 1);
     read_fits_map( nside, invsigma, invnoisefile, 1);   
     
     maskscan=(int *)calloc( Npixtot, sizeof(int));
     for( i=0; i<Npixtot; i++) {
         maskscan[i]=(int)(maskd[i]);
	 }
     mask2scan( maskscan, pixelisation, &scan);
     free(maskscan);
      
     w_scal=(double *)calloc( 2*Npixtot, sizeof(double));
     for( i=0; i<Npixtot; i++) {
         w_scal[i] = wread[i]*invsigma[i];
	 w_scal[Npixtot+i]=0.0;
	 }
	 
     free(invsigma);
     free(wread);
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
  
  long int nplm;

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

  if( my_rank == root ) {printf("-DISTRIBUTE MASK AND SPIN-0 WINDOW\n");}
  int i_pix, i_ring;
 
  double *local_mask;
  double *local_w_scal;
  double *local_w8ring;
 
  local_w8ring=(double *)calloc( 2.0*(disprop.lastringvalue-disprop.firstringvalue+1), sizeof(double));
  for( i_ring=0; i_ring<2.0*(disprop.lastringvalue-disprop.firstringvalue+1); i_ring++) {
    local_w8ring[i_ring]=1.;
  }

  local_mask=(double *)calloc( disprop.mapsizevalue, sizeof(double));
  distribute_map( pixelisation, 
		  1, 0, 1, 
		  disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		  local_mask, maskd, 
		  my_rank, my_procnum, root, MPI_COMM_WORLD);

  local_w_scal=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  distribute_map( pixelisation, 
		  1, 0, 2, 
		  disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		  local_w_scal, w_scal, 
		  my_rank, my_procnum, root, MPI_COMM_WORLD);
		  
  if( my_rank == root ) {printf("-COMPUTE VECTOR AND TENSOR WINDOWS\n");}
  
  
  int l, m;
  double f_l;
  
  double *w_vect;
  double *w_tens;
  double *local_w_vect;
  double *local_w_tens;

  s2hat_dcomplex *local_wlm;
  
  local_w_vect=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  local_w_tens=(double *)calloc( 2*disprop.mapsizevalue, sizeof(double));
  
  if( my_rank == root ) {printf("     -Map2Alm for the spin-0 Window\n");}
  int s=0;
  local_wlm=(s2hat_dcomplex *)calloc( 2*(nlmax+1)*disprop.nmvals, sizeof(s2hat_dcomplex));
  s2hat_map2alm_spin( pixelisation, scan, s,
		      nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, local_w8ring, disprop.mapsizevalue, 
		      local_w_scal, nlmax, local_wlm,
		      my_procnum, my_rank, MPI_COMM_WORLD);
/*double *feinte;
s2hat_map2alm( 0, pixelisation, scan, 
               nlmax, nmmax, disprop.nmvals, mvals, 1, 1, disprop.firstringvalue,
	       disprop.lastringvalue, local_w8ring, disprop.mapsizevalue, 
	       local_w_scal, nlmax, local_wlm, 0, feinte, 
	       my_procnum, my_rank, MPI_COMM_WORLD);*/ 
  
    if( my_rank == root ) {printf("     -Start spin-1 Window\n");}
    for ( m=0; m<disprop.nmvals; m++){
    l=0;
    f_l=sqrt((double)l*((double)l+1.));
 
    (local_wlm[m*(nlmax+1)+l]).re = 0.0;
    (local_wlm[m*(nlmax+1)+l]).im = 0.0;
 
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = 0.0;
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = 0.0;

    for ( l=1; l<=nlmax; l++) {
    f_l=sqrt((double)l*((double)l+1.));

    (local_wlm[m*(nlmax+1)+l]).re *= f_l;
    (local_wlm[m*(nlmax+1)+l]).im *= f_l;
 
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re *= f_l;
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im *= f_l;
    }
    }

   s=1;
   s2hat_alm2map_spin( pixelisation, scan, s,
		       nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		       local_w_vect, lda, local_wlm,
		       my_procnum, my_rank, MPI_COMM_WORLD);

 
    if( my_rank == root ) {printf("     -Start spin-2 Window\n");}
    for ( m=0; m<disprop.nmvals; m++){
    l=1;
    f_l=sqrt((double)(l-1)*((double)(l+2)));

    (local_wlm[m*(nlmax+1)+l]).re = 0.0;
    (local_wlm[m*(nlmax+1)+l]).im = 0.0;
 
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re = 0.0;
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im = 0.0;

    for ( l=2; l<=nlmax; l++) {
    f_l=sqrt((double)(l-1)*((double)(l+2)));
 
    (local_wlm[m*(nlmax+1)+l]).re *= f_l;
    (local_wlm[m*(nlmax+1)+l]).im *= f_l;
 
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).re *= f_l;
    (local_wlm[(nlmax+1)*disprop.nmvals+m*(nlmax+1)+l]).im *= f_l;
    }
    }

   s=2;
   s2hat_alm2map_spin( pixelisation, scan, s,
		       nlmax, nmmax, disprop.nmvals, mvals, 1, disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		       local_w_tens, lda, local_wlm,
		       my_procnum, my_rank, MPI_COMM_WORLD);

   if( my_rank == root ) {printf("     -Remove ringing in pixels\n");}
   for( i_pix=0; i_pix<disprop.mapsizevalue; i_pix++)
    {
      local_w_vect[i_pix] = (double)(local_mask[i_pix])*local_w_vect[i_pix];
      local_w_vect[disprop.mapsizevalue+i_pix] = (double)(local_mask[i_pix])*local_w_vect[disprop.mapsizevalue+i_pix];
 
      local_w_tens[i_pix] = (double)(local_mask[i_pix])*local_w_tens[i_pix];
      local_w_tens[disprop.mapsizevalue+i_pix] = (double)(local_mask[i_pix])*local_w_tens[disprop.mapsizevalue+i_pix];
   }
  

    if( my_rank == root ) {
      printf("     -Collect and write map\n");
      w_vect=(double *)calloc( 2*Npixtot, sizeof(double));
      w_tens=(double *)calloc( 2*Npixtot, sizeof(double));
      }
      collect_map( pixelisation, 1, 0, 2, 
		   w_vect, 
		   disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		   local_w_vect, 
		   my_rank, my_procnum, root, MPI_COMM_WORLD);
      collect_map( pixelisation, 1, 0, 2, 
		   w_tens, 
		   disprop.firstringvalue, disprop.lastringvalue, disprop.mapsizevalue, 
		   local_w_tens, 
		   my_rank, my_procnum, root, MPI_COMM_WORLD);
      
      if( my_rank==root) {
	/*write in fits file*/
	/********************/
	printf("writing\n");
	write_fits_map( 1, (int)Npixtot, w_scal, output_spin0);
	write_fits_map( 2, (int)Npixtot, w_vect, output_spin1);
	write_fits_map( 2, (int)Npixtot, w_tens, output_spin2);
      }

  
  free(local_wlm);
  free(local_w_scal);
  free(local_w_vect);
  free(local_w_tens);
  free(local_mask);
 
  if( my_rank==root) {
    free(w_scal);
    free(w_vect);
    free(w_tens);
    free(maskd);
  }
  
  /*********/
  /*end MPI*/
  /*********/
  MPI_Finalize();
  return(0);
}

