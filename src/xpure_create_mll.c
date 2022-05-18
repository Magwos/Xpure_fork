//#include "xpol_s2hat.h"
#include "xpure.h"

#define ADD_B_TERMS 1
#define FORCE_ANALYTICAL_DERIVATIVES 0
#define COMPUTE_ONLY_SPIN0 0
void make_mll_std( int lmax, int lmin_i, int lmax_i,
		   double *well_TT, double *well_EE, double *well_BB,
		   double *well_TE, double *well_TB, double *well_EB,
		   double *mll_TT_TT, 
		   double *mll_TE_TE, double *mll_TB_TB,
		   double *mll_EE_EE, double *mll_EE_BB, 
		   double *mll_BB_BB, double *mll_BB_EE, 
		   double *mll_EB_EB);

void make_mll_pure( int lmax, int lmin_i, int lmax_i,
		    double *well_TT, 
		    double *well_EE_spin00, double *well_EE_spin11, double *well_EE_spin22, double *well_EE_spin01, double *well_EE_spin02, double *well_EE_spin12,
		    double *well_BB_spin00, double *well_BB_spin11, double *well_BB_spin22, double *well_BB_spin01, double *well_BB_spin02, double *well_BB_spin12,
		    double *well_TE_spin00, double *well_TE_spin01, double *well_TE_spin02, 
		    double *well_TB_spin00, double *well_TB_spin01, double *well_TB_spin02,
		    double *well_EB_spin00, double *well_EB_spin11, double *well_EB_spin22, double *well_EB_spin01, double *well_EB_spin10, double *well_EB_spin02, double *well_EB_spin20, double *well_EB_spin12, double *well_EB_spin21,
		    double *mll_TT_TT,
		    double *mll_TE_TE, double *mll_TE_TB,
		    double *mll_TB_TB, double *mll_TB_TE,
		    double *mll_EE_EE, double *mll_EE_BB, double *mll_EE_EB,
		    double *mll_BB_BB, double *mll_BB_EE, double *mll_BB_EB,
		    double *mll_EB_EB, double *mll_EB_EE, double *mll_EB_BB);

void make_mll_hybrid( int lmax, int lmin_i, int lmax_i,
		      double *well_TT,
		      double *well_EE_spin00,
		      double *well_BB_spin00, double *well_BB_spin11, double *well_BB_spin22,double *well_BB_spin01, double *well_BB_spin02, double *well_BB_spin12,
		      double *well_TE_spin00,
		      double *well_TB_spin00, double *well_TB_spin01, double *well_TB_spin02,
		      double *well_EB_spin00, double *well_EB_spin01, double *well_EB_spin10, double *well_EB_spin02, double *well_EB_spin20,
		      double *mll_TT_TT,
		      double *mll_TE_TE,
		      double *mll_TB_TB, double *mll_TB_TE,
		      double *mll_EE_EE, double *mll_EE_BB,
		      double *mll_BB_BB, double *mll_BB_EE, double *mll_BB_EB,
		      double *mll_EB_EB, double *mll_EB_EE, double *mll_EB_BB);

void wig3j_( double *L2, double *L3, double *M2, double *M3,
	     double *L1MIN, double *L1MAX, double *THRCOF, int *NDIM, int *IER);
void wig3j_c( int l2, int l3, int m2, int m3, double *wigner);


//=========================================================================
//                          PROGRAMME PRINCIPAL 
//=========================================================================
int main(int argc, char * argv[])
{
  int rank, nprocs;
  int gangroot=0, root=0;

  /* Parameters */
  int i;
  long p;
  int nside, nmasks, lmax;
  int mode;
  int writewl;
  char *pfile;
  char *maskfile_T;
  char *maskfile_E_spin0, *maskfile_E_spin1, *maskfile_E_spin2;
  char *maskfile_B_spin0, *maskfile_B_spin1, *maskfile_B_spin2;
  char *mllfileTT_TT; 
  char *mllfileTE_TE, *mllfileTE_TB;
  char *mllfileTB_TB, *mllfileTB_TE;
  char *mllfileEE_EE, *mllfileEE_BB, *mllfileEE_EB;
  char *mllfileBB_BB, *mllfileBB_EE, *mllfileBB_EB;
  char *mllfileEB_EB, *mllfileEB_EE, *mllfileEB_BB;
  char val[1024], cmd[256];

  long npix;
  
  /* MPI Parameters */
  MPI_Comm gang_comm;
  MPI_Comm root_comm;
  int ngangs, gangsize, gangnmaps, gangnum, gangrank;

  MPI_Init( &argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs);


  //-----------------------------------------------------------------------
  // Read parameters
  //-----------------------------------------------------------------------
  if( argc != 2) {
    printf( "Call is \n");
    printf( "  xpol_create_mll paramfile\n");
    exit(-1);
  }

  pfile = argv[1];
  if( rank == root) printf( "Read parameters...\n");
  fflush( stdout);

  get_parameter( pfile, "nside", val);
  nside = atoi( val);
  get_parameter( pfile, "nmasks", val);
  nmasks = atoi( val);
  get_parameter( pfile, "lmax", val);
  lmax  = atoi( val);
  get_parameter( pfile, "mode", val);
  mode  = atoi( val);

  if( rank == root) {
    if( mode == 0) printf("STANDARD FORMALISM...\n");
    if( mode == 1) printf("PURE FORMALISM...\n");
    if( mode == 2) printf("HYBRID FORMALISM...\n");
  }

  if( lmax > 3*nside-1 || lmax <= 0) {
    lmax = 3*nside-1;
    printf( "WARNING : lmax too high, set at lmax=%i\n", lmax);
  }  
  
  npix=12*(long)nside*(long)nside;
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // MPI initialization
  // divide all the procs into gangs - each gang will work on one set of maps/alms
  //-----------------------------------------------------------------------
  ngangs = nmasks;
  gangsize = nprocs/ngangs;
  if( gangsize*ngangs != nprocs) {
    printf( "Need %i procs for %i gangs of %i procs\n", gangsize*ngangs, ngangs, gangsize);
    exit( 1);
  } else if( rank == root) printf( "%i gangs with %i proc per gang\n", ngangs, gangsize);

  //one mask per gang
  gangnmaps = 1;
  if( gangnmaps*ngangs != nmasks) exit( 1);
  else if( rank == root) printf( "%i maps per gang\n", gangnmaps);

  gangnum  = rank/gangsize;
  gangrank = rank%gangsize;
  MPI_Comm_split( MPI_COMM_WORLD, gangnum, gangrank, &gang_comm);
  MPI_Comm_split( MPI_COMM_WORLD, gangrank, gangnum, &root_comm);
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Read Inputs
  //-----------------------------------------------------------------------
  maskfile_T       = (char *) malloc( 1024*sizeof(char));
  maskfile_E_spin0 = (char *) malloc( 1024*sizeof(char));
  maskfile_E_spin1 = (char *) malloc( 1024*sizeof(char));
  maskfile_E_spin2 = (char *) malloc( 1024*sizeof(char));
  maskfile_B_spin0 = (char *) malloc( 1024*sizeof(char));
  maskfile_B_spin1 = (char *) malloc( 1024*sizeof(char));
  maskfile_B_spin2 = (char *) malloc( 1024*sizeof(char));

  sprintf( cmd, "maskfile%d_T", gangnum+1);
  get_parameter( pfile, cmd, maskfile_T);
  
  sprintf( cmd, "maskfile%d_E_spin0", gangnum+1);
  get_parameter( pfile, cmd, maskfile_E_spin0);
  sprintf( cmd, "maskfile%d_E_spin1", gangnum+1);
  get_parameter( pfile, cmd, maskfile_E_spin1);
  sprintf( cmd, "maskfile%d_E_spin2", gangnum+1);
  get_parameter( pfile, cmd, maskfile_E_spin2);

  sprintf( cmd, "maskfile%d_B_spin0", gangnum+1);
  get_parameter( pfile, cmd, maskfile_B_spin0);
  sprintf( cmd, "maskfile%d_B_spin1", gangnum+1);
  get_parameter( pfile, cmd, maskfile_B_spin1);
  sprintf( cmd, "maskfile%d_B_spin2", gangnum+1);
  get_parameter( pfile, cmd, maskfile_B_spin2);

  mllfileTT_TT = (char *) malloc( 1024*sizeof(char));
  mllfileEE_EE = (char *) malloc( 1024*sizeof(char));
  mllfileEE_BB = (char *) malloc( 1024*sizeof(char));
  mllfileEE_EB = (char *) malloc( 1024*sizeof(char));
  mllfileBB_BB = (char *) malloc( 1024*sizeof(char));
  mllfileBB_EE = (char *) malloc( 1024*sizeof(char));
  mllfileBB_EB = (char *) malloc( 1024*sizeof(char));
  mllfileTE_TE = (char *) malloc( 1024*sizeof(char));
  mllfileTE_TB = (char *) malloc( 1024*sizeof(char));
  mllfileTB_TB = (char *) malloc( 1024*sizeof(char));
  mllfileTB_TE = (char *) malloc( 1024*sizeof(char));
  mllfileEB_BB = (char *) malloc( 1024*sizeof(char));
  mllfileEB_EE = (char *) malloc( 1024*sizeof(char));
  mllfileEB_EB = (char *) malloc( 1024*sizeof(char));


  sprintf( cmd, "mllfile_TT_TT_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTT_TT);

  sprintf( cmd, "mllfile_EE_EE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEE_EE);
  sprintf( cmd, "mllfile_EE_BB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEE_BB);
  sprintf( cmd, "mllfile_EE_EB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEE_EB);

  sprintf( cmd, "mllfile_BB_BB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileBB_BB);
  sprintf( cmd, "mllfile_BB_EE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileBB_EE);
  sprintf( cmd, "mllfile_BB_EB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileBB_EB);
  
  sprintf( cmd, "mllfile_TE_TE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTE_TE);
  sprintf( cmd, "mllfile_TE_TB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTE_TB);

  sprintf( cmd, "mllfile_TB_TB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTB_TB);
  sprintf( cmd, "mllfile_TB_TE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTB_TE);

  sprintf( cmd, "mllfile_EB_EB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEB_EB);
  sprintf( cmd, "mllfile_EB_EE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEB_EE);
  sprintf( cmd, "mllfile_EB_BB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEB_BB);
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // define the S2HAT structures
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( " - Define s2hat structures\n");
  s2hat_pixeltype cpixelization;
  s2hat_scandef cscan;
  s2hat_pixparameters pixpar;

  int nlmax=lmax, nmmax=lmax;
  int lda = nlmax+1;   /* nlmax(+1?)=S2HAT / nstokes=HEALPIX */

  /* define the maps parameters and data distribution */
  int pixchoice = PIXCHOICE_HEALPIX;   /* only supported now */
  pixpar.par1 = nside;
  set_pixelization( pixchoice, pixpar, &cpixelization);   /* define the C pixelization structure */

  /* scan definition */
  if( gangrank == gangroot) printf( " - Scan definition\n");
  double *mask;
  int *intmask;
  double gang_fsky;
  if( gangrank == gangroot) {

    /* alloc */
    mask = (double *) calloc( npix, sizeof(double));

    /* read mask files */
    read_fits_map( nside, mask, maskfile_T, 1);
    
    /* cast int mask */
    intmask = (int *) calloc( npix, sizeof( int));
    for( p=0; p<npix; p++)
      if( mask[p] != 0) {
	intmask[p] = 1;
	gang_fsky++;
      }      

    /* create scan */
    mask2scan( intmask, cpixelization, &cscan);

    /* free */
    free( intmask);
    free(    mask);

    printf( "sky fraction for gang %d : %6.2f %% \n", (int)gangnum, (double)gang_fsky/(double)npix*100.);
    fflush(stdout);
  }
  MPI_scanBcast( cpixelization, &cscan, gangroot, gangrank, gang_comm);
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // get local sizes
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( "%i - Get local sizes\n", gangnum);
  int plms=0, nmvals, first_ring, last_ring, map_size;
  int *mvals;
  long int nplm;
  get_local_data_sizes( plms, cpixelization, cscan, nlmax, nmmax, gangrank, gangsize,
			&nmvals, &first_ring, &last_ring, &map_size, &nplm, gangroot, gang_comm);  

  mvals = (int *) calloc( nmvals, sizeof( int));
  find_mvalues( gangrank, gangsize, nmmax, nmvals, mvals);

/*   printf( "nlmax=%d \t nmmax=%d \t gangrank=%d \t gangsize=%d \t gangroot=%d\n", nlmax, nmmax, gangrank, gangsize, gangroot); */
/*   printf( "nmvals=%d \t first_ring=%d \t last_ring=%d \t map_size=%d \t nplm=%lld\n", nmvals, first_ring, last_ring, map_size, nplm); */
  MPI_Barrier( MPI_COMM_WORLD);
  

  //-----------------------------------------------------------------------
  // distribute masks
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( "%i - Distribute mask\n", gangnum);
  int mapnum=0;
  int ncomp = 2;
  double *local_mask_T;
  double *local_mask_E_spin0, *local_mask_E_spin1, *local_mask_E_spin2;
  double *local_mask_B_spin0, *local_mask_B_spin1, *local_mask_B_spin2;
  local_mask_T  = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_E_spin0 = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_E_spin1 = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_E_spin2 = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_B_spin0 = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_B_spin1 = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_B_spin2 = (double *) calloc( ncomp*map_size, sizeof(double)); 

  switch( mode) {

  case 0:
    //Standard formalism
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_T, first_ring, last_ring, map_size, 
			 local_mask_T, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_E_spin0, first_ring, last_ring, map_size, 
			 local_mask_E_spin0, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_B_spin0, first_ring, last_ring, map_size, 
			 local_mask_B_spin0, gangrank, gangsize, gangroot, gang_comm);
    break;

  case 1:
    //Pure formalism
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_T, first_ring, last_ring, map_size, 
			 local_mask_T, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_E_spin0, first_ring, last_ring, map_size, 
			 local_mask_E_spin0, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, ncomp, maskfile_E_spin1, first_ring, last_ring, map_size, 
			 local_mask_E_spin1, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, ncomp, maskfile_E_spin2, first_ring, last_ring, map_size, 
			 local_mask_E_spin2, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_B_spin0, first_ring, last_ring, map_size, 
			 local_mask_B_spin0, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, ncomp, maskfile_B_spin1, first_ring, last_ring, map_size, 
			 local_mask_B_spin1, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, ncomp, maskfile_B_spin2, first_ring, last_ring, map_size, 
			 local_mask_B_spin2, gangrank, gangsize, gangroot, gang_comm);
    break;

  case 2:
    //Hybrid formalism
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_T, first_ring, last_ring, map_size, 
			 local_mask_T, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_E_spin0, first_ring, last_ring, map_size, 
			 local_mask_E_spin0, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, 1, maskfile_B_spin0, first_ring, last_ring, map_size, 
			 local_mask_B_spin0, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, ncomp, maskfile_B_spin1, first_ring, last_ring, map_size, 
			 local_mask_B_spin1, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, gangnmaps, mapnum, ncomp, maskfile_B_spin2, first_ring, last_ring, map_size, 
			 local_mask_B_spin2, gangrank, gangsize, gangroot, gang_comm);
    break;

  }
  MPI_Barrier( MPI_COMM_WORLD);

  
  /* compute rings */
  if( gangrank == gangroot) printf( "%i - S2HAT : compute rings\n", gangnum);
  long nrings = last_ring-first_ring+1;
  double *local_w8ring;
  local_w8ring = (double *) malloc( nrings*ncomp*sizeof(double));
  for( i=0; i<nrings*ncomp; i++) local_w8ring[i] = 1.;
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // MAP2ALM
  //-----------------------------------------------------------------------
  /* allocate outputs */
  if( gangrank == gangroot) printf( "%i - S2HAT : map2alm\n", gangnum);
  long nwlms = (nlmax+1)*nmvals*ncomp;
  s2hat_dcomplex *local_wlm_T;
  s2hat_dcomplex *local_wlm_Espin0, *local_wlm_Espin1, *local_wlm_Espin2;
  s2hat_dcomplex *local_wlm_Bspin0, *local_wlm_Bspin1, *local_wlm_Bspin2;
  local_wlm_T   = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));
  local_wlm_Espin0 = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));
  local_wlm_Espin1 = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));
  local_wlm_Espin2 = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));
  local_wlm_Bspin0 = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));
  local_wlm_Bspin1 = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));
  local_wlm_Bspin2 = (s2hat_dcomplex *) calloc( nwlms, sizeof( s2hat_dcomplex));

  /* do map2alm spinned transforms */
  s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
		      local_w8ring, map_size, local_mask_T, lda, local_wlm_T, gangsize, gangrank, gang_comm);

  switch( mode) {
    
  case 0:
    /* standard */
    s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_E_spin0, lda, local_wlm_Espin0, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin0, lda, local_wlm_Bspin0, gangsize, gangrank, gang_comm);
    break;
    
  case 1:
    /* pure */
    s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_E_spin0, lda, local_wlm_Espin0, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 1, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_E_spin1, lda, local_wlm_Espin1, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 2, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_E_spin2, lda, local_wlm_Espin2, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin0, lda, local_wlm_Bspin0, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 1, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin1, lda, local_wlm_Bspin1, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 2, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin2, lda, local_wlm_Bspin2, gangsize, gangrank, gang_comm);
    break;
    
  case 2:
    /* hybrid */
    s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_E_spin0, lda, local_wlm_Espin0, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin0, lda, local_wlm_Bspin0, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 1, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin1, lda, local_wlm_Bspin1, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 2, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_mask_B_spin2, lda, local_wlm_Bspin2, gangsize, gangrank, gang_comm);
    break;
    
  }
  
  /* conclude */
  destroy_pixelization( cpixelization);
  destroy_scan( cscan);
  free( local_mask_T   );
  free( local_mask_E_spin0);
  free( local_mask_E_spin1);
  free( local_mask_E_spin2);
  free( local_mask_B_spin0);
  free( local_mask_B_spin1);
  free( local_mask_B_spin2);
  free( local_w8ring);

/*   for( i=0; i<nwlms; i+=10) printf( "wlm[%d] = (%e,%e)\n", i, local_wlm[i].re, local_wlm[i].im); */
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Compute all cl
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( "%i - compute all wells...\n", gangnum);
  int nxspec;
  double *wl_TT;
  double *wl_TEspin00, *wl_TEspin01, *wl_TEspin02;
  double *wl_TBspin00, *wl_TBspin01, *wl_TBspin02;
  double *wl_EEspin00, *wl_EEspin11, *wl_EEspin22, *wl_EEspin01, *wl_EEspin02, *wl_EEspin12;
  double *wl_BBspin00, *wl_BBspin11, *wl_BBspin22, *wl_BBspin01, *wl_BBspin02, *wl_BBspin12;
  double *wl_EBspin00, *wl_EBspin11, *wl_EBspin22, *wl_EBspin01, *wl_EBspin10, *wl_EBspin02, *wl_EBspin20, *wl_EBspin12, *wl_EBspin21;

  //well
  switch( mode) {

  case 0:
    //Start Std. Formalism
    if( gangrank == gangroot) printf("\n Standard formalism...");
    nxspec      = ncomp;
    wl_TT       = (double *) malloc( ncomp*(lmax+1)*sizeof( double));
    wl_TEspin00 = (double *) malloc( ncomp*(lmax+1)*sizeof( double));
    wl_TBspin00 = (double *) malloc( ncomp*(lmax+1)*sizeof( double));
    wl_EEspin00 = (double *) malloc( ncomp*(lmax+1)*sizeof( double));
    wl_BBspin00 = (double *) malloc( ncomp*(lmax+1)*sizeof( double));
    wl_EBspin00 = (double *) malloc( ncomp*(lmax+1)*sizeof( double));

    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_T, local_wlm_T, nxspec, wl_TT, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Espin0, nxspec, wl_TEspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin0, nxspec, wl_TBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Espin0, nxspec, wl_EEspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin0, nxspec, wl_BBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin0, nxspec, wl_EBspin00, gangrank, gangsize, gangroot, gang_comm);
    
    /* Broadcast to all gang procs */
    if( gangrank == gangroot) printf( "\t* share xwl\n");
    MPI_Bcast( wl_TT, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TEspin00, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin00, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin00, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin00, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin00, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    
    free( local_wlm_T );
    free( local_wlm_Espin0);
    free( local_wlm_Espin1);
    free( local_wlm_Espin2);
    free( local_wlm_Bspin0);
    free( local_wlm_Bspin1);
    free( local_wlm_Bspin2);
    free( mvals);
    
    break;

  case 1:
    //Start Pure Formalism                                                                                                                                                                                                  
    if( gangrank == gangroot) printf("\n Pure formalism...");
    nxspec      = ncomp*ncomp;
    wl_TT       = (double *) malloc(  ncomp*(lmax+1)*sizeof( double));  /*TT part*/
    wl_EEspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*EE part*/
    wl_EEspin11 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EEspin22 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EEspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EEspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EEspin12 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*BB part*/
    wl_BBspin11 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin22 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin12 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_TEspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*TE part */
    wl_TEspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_TEspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_TBspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*TB part*/
    wl_TBspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_TBspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double)); /*EB part*/
    wl_EBspin11 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin22 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin10 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin20 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin12 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin21 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));

    /*TT spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_T, local_wlm_T, ncomp, wl_TT, gangrank, gangsize, gangroot, gang_comm);
    /*EE spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_Espin0, local_wlm_Espin0, nxspec, wl_EEspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_Espin1, local_wlm_Espin1, nxspec, wl_EEspin11, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_Espin2, local_wlm_Espin2, nxspec, wl_EEspin22, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_Espin0, local_wlm_Espin1, nxspec, wl_EEspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_Espin0, local_wlm_Espin2, nxspec, wl_EEspin02, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
		 local_wlm_Espin1, local_wlm_Espin2, nxspec, wl_EEspin12, gangrank, gangsize, gangroot, gang_comm);
    /*BB spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin0, nxspec, wl_BBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin1, local_wlm_Bspin1, nxspec, wl_BBspin11, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin2, local_wlm_Bspin2, nxspec, wl_BBspin22, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin1, nxspec, wl_BBspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin2, nxspec, wl_BBspin02, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin1, local_wlm_Bspin2, nxspec, wl_BBspin12, gangrank, gangsize, gangroot, gang_comm);
    /*TE spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Espin0, nxspec, wl_TEspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Espin1, nxspec, wl_TEspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Espin2, nxspec, wl_TEspin02, gangrank, gangsize, gangroot, gang_comm);
    /*TB spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin0, nxspec, wl_TBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin1, nxspec, wl_TBspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin2, nxspec, wl_TBspin02, gangrank, gangsize, gangroot, gang_comm);
    /*EB spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin0, nxspec, wl_EBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin1, local_wlm_Bspin1, nxspec, wl_EBspin11, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin2, local_wlm_Bspin2, nxspec, wl_EBspin22, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin1, nxspec, wl_EBspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin1, local_wlm_Bspin0, nxspec, wl_EBspin10, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin2, nxspec, wl_EBspin02, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin2, local_wlm_Bspin0, nxspec, wl_EBspin20, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin1, local_wlm_Bspin2, nxspec, wl_EBspin12, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin2, local_wlm_Bspin1, nxspec, wl_EBspin21, gangrank, gangsize, gangroot, gang_comm);

    /* Broadcast to all gang procs */
    if( gangrank == gangroot) printf( "\t* share xwl\n");
    MPI_Bcast( wl_TT, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin11, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin22, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin12, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin11, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin22, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin12, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TEspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TEspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TEspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin11, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin22, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin10, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin20, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin12, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin21, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);

    free( local_wlm_T );
    free( local_wlm_Espin0);
    free( local_wlm_Espin1);
    free( local_wlm_Espin2);
    free( local_wlm_Bspin0);
    free( local_wlm_Bspin1);
    free( local_wlm_Bspin2);
    free( mvals);

    break;
    

  case 2:
    //Hybrid formalism
    if( gangrank == gangroot) printf("\n Hybrid formalism...");
    nxspec      = ncomp*ncomp;
    wl_TT       = (double *) malloc( ncomp*(lmax+1)*sizeof( double));   /*TT part*/
    wl_EEspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*EE part*/
    wl_BBspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*BB part*/
    wl_BBspin11 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin22 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_BBspin12 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_TEspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*TE part */
    wl_TBspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));  /*TB part*/
    wl_TBspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_TBspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin00 = (double *) malloc( nxspec*(lmax+1)*sizeof( double)); /*EB part*/
    wl_EBspin01 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin10 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin02 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));
    wl_EBspin20 = (double *) malloc( nxspec*(lmax+1)*sizeof( double));

    /*TT spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_T, ncomp, wl_TT, gangrank, gangsize, gangroot, gang_comm);
    /*EE spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Espin0, nxspec, wl_EEspin00, gangrank, gangsize, gangroot, gang_comm);
    /*BB spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin0, nxspec, wl_BBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin1, local_wlm_Bspin1, nxspec, wl_BBspin11, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin2, local_wlm_Bspin2, nxspec, wl_BBspin22, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin1, nxspec, wl_BBspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin0, local_wlm_Bspin2, nxspec, wl_BBspin02, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Bspin1, local_wlm_Bspin2, nxspec, wl_BBspin12, gangrank, gangsize, gangroot, gang_comm);
    /*TE spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Espin0, nxspec, wl_TEspin00, gangrank, gangsize, gangroot, gang_comm);
    /*TB spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin0, nxspec, wl_TBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin1, nxspec, wl_TBspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_T, local_wlm_Bspin2, nxspec, wl_TBspin02, gangrank, gangsize, gangroot, gang_comm);
    /*EB spectra*/
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin0, nxspec, wl_EBspin00, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin1, nxspec, wl_EBspin01, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin1, local_wlm_Bspin0, nxspec, wl_EBspin10, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin0, local_wlm_Bspin2, nxspec, wl_EBspin02, gangrank, gangsize, gangroot, gang_comm);
    collect_xls( gangnmaps, mapnum, gangnmaps, mapnum, ncomp, nlmax, nmvals, mvals, lda,
                 local_wlm_Espin2, local_wlm_Bspin0, nxspec, wl_EBspin20, gangrank, gangsize, gangroot, gang_comm);
    
    /* Broadcast to all gang procs */
    if( gangrank == gangroot) printf( "\t* share xwl\n");
    MPI_Bcast( wl_TT, ncomp*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EEspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin11, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin22, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_BBspin12, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TEspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_TBspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin00, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin01, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin02, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin10, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    MPI_Bcast( wl_EBspin20, nxspec*(lmax+1), MPI_DOUBLE, gangroot, gang_comm);
    
    free( local_wlm_T );
    free( local_wlm_Espin0);
    free( local_wlm_Espin1);
    free( local_wlm_Espin2);
    free( local_wlm_Bspin0);
    free( local_wlm_Bspin1);
    free( local_wlm_Bspin2);
    free( mvals);

    break;
    
  }
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Compute Mll (one per gang)
  //-----------------------------------------------------------------------
  // spread the ell into the procs of the gang
  int nell = (lmax+1)*(lmax+1);
  int local_nell = (lmax-1)/gangsize;
  int local_lmin = 2+gangrank*local_nell;
  int local_lmax = local_lmin + local_nell-1;
  if( gangrank == gangsize-1) local_lmax=lmax;
  
  double *localmllTT_TT;
  double *localmllTE_TE, *localmllTE_TB;
  double *localmllTB_TB, *localmllTB_TE;
  double *localmllEE_EE, *localmllEE_BB, *localmllEE_EB;
  double *localmllBB_BB, *localmllBB_EE, *localmllBB_EB;
  double *localmllEB_EB, *localmllEB_EE, *localmllEB_BB;

  //compute polarization Mll spread in the gang
  printf( "%d-%d - compute Mll... [%d,%d]\n", gangnum, gangrank, local_lmin, local_lmax);
  localmllTT_TT = (double *) calloc( nell, sizeof( double));
  localmllTE_TE = (double *) calloc( nell, sizeof( double));
  localmllTE_TB = (double *) calloc( nell, sizeof( double));
  localmllTB_TB = (double *) calloc( nell, sizeof( double));
  localmllTB_TE = (double *) calloc( nell, sizeof( double));
  localmllEE_EE = (double *) calloc( nell, sizeof( double));
  localmllEE_BB = (double *) calloc( nell, sizeof( double));
  localmllEE_EB = (double *) calloc( nell, sizeof( double));
  localmllBB_BB = (double *) calloc( nell, sizeof( double));
  localmllBB_EE = (double *) calloc( nell, sizeof( double));
  localmllBB_EB = (double *) calloc( nell, sizeof( double));
  localmllEB_EB = (double *) calloc( nell, sizeof( double));
  localmllEB_EE = (double *) calloc( nell, sizeof( double));
  localmllEB_BB = (double *) calloc( nell, sizeof( double));
  MPI_Barrier( MPI_COMM_WORLD);

  //Compute Mll
  if( mode == 0) {
    if( gangrank == gangroot) printf( "Compute all kernels in the STANDARD formalism !\n");
    make_mll_std( lmax, local_lmin, local_lmax,
		  wl_TT, wl_EEspin00, wl_BBspin00, wl_TEspin00, wl_TBspin00, wl_EBspin00,
		  localmllTT_TT, 
		  localmllTE_TE, localmllTB_TB,
		  localmllEE_EE, localmllEE_BB, 
		  localmllBB_BB, localmllBB_EE, 
		  localmllEB_EB);
    free( wl_TT);
    free( wl_TEspin00);
    free( wl_TBspin00);
    free( wl_EEspin00);
    free( wl_BBspin00);
    free( wl_EBspin00);
    MPI_Barrier( MPI_COMM_WORLD);

    if( gangrank == gangroot) printf("\t Collect and write Mll\n");
    collect_write_mll( nlmax, mllfileTT_TT, localmllTT_TT, gangrank, gangroot, gang_comm);
    free( localmllTT_TT);
    collect_write_mll( nlmax, mllfileTE_TE, localmllTE_TE, gangrank, gangroot, gang_comm);
    free( localmllTE_TE);
    collect_write_mll( nlmax, mllfileTB_TB, localmllTB_TB, gangrank, gangroot, gang_comm);
    free( localmllTB_TB);
    collect_write_mll( nlmax, mllfileEE_EE, localmllEE_EE, gangrank, gangroot, gang_comm);
    free( localmllEE_EE);
    collect_write_mll( nlmax, mllfileEE_BB, localmllEE_BB, gangrank, gangroot, gang_comm);
    free( localmllEE_BB);
    collect_write_mll( nlmax, mllfileBB_EE, localmllBB_EE, gangrank, gangroot, gang_comm);
    free( localmllBB_EE);
    collect_write_mll( nlmax, mllfileBB_BB, localmllBB_BB, gangrank, gangroot, gang_comm);
    free( localmllBB_BB);
    collect_write_mll( nlmax, mllfileEB_EB, localmllEB_EB, gangrank, gangroot, gang_comm);
    free( localmllEB_EB);

  }
  
  if( mode == 1) {
    if( gangrank == gangroot) printf( "Compute all kernels in the PURE formalism !\n");
    make_mll_pure( lmax, local_lmin, local_lmax,
		   wl_TT, 
		   wl_EEspin00, wl_EEspin11, wl_EEspin22, wl_EEspin01, wl_EEspin02, wl_EEspin12,
		   wl_BBspin00, wl_BBspin11, wl_BBspin22, wl_BBspin01, wl_BBspin02, wl_BBspin12,
		   wl_TEspin00, wl_TEspin01, wl_TEspin02, 
		   wl_TBspin00, wl_TBspin01, wl_TBspin02,
		   wl_EBspin00, wl_EBspin11, wl_EBspin22, wl_EBspin01, wl_EBspin10, wl_EBspin02, wl_EBspin20, wl_EBspin12, wl_EBspin21,
		   localmllTT_TT,
		   localmllTE_TE, localmllTE_TB,
		   localmllTB_TB, localmllTB_TE,
		   localmllEE_EE, localmllEE_BB, localmllEE_EB,
		   localmllBB_BB, localmllBB_EE, localmllBB_EB,
		   localmllEB_EB, localmllEB_EE, localmllEB_BB);
    free( wl_TT);
    free( wl_TEspin00); free( wl_TEspin01); free( wl_TEspin02);
    free( wl_TBspin00); free( wl_TBspin01); free( wl_TBspin02);
    free( wl_EEspin00); free( wl_EEspin11); free( wl_EEspin22); free( wl_EEspin01); free( wl_EEspin02); free( wl_EEspin12);
    free( wl_BBspin00); free( wl_BBspin11); free( wl_BBspin22); free( wl_BBspin01); free( wl_BBspin02); free( wl_BBspin12);
    free( wl_EBspin00); free( wl_EBspin11); free( wl_EBspin22); free( wl_EBspin01); free( wl_EBspin02); free( wl_EBspin12);
    free( wl_EBspin10); free( wl_EBspin20); free( wl_EBspin21);
    MPI_Barrier( MPI_COMM_WORLD);


    if( gangrank == gangroot) printf("\t Collect and write Mll\n");
    collect_write_mll( nlmax, mllfileTT_TT, localmllTT_TT, gangrank, gangroot, gang_comm);
    free( localmllTT_TT);

    collect_write_mll( nlmax, mllfileTE_TE, localmllTE_TE, gangrank, gangroot, gang_comm);
    free( localmllTE_TE);
    collect_write_mll( nlmax, mllfileTE_TB, localmllTE_TB, gangrank, gangroot, gang_comm);
    free( localmllTE_TB);
    collect_write_mll( nlmax, mllfileTB_TB, localmllTB_TB, gangrank, gangroot, gang_comm);
    free( localmllTB_TB);
    collect_write_mll( nlmax, mllfileTB_TE, localmllTB_TE, gangrank, gangroot, gang_comm);
    free( localmllTB_TE);

    collect_write_mll( nlmax, mllfileEE_EE, localmllEE_EE, gangrank, gangroot, gang_comm);
    free( localmllEE_EE);
    collect_write_mll( nlmax, mllfileEE_BB, localmllEE_BB, gangrank, gangroot, gang_comm);
    free( localmllEE_BB);
    collect_write_mll( nlmax, mllfileEE_EB, localmllEE_EB, gangrank, gangroot, gang_comm);
    free( localmllEE_EB);

    collect_write_mll( nlmax, mllfileBB_EE, localmllBB_EE, gangrank, gangroot, gang_comm);
    free( localmllBB_EE);
    collect_write_mll( nlmax, mllfileBB_BB, localmllBB_BB, gangrank, gangroot, gang_comm);
    free( localmllBB_BB);
    collect_write_mll( nlmax, mllfileBB_EB, localmllBB_EB, gangrank, gangroot, gang_comm);
    free( localmllBB_EB);

    collect_write_mll( nlmax, mllfileEB_EB, localmllEB_EB, gangrank, gangroot, gang_comm);
    free( localmllEB_EB);
    collect_write_mll( nlmax, mllfileEB_EE, localmllEB_EE, gangrank, gangroot, gang_comm);
    free( localmllEB_EE);
    collect_write_mll( nlmax, mllfileEB_BB, localmllEB_BB, gangrank, gangroot, gang_comm);
    free( localmllEB_BB);

  }
  
  if( mode == 2) {
    if( gangrank == gangroot) printf( "Compute all kernels in the HYBRID formalism !\n");
    make_mll_hybrid( lmax, local_lmin, local_lmax,
		     wl_TT,
		     wl_EEspin00,
		     wl_BBspin00, wl_BBspin11, wl_BBspin22, wl_BBspin01, wl_BBspin02, wl_BBspin12,
		     wl_TEspin00,
		     wl_TBspin00, wl_TBspin01, wl_TBspin02,
		     wl_EBspin00, wl_EBspin01, wl_EBspin10, wl_EBspin02, wl_EBspin20,
		     localmllTT_TT,
		     localmllTE_TE,
		     localmllTB_TB, localmllTB_TE,
		     localmllEE_EE, localmllEE_BB,
		     localmllBB_BB, localmllBB_EE, localmllBB_EB,
		     localmllEB_EB, localmllEB_EE, localmllEB_BB);
    free( wl_TT);
    free( wl_TEspin00);
    free( wl_TBspin00);
    free( wl_TBspin01);
    free( wl_TBspin02);
    free( wl_EEspin00);
    free( wl_BBspin00);
    free( wl_BBspin11);
    free( wl_BBspin22);
    free( wl_BBspin01);
    free( wl_BBspin02);
    free( wl_BBspin12);
    free( wl_EBspin00);
    free( wl_EBspin01);
    free( wl_EBspin02);
    free( wl_EBspin10);
    free( wl_EBspin20);
    MPI_Barrier( MPI_COMM_WORLD);

    if( gangrank == gangroot) printf("\t Collect and write Mll\n");

    collect_write_mll( nlmax, mllfileTT_TT, localmllTT_TT, gangrank, gangroot, gang_comm);
    free( localmllTT_TT);

    collect_write_mll( nlmax, mllfileTE_TE, localmllTE_TE, gangrank, gangroot, gang_comm);
    free( localmllTE_TE);
    collect_write_mll( nlmax, mllfileTB_TB, localmllTB_TB, gangrank, gangroot, gang_comm);
    free( localmllTB_TB);
    collect_write_mll( nlmax, mllfileTB_TE, localmllTB_TE, gangrank, gangroot, gang_comm);
    free( localmllTB_TE);

    collect_write_mll( nlmax, mllfileEE_EE, localmllEE_EE, gangrank, gangroot, gang_comm);
    free( localmllEE_EE);
    collect_write_mll( nlmax, mllfileEE_BB, localmllEE_BB, gangrank, gangroot, gang_comm);
    free( localmllEE_BB);

    collect_write_mll( nlmax, mllfileBB_EE, localmllBB_EE, gangrank, gangroot, gang_comm);
    free( localmllBB_EE);
    collect_write_mll( nlmax, mllfileBB_BB, localmllBB_BB, gangrank, gangroot, gang_comm);
    free( localmllBB_BB);
    collect_write_mll( nlmax, mllfileBB_EB, localmllBB_EB, gangrank, gangroot, gang_comm);
    free( localmllBB_EB);

    collect_write_mll( nlmax, mllfileEB_EB, localmllEB_EB, gangrank, gangroot, gang_comm);
    free( localmllEB_EB);
    collect_write_mll( nlmax, mllfileEB_EE, localmllEB_EE, gangrank, gangroot, gang_comm);
    free( localmllEB_EE);
    collect_write_mll( nlmax, mllfileEB_BB, localmllEB_BB, gangrank, gangroot, gang_comm);
    free( localmllEB_BB);

  }
  MPI_Barrier( MPI_COMM_WORLD);
  

  //-----------------------------------------------------------------------
  // Free
  //-----------------------------------------------------------------------
  free( maskfile_T );
  free( maskfile_E_spin0);
  free( maskfile_E_spin1);
  free( maskfile_E_spin2);
  free( maskfile_B_spin0);
  free( maskfile_B_spin1);
  free( maskfile_B_spin2);
  free( mllfileTT_TT);
  free( mllfileTE_TE);
  free( mllfileTE_TB);
  free( mllfileTB_TB);
  free( mllfileTB_TE);
  free( mllfileEE_EE);
  free( mllfileEE_BB);
  free( mllfileEE_EB);
  free( mllfileBB_BB);
  free( mllfileBB_EE);
  free( mllfileBB_EB);
  free( mllfileEB_EB);
  free( mllfileEB_EE);
  free( mllfileEB_BB);

  MPI_Finalize();
  return( 0);

}


void make_mll_std( int lmax, int lmin_i, int lmax_i,
		   double *well_TT, double *well_EE, double *well_BB,
		   double *well_TE, double *well_TB, double *well_EB,
		   double *mll_TT_TT, 
		   double *mll_TE_TE, double *mll_TB_TB,
		   double *mll_EE_EE, double *mll_EE_BB, 
		   double *mll_BB_BB, double *mll_BB_EE, 
		   double *mll_EB_EB)
{
  double sum_TT_TT;
  double sum_TE_TE, sum_TE_TB;
  double sum_TB_TB, sum_TB_TE;
  double sum_EE_EE, sum_EE_BB, sum_EE_EB;
  double sum_BB_BB, sum_BB_EE, sum_BB_EB;
  double sum_EB_EB, sum_EB_EE, sum_EB_BB;
  double factor1, factor2;
  double *wigner, *wigner0;
  int l1, l2, l3, limit;

  //alloc size max = l1+l2+1s
  wigner  = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner0 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));

  //create mll
  for( l1=lmin_i; l1<=lmax_i; l1++) {
    for( l2=2; l2<=lmax; l2++) {
      sum_TT_TT = 0.;
      sum_TE_TE = 0.;
      sum_TB_TB = 0.;
      sum_EE_EE = 0.;
      sum_EE_BB = 0.;
      sum_BB_BB = 0.;
      sum_BB_EE = 0.;
      sum_EB_EB = 0.;

      wig3j_c( l1, l2,  0, 0, wigner );
      wig3j_c( l1, l2, -2, 2, wigner0);

      limit = MIN( (l1+l2), (int)lmax);
      for( l3=abs(l1-l2); l3<=limit; l3++) {

	if( (l1+l2+l3)%2 == 0) {
	  
	  sum_TT_TT += (2.*(double)l3+1.) * ( (   wigner[l3]*wigner[l3] ) * well_TT[l3]);       /* TT calculation */
	  sum_TE_TE += (2.*(double)l3+1.) * ( (2.*wigner[l3]*wigner0[l3]) * well_TE[l3]);	/* TE calculation */
	  sum_TB_TB += (2.*(double)l3+1.) * ( (2.*wigner[l3]*wigner0[l3]) * well_TB[l3]);	/* TB calculation */
	  sum_EE_EE += (2.*(double)l3+1.) * ( (4.*wigner0[l3]*wigner0[l3]) * well_EE[l3]);	/* EE calculation */
          sum_BB_BB += (2.*(double)l3+1.) * ( (4.*wigner0[l3]*wigner0[l3]) * well_BB[l3]);	/* BB calculation */
	  sum_EB_EB += (2.*(double)l3+1.) * ( (4.*wigner0[l3]*wigner0[l3]) * well_EB[l3]);	/* EB calculation */

	} else {

          sum_EE_BB += (2.*(double)l3+1.) * ( (4.*wigner0[l3]*wigner0[l3]) * well_EE[l3]);      /* Start EE calculation */
          sum_BB_EE += (2.*(double)l3+1.) * ( (4.*wigner0[l3]*wigner0[l3]) * well_BB[l3]);      /* Start BB calculation */
	  sum_EB_EB -= (2.*(double)l3+1.) * ( (4.*wigner0[l3]*wigner0[l3]) * well_EB[l3]);      /* Start EB calculation */

	}

      } //end loop on l3
      
      mll_TT_TT[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (4.*M_PI) * sum_TT_TT;

      mll_TE_TE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TE_TE;
      mll_TB_TB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TB_TB;

      mll_EE_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EE_EE;
      mll_EE_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EE_BB;
      mll_BB_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_BB_EE;
      mll_BB_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_BB_BB;
      mll_EB_EB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_EB;

    } //end loop l2
  } //end loop l1

  free( wigner );
  free( wigner0);
}


void make_mll_pure( int lmax, int lmin_i, int lmax_i,
                    double *well_TT, 
                    double *well_EE_spin00, double *well_EE_spin11, double *well_EE_spin22, double *well_EE_spin01, double *well_EE_spin02, double *well_EE_spin12,
                    double *well_BB_spin00, double *well_BB_spin11, double *well_BB_spin22, double *well_BB_spin01, double *well_BB_spin02, double *well_BB_spin12,
                    double *well_TE_spin00, double *well_TE_spin01, double *well_TE_spin02, 
                    double *well_TB_spin00, double *well_TB_spin01, double *well_TB_spin02,
                    double *well_EB_spin00, double *well_EB_spin11, double *well_EB_spin22, double *well_EB_spin01, double *well_EB_spin10, double *well_EB_spin02, double *well_EB_spin20, double *well_EB_spin12, double *well_EB_spin21,
                    double *mll_TT_TT,
                    double *mll_TE_TE, double *mll_TE_TB,
                    double *mll_TB_TB, double *mll_TB_TE,
                    double *mll_EE_EE, double *mll_EE_BB, double *mll_EE_EB,
                    double *mll_BB_BB, double *mll_BB_EE, double *mll_BB_EB,
                    double *mll_EB_EB, double *mll_EB_EE, double *mll_EB_BB)
{
  double sum_TT_TT;
  double sum_TE_TE, sum_TE_TB;
  double sum_TB_TB, sum_TB_TE;
  double sum_EE_EE, sum_EE_BB, sum_EE_EB;
  double sum_BB_BB, sum_BB_EE, sum_BB_EB;
  double sum_EB_EB, sum_EB_EE, sum_EB_BB;
  double factor1, factor2;
  double *wigner, *wigner0, *wigner1, *wigner2;
  int l1, l2, l3, limit;

  //alloc size max = l1+l2+1s
  wigner  = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner0 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner1 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner2 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));

  //create mll
  for( l1=lmin_i; l1<=lmax_i; l1++) {
    for( l2=2; l2<=lmax; l2++) {
      sum_TT_TT = 0.;
      sum_TE_TE = 0.;
      sum_TE_TB = 0.;
      sum_TB_TB = 0.;
      sum_TB_TE = 0.;
      sum_EE_EE = 0.;
      sum_EE_BB = 0.;
      sum_EE_EB = 0.;
      sum_BB_BB = 0.;
      sum_BB_EE = 0.;
      sum_BB_EB = 0.;
      sum_EB_BB = 0.;
      sum_EB_EE = 0.;
      sum_EB_EB = 0.;

      wig3j_c( l1, l2,  0, 0, wigner );
      wig3j_c( l1, l2, -2, 2, wigner0);
      wig3j_c( l1, l2, -1, 2, wigner1);
      wig3j_c( l1, l2,  0, 2, wigner2);

      /* (l+1)!/(l-1)! */
      if( l1 > 0) factor1 = (double)l1*(double)(l1+1);
      else factor1 = 0.;

      /* (l-2)!/(l+2)! */
      if( l1 > 1) factor2 = 1. / (double)(l1+2) / (double)(l1+1) / (double)l1 / (double)(l1-1);
      else factor2 = 0.;

      limit = MIN( (l1+l2), (int)lmax);
      for( l3=abs(l1-l2); l3<=limit; l3++) {

	if( (l1+l2+l3)%2 == 0) {
	  /* TT calculation */
	  sum_TT_TT += (2.*(double)l3+1.) * ( wigner[l3]*wigner[l3] ) * well_TT[l3];

	  /* TE calculation */
	  sum_TE_TE += (2.*(double)l3+1.) * (    (2.*wigner[l3]*wigner0[l3]) * well_TE_spin00[l3] +
						 (2.*wigner[l3]*wigner1[l3]) * well_TE_spin01[l3] * 2.*sqrt(factor1*factor2) +
						 (2.*wigner[l3]*wigner2[l3]) * well_TE_spin02[l3] * sqrt(factor2)
					    );
	  sum_TE_TB += (2.*(double)l3+1.) * (    (2.*wigner[l3]*wigner0[l3]) * well_TE_spin00[2*(lmax+1)+l3] +
                                                 (2.*wigner[l3]*wigner1[l3]) * well_TE_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (2.*wigner[l3]*wigner2[l3]) * well_TE_spin02[2*(lmax+1)+l3] * sqrt(factor2)
					    );

	  /* TB calculation */
	  sum_TB_TB += (2.*(double)l3+1.) * (    (2.*wigner[l3]*wigner0[l3]) * well_TB_spin00[l3] +
                                                 (2.*wigner[l3]*wigner1[l3]) * well_TB_spin01[l3] * 2.*sqrt(factor1*factor2) +
                                                 (2.*wigner[l3]*wigner2[l3]) * well_TB_spin02[l3] * sqrt(factor2)
						 );
          sum_TB_TE += (2.*(double)l3+1.) * (    (2.*wigner[l3]*wigner0[l3]) * well_TB_spin00[2*(lmax+1)+l3] +
                                                 (2.*wigner[l3]*wigner1[l3]) * well_TB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (2.*wigner[l3]*wigner2[l3]) * well_TB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );


	  /* EE calculation */
	  sum_EE_EE += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_EE_spin00[l3] +
					         (4.*wigner1[l3]*wigner1[l3]) * well_EE_spin11[l3] * 4.*factor1*factor2 +
						 (4.*wigner2[l3]*wigner2[l3]) * well_EE_spin22[l3] * factor2 +
					      2.*(4.*wigner0[l3]*wigner1[l3]) * well_EE_spin01[l3] * 2.*sqrt(factor1*factor2) +
					      2.*(4.*wigner0[l3]*wigner2[l3]) * well_EE_spin02[l3] * sqrt(factor2) +
					      2.*(4.*wigner1[l3]*wigner2[l3]) * well_EE_spin12[l3] * 2.*sqrt(factor1)*factor2
                                            );
	  sum_EE_BB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EE_spin11[(lmax+1)+l3] * 4.*factor1*factor2 +
						 (4.*wigner2[l3]*wigner2[l3]) * well_EE_spin22[(lmax+1)+l3] * factor2 +
					      2.*(4.*wigner1[l3]*wigner2[l3]) * well_EE_spin12[(lmax+1)+l3] * 2.*sqrt(factor1)*factor2
					    );
	  sum_EE_EB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EE_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
						 (4.*wigner2[l3]*wigner2[l3]) * well_EE_spin22[2*(lmax+1)+l3] * factor2 +
						 (4.*wigner1[l3]*wigner2[l3]) * (well_EE_spin12[2*(lmax+1)+l3] + well_EE_spin12[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
						 (4.*wigner0[l3]*wigner1[l3]) * well_EE_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
						 (4.*wigner0[l3]*wigner2[l3]) * well_EE_spin02[2*(lmax+1)+l3] * sqrt(factor2)
					    );
	  
	  /* BB calculation */
          sum_BB_BB += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_BB_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[l3] * factor2 +
                                              2.*(4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[l3] * 2.*sqrt(factor1*factor2) +
                                              2.*(4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[l3] * sqrt(factor2) +
                                              2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[l3] * 2.*sqrt(factor1)*factor2
                                            );
          sum_BB_EE += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[(lmax+1)+l3] * factor2 +
					      2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[(lmax+1)+l3] * 2.*sqrt(factor1)*factor2
					    );
          sum_BB_EB -= (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_BB_spin12[2*(lmax+1)+l3] + well_BB_spin12[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
					    );

	  /* EB calculation */
	  sum_EB_EB +=  (2.*(double)l3+1.) * (   (4.*wigner0[l3]*wigner0[l3]) * well_EB_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[l3] * factor2 +
						 (4.*wigner0[l3]*wigner1[l3]) * (well_EB_spin01[l3] + well_EB_spin10[l3]) * 2.*sqrt(factor1*factor2) +
						 (4.*wigner0[l3]*wigner2[l3]) * (well_EB_spin02[l3] + well_EB_spin20[l3]) * sqrt(factor2) +
						 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[l3] + well_EB_spin21[l3]) * 2.*sqrt(factor1)*factor2 -
						 (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[(lmax+1)+l3] * 4.*factor1*factor2 -
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[(lmax+1)+l3] * factor2 -
						 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[(lmax+1)+l3] + well_EB_spin21[(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2
						 );
	  sum_EB_EE -= (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[2*(lmax+1)+l3] + well_EB_spin21[2*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );
	  sum_EB_BB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[3*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[3*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[3*(lmax+1)+l3] + well_EB_spin21[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[3*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[3*(lmax+1)+l3] * sqrt(factor2)
                                                 );

	} else {
          /* EE calculation */
          sum_EE_BB += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_EE_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_EE_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EE_spin22[l3] * factor2 +
						 2.*(4.*wigner0[l3]*wigner1[l3]) * well_EE_spin01[l3] * 2.*sqrt(factor1*factor2) +
						 2.*(4.*wigner0[l3]*wigner2[l3]) * well_EE_spin02[l3] * sqrt(factor2) +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_EE_spin12[l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_EE_EE += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EE_spin11[(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EE_spin22[(lmax+1)+l3] * factor2 +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_EE_spin12[(lmax+1)+l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_EE_EB -= (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EE_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EE_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EE_spin12[2*(lmax+1)+l3] + well_EE_spin12[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EE_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EE_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );

	  /* BB calculation */
          sum_BB_EE += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_BB_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[l3] * factor2 +
						 2.*(4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[l3] * 2.*sqrt(factor1*factor2) +
						 2.*(4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[l3] * sqrt(factor2) +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_BB_BB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[(lmax+1)+l3] * factor2 +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[(lmax+1)+l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_BB_EB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_BB_spin12[2*(lmax+1)+l3] + well_BB_spin12[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );

	  /* EB calculation */
          sum_EB_EB -=  (2.*(double)l3+1.) * (   (4.*wigner0[l3]*wigner0[l3]) * well_EB_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[l3] * factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * (well_EB_spin01[l3] + well_EB_spin10[l3]) * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * (well_EB_spin02[l3] + well_EB_spin20[l3]) * sqrt(factor2) +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[l3] + well_EB_spin21[l3]) * 2.*sqrt(factor1)*factor2 -
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[(lmax+1)+l3] * 4.*factor1*factor2 -
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[(lmax+1)+l3] * factor2 -
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[(lmax+1)+l3] + well_EB_spin21[(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2
                                                 );
          sum_EB_EE += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[3*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[3*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[3*(lmax+1)+l3] + well_EB_spin21[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[3*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[3*(lmax+1)+l3] * sqrt(factor2)
                                                 );
          sum_EB_BB -= (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_EB_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_EB_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_EB_spin12[2*(lmax+1)+l3] + well_EB_spin21[2*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
                                                 );
	}

      } //end loop on l3
      
      mll_TT_TT[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (4.*M_PI) * sum_TT_TT;

      mll_TE_TE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TE_TE;
      mll_TE_TB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TE_TB;
      mll_TB_TE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TB_TE;
      mll_TB_TB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TB_TB;

      mll_EE_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EE_EE;
      mll_EE_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EE_BB;
      mll_EE_EB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_EE_EB;
      mll_BB_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_BB_EE;
      mll_BB_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_BB_BB;
      mll_BB_EB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_BB_EB;
      mll_EB_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_EE;
      mll_EB_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_BB;
      mll_EB_EB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_EB;

   } //end loop l2
 } //end loop l1

  free( wigner );
  free( wigner0);
  free( wigner1);
  free( wigner2);
}



void make_mll_hybrid( int lmax, int lmin_i, int lmax_i,
		      double *well_TT,
		      double *well_EE_spin00,
		      double *well_BB_spin00, double *well_BB_spin11, double *well_BB_spin22,double *well_BB_spin01, double *well_BB_spin02, double *well_BB_spin12,
		      double *well_TE_spin00,
		      double *well_TB_spin00, double *well_TB_spin01, double *well_TB_spin02,
		      double *well_EB_spin00, double *well_EB_spin01, double *well_EB_spin10, double *well_EB_spin02, double *well_EB_spin20,
		      double *mll_TT_TT,
		      double *mll_TE_TE,
		      double *mll_TB_TB, double *mll_TB_TE,
		      double *mll_EE_EE, double *mll_EE_BB,
		      double *mll_BB_BB, double *mll_BB_EE, double *mll_BB_EB,
		      double *mll_EB_EB, double *mll_EB_EE, double *mll_EB_BB)
{
  double sum_TT_TT;
  double sum_TE_TE, sum_TE_TB;
  double sum_TB_TB, sum_TB_TE;
  double sum_EE_EE, sum_EE_BB, sum_EE_EB;
  double sum_BB_BB, sum_BB_EE, sum_BB_EB;
  double sum_EB_EB, sum_EB_EE, sum_EB_BB;
  double factor1, factor2;
  double *wigner, *wigner0, *wigner1, *wigner2;
  int l1, l2, l3, limit;

  //alloc size max = l1+l2+1s
  wigner  = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner0 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner1 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));
  wigner2 = (double *) malloc( ((lmax_i+1)+(lmax+1)+1) * sizeof(double));

  //create mll
  for( l1=lmin_i; l1<=lmax_i; l1++) {
    for( l2=2; l2<=lmax; l2++) {
      sum_TT_TT = 0.;
      sum_TE_TE = 0.;
      sum_TB_TB = 0.;
      sum_TB_TE = 0.;
      sum_EE_EE = 0.;
      sum_EE_BB = 0.;
      sum_BB_BB = 0.;
      sum_BB_EE = 0.;
      sum_BB_EB = 0.;
      sum_EB_BB = 0.;
      sum_EB_EE = 0.;
      sum_EB_EB = 0.;

      wig3j_c( l1, l2,  0, 0, wigner );
      wig3j_c( l1, l2, -2, 2, wigner0);
      wig3j_c( l1, l2, -1, 2, wigner1);
      wig3j_c( l1, l2,  0, 2, wigner2);

      /* (l+1)!/(l-1)! */
      if( l1 > 0) factor1 = (double)l1*(double)(l1+1);
      else factor1 = 0.;

      /* (l-2)!/(l+2)! */
      if( l1 > 1) factor2 = 1. / (double)(l1+2) / (double)(l1+1) / (double)l1 / (double)(l1-1);
      else factor2 = 0.;

      limit = MIN( (l1+l2), (int)lmax);
      for( l3=abs(l1-l2); l3<=limit; l3++) {

	if( (l1+l2+l3)%2 == 0) {
	  /* TT calculation */
	  sum_TT_TT += (2.*(double)l3+1.) * ( wigner[l3]*wigner[l3] ) * well_TT[l3];

	  /* TE calculation */
	  sum_TE_TE += (2.*(double)l3+1.) * ( (2.*wigner[l3]*wigner0[l3]) * well_TE_spin00[l3]);

	  /* TB calculation */
	  sum_TB_TB += (2.*(double)l3+1.) * (    (2.*wigner[l3]*wigner0[l3]) * well_TB_spin00[l3] +
                                                 (2.*wigner[l3]*wigner1[l3]) * well_TB_spin01[l3] * 2.*sqrt(factor1*factor2) +
                                                 (2.*wigner[l3]*wigner2[l3]) * well_TB_spin02[l3] * sqrt(factor2)
						 );
          sum_TB_TE += (2.*(double)l3+1.) * (    (2.*wigner[l3]*wigner0[l3]) * well_TB_spin00[2*(lmax+1)+l3] +
                                                 (2.*wigner[l3]*wigner1[l3]) * well_TB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (2.*wigner[l3]*wigner2[l3]) * well_TB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );

	  /* EE calculation */
	  sum_EE_EE += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_EE_spin00[l3]);

	  /* BB calculation */
          sum_BB_BB += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_BB_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[l3] * factor2 +
						 2.*(4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[l3] * 2.*sqrt(factor1*factor2) +
						 2.*(4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[l3] * sqrt(factor2) +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_BB_EE += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[(lmax+1)+l3] * factor2 +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[(lmax+1)+l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_BB_EB -= (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_BB_spin12[2*(lmax+1)+l3] + well_BB_spin12[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );

	  /* EB calculation */
	  sum_EB_EB +=  (2.*(double)l3+1.) * (   (4.*wigner0[l3]*wigner0[l3]) * well_EB_spin00[l3] +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[l3] * 2.*sqrt(factor1*factor2) +
						 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[l3] * sqrt(factor2)
						 );
	  sum_EB_EE -= (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );
          sum_EB_BB += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[3*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[3*(lmax+1)+l3] * sqrt(factor2)
                                                 );

	} else {
          /* EE calculation */
          sum_EE_BB += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_EE_spin00[l3]);

	  /* BB calculation */
          sum_BB_EE += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner0[l3]) * well_BB_spin00[l3] +
                                                 (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[l3] * factor2 +
						 2.*(4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[l3] * 2.*sqrt(factor1*factor2) +
						 2.*(4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[l3] * sqrt(factor2) +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_BB_BB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[(lmax+1)+l3] * factor2 +
						 2.*(4.*wigner1[l3]*wigner2[l3]) * well_BB_spin12[(lmax+1)+l3] * 2.*sqrt(factor1)*factor2
						 );
          sum_BB_EB += (2.*(double)l3+1.) * (    (4.*wigner1[l3]*wigner1[l3]) * well_BB_spin11[2*(lmax+1)+l3] * 4.*factor1*factor2 +
                                                 (4.*wigner2[l3]*wigner2[l3]) * well_BB_spin22[2*(lmax+1)+l3] * factor2 +
                                                 (4.*wigner1[l3]*wigner2[l3]) * (well_BB_spin12[2*(lmax+1)+l3] + well_BB_spin12[3*(lmax+1)+l3]) * 2.*sqrt(factor1)*factor2 +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_BB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_BB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
						 );

	  /* EB calculation */
	  sum_EB_EB -=  (2.*(double)l3+1.) * (   (4.*wigner0[l3]*wigner0[l3]) * well_EB_spin00[l3] +
                                                 (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[l3] * sqrt(factor2)
						 );
          sum_EB_EE += (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[3*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[3*(lmax+1)+l3] * sqrt(factor2)
                                                 );
          sum_EB_BB -= (2.*(double)l3+1.) * (    (4.*wigner0[l3]*wigner1[l3]) * well_EB_spin01[2*(lmax+1)+l3] * 2.*sqrt(factor1*factor2) +
                                                 (4.*wigner0[l3]*wigner2[l3]) * well_EB_spin02[2*(lmax+1)+l3] * sqrt(factor2)
                                                 );
	}

      } //end loop on l3
      
      mll_TT_TT[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (4.*M_PI) * sum_TT_TT;

      mll_TE_TE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TE_TE;
      mll_TB_TE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TB_TE;
      mll_TB_TB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (8.*M_PI) * sum_TB_TB;

      mll_EE_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EE_EE;
      mll_EE_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EE_BB;
      mll_BB_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_BB_EE;
      mll_BB_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_BB_BB;
      mll_BB_EB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / ( 8.*M_PI) * sum_BB_EB;
      mll_EB_EE[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_EE;
      mll_EB_BB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_BB;
      mll_EB_EB[l1*(lmax+1)+l2] = (2.*(double)l2+1.) / (16.*M_PI) * sum_EB_EB;

    } //end loop l2
  } //end loop l1

  free( wigner );
  free( wigner0);
  free( wigner1);
  free( wigner2);
}


//WIGNER
void wig3j_c( int l2, int l3, int m2, int m3, double *wigner)
{
  double *THRCOF;
  double l1min, l1max;
  double dl2, dl3, dm2, dm3;
  int ndim, ier=0, l;

  l1min = MAX( abs(l2-l3), abs(m2+m3));
  l1max = l2 + l3;
  ndim = (int)(l1max-l1min+1);

  dl2 = (double)l2;
  dl3 = (double)l3;
  dm2 = (double)m2;
  dm3 = (double)m3;

  for( l=0; l<l1min; l++) wigner[l] = 0.;
  THRCOF = wigner + (int)l1min;    //wig3j start at max( abs(l2-l3), abs(m2+m3))

  if( l2<abs(m2) || l3<abs(m3) || (l1max-l1min)<0 ) for( l=0; l<ndim; l++) THRCOF[l] = 0.;
  else wig3j_( &dl2, &dl3, &dm2, &dm3, &l1min, &l1max, THRCOF, &ndim, &ier);

  if( ier) {
    for( l=0; l<ndim; l++) THRCOF[l] = 0.;
    printf( "err=%d  l2=%d, l3=%d, m2=%d, m3=%d : ", ier, l2, l3, m2, m3);
    switch( ier)
    {
    case 1 : printf( "Either L2.LT.ABS(M2) or L3.LT.ABS(M3)\n"); break;
    case 2 : printf( "Either L2+ABS(M2) or L3+ABS(M3) non-integer\n"); break;
    case 3 : printf( "L1MAX-L1MIN not an integer (l1min=%d, l1max=%d)\n", (int)l1min, (int)l1max); break;
    case 4 : printf( "L1MAX less than L1MIN (l1min=%d, l1max=%d)\n", (int)l1min, (int)l1max); break;
    case 5 : printf( "NDIM less than L1MAX-L1MIN+1 (ndim=%d)\n", ndim); break;
    }
    fflush(stdout);
  }

}

