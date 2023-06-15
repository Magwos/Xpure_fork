//#include "xpol_s2hat.h"
#include "xpure.h"
#include "s2hat_pure.h"
// #include "sprng.h"
#include <fitsio.h>

#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define WRITE_FULLMBB 0
#define WRITE_MBB 0
#define TESTOUT 0
#define WRITE_MAP 0

void PrintHelp();

void write_fits_pseudo( int nele, double *vect, char *outfile, int col);
void write_fits_pure( int nele, double *vect, char *outfile, int col);
void read_fits_puremll( char *infile, int nele, double *mll);
// int get_random_alms( int*, int, double*, int, int, int*, s2hat_dcomplex*);
// void sprng_gaussran( int*, int, double*);
void apply_mask( int nmaps, int map_size, int ncomp,
		 double *local_map, double *local_mask, double *local_apodizedmap);
void apply_maskTT( int nmaps, int map_size,
                 double *local_map, double *local_mask, double *local_apodizedmap);



//=========================================================================
//                          PROGRAMME PRINCIPAL 
//=========================================================================
int main(int argc, char * argv[])
{
  int nprocs, l, b, m;
  int root=0;
  int gangroot=0;

  /* Parameters */
  int i, j;
  long p;
  int mode;
  int nside, nmaps, nstokes, ncomp, lmax, nbins;
  int nlmaxSim, nmmaxSim;
  int write_pseudo, is_set, is_noise, is_bintab, is_nhit, cmbFromMap, cmbFromCl, combine_masks;
  char mask_list[1024];
  char *pfile;
  char **mapfile, **beamfile;
  char *mllfileTT_TT, *mllfileTE_TE, *mllfileTB_TB, *mllfileEE_EE, *mllfileEE_BB, *mllfileBB_BB, *mllfileBB_EE, *mllfileEB_EB;
  char *maskfile_T, *maskfile_E_spin0, *maskfile_E_spin1, *maskfile_E_spin2, *maskfile_B_spin0, *maskfile_B_spin1, *maskfile_B_spin2;
  char *nhitfileT, *nhitfileP;

  char val[1024], cmd[256], mbbfileT[1024], mbbfile1[1024], mbbfile2[1024];
  char binfile[1024], cellfile[1024], pseudofile[1024], inpCellfile[1024], inpBellfile[1024];
  double *sigmaP, *sigmaT;
  double *biasT, *biasP; /* sigma */
  char *healpixdatadir=HEALPIXDATA;
  char pixfile[1024];
  double *wint, *winp, *pixwin;
  double specIndex;

  long npix;
  
  /* MPI Parameters */
  MPI_Comm gang_comm;
  MPI_Comm root_comm;
  int nmasks, ngangs, gangsize, gangnmaps, gangnum, gangrank;

  int rank;

  MPI_Init( &argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs);

  if( rank == root) {
    printf( "                      ***********************************\n");
    printf( "\n");
    printf( "                                                         \n");
    printf( "                                                         \n");
    printf( "                                                         \n");
    printf( "                        XX    X                          \n");
    printf( "                         XX  XX                          \n");
    printf( "                          XXXX  PPPPP  U   U RRRR  EEEEE \n");
    printf( "                           XX   P    P U   U R   R E     \n");
    printf( "                          XXXX  PPPPP  U   U RRRR  EEE   \n");
    printf( "                         XX  XX P      U   U R  R  E     \n");
    printf( "                        XX    X P      UUUUU R   R EEEEE \n");
    printf( "\n");
    printf( "                      ***********************************\n");
    printf( "\n");
  }    

  //-----------------------------------------------------------------------
  // Read parameters
  //-----------------------------------------------------------------------
  if( argc != 2) {
    if(rank == root) printf( "Call is \n");
    if(rank == root) printf( "  xpure paramfile\n");
    MPI_Finalize();
    exit(-1);
  }

  pfile = argv[1];
  //if( strcmp(pfile, "help") <= 0) {
  //  if( rank == root) PrintHelp();
  //  exit( -1);
  //}


  if( rank == root) printf( "lecture parametres...\n");
  fflush( stdout);

  get_parameter( pfile, "nside", val);
  nside = atoi( val);
  get_parameter( pfile, "nmaps", val);
  nmaps = atoi( val);
  get_parameter( pfile, "nmasks", val); /* nmasks = ngangs = 1 or nbins */
  nmasks = ngangs = atoi( val);

  /* beamfile */
  beamfile = (char **) malloc( nmaps*sizeof(char *));
  for( i=0; i<nmaps; i++) {
    beamfile[i] = (char *) malloc( 1024*sizeof(char));
    sprintf( cmd, "bellfile%d", i+1);
    get_parameter( pfile, cmd, beamfile[i]);
  }

  /* mapfile */
  cmbFromMap = 1;
  mapfile = (char **) malloc( nmaps*sizeof(char *));
  for( i=0; i<nmaps; i++) {
    mapfile[i]  = (char *) malloc( 1024*sizeof(char));

    sprintf( cmd, "mapfile%d", i+1);
    cmbFromMap = get_parameter( pfile, cmd, mapfile[i]) && cmbFromMap;
  }

  /* inpCellfile and lmaxSim */
  cmbFromCl = 0;
  if( !cmbFromMap) {
    cmbFromCl = 1;

    cmbFromCl = cmbFromCl && get_parameter( pfile, "inpCellfile", inpCellfile);
    get_parameter( pfile, "inpBellfile", inpBellfile);

    is_set = get_parameter( pfile, "lmaxSim", val);
    if( is_set) nlmaxSim = atoi( val); else nlmaxSim = lmax;
    nmmaxSim = nlmaxSim;
  }
  if( rank == root) printf( " ** cmbFromMap = %d\n", cmbFromMap);
  if( rank == root) printf( " ** cmbFromCl  = %d\n", cmbFromCl );


  /* Sigma for noise simulation */
  is_noise = 1;
  sigmaT = (double *) malloc( nmaps*sizeof(double));
  sigmaP = (double *) malloc( nmaps*sizeof(double));
  for( i=0; i<nmaps; i++) {
    sprintf( cmd, "sigmaT%d", i+1);
    is_noise = (is_noise && get_parameter( pfile, cmd, val));
    sigmaT[i] = atof( val);
   
    sprintf( cmd, "sigmaP%d", i+1);
    is_noise = (is_noise && get_parameter( pfile, cmd, val));
    sigmaP[i] = atof( val);
  }
  if( is_noise && rank == root) printf( "Add Noise !\n");

  //Parameters
  is_set = get_parameter( pfile, "lmax", val);
  if( is_set) lmax = atoi( val);
  else {
    lmax = 3*nside-1;
    if( rank == root) printf( "lmax set to lmax=%i\n", lmax);
  }

  if( lmax > 3*nside-1 || lmax <= 0) {
    lmax = 3*nside-1;
    if( rank == root) printf( "WARNING : lmax too high, set at lmax=%i\n", lmax);
  }


  /* Standard, Pure or Hybrid */
  get_parameter( pfile, "mode", val);
  mode  = atoi( val);

  if( rank == root) {
    if( mode == 0) printf("STANDARD FORMALISM...\n");
    if( mode == 1) printf("PURE FORMALISM...\n");
    if( mode == 2) printf("HYBRID FORMALISM...\n");
  }


  /* combine masks ? */
  combine_masks = get_parameter( pfile, "mask_list", mask_list);


  /* Remove noise bias */
  biasT = (double *) calloc( nmaps, sizeof(double));
  biasP = (double *) calloc( nmaps, sizeof(double));
  for( i=0; i<nmaps; i++) {
    sprintf( cmd, "noise_biasT_%d", i+1);
    is_set = get_parameter( pfile, cmd, val);
    if( is_set) biasT[i] = atof( val);
/*     if( rank == root) printf( "biasT[%d] = %e\n", i, biasT[i]); */

    sprintf( cmd, "noise_biasP_%d", i+1);
    is_set = get_parameter( pfile, cmd, val);
    if( is_set) biasP[i] = atof( val);
/*     if( rank == root) printf( "biasP[%d] = %e\n", i, biasP[i]); */
  }

  npix = 12*(long)nside*(long)nside;
  nstokes = 2;   /* 2 nstokes E and B */
  ncomp   = 2;   /* 2 spin masks */


  //-----------------------------------------------------------------------
  // MPI initialization
  // divide all the procs into gangs - each gang will work on one set of maps/alms
  //-----------------------------------------------------------------------
  gangsize = nprocs/ngangs; /* nproc per gang */
  if( gangsize*ngangs != nprocs) {
    if( rank == root)
      printf( "Need %i procs for %i gangs of %i procs\n", gangsize*ngangs, ngangs, gangsize);
    exit( 1);
  } else if( rank == root) printf( "%i gangs with %i proc per gang\n", ngangs, gangsize);

  gangnmaps = nmaps;

  gangnum  = rank/gangsize;
  gangrank = rank%gangsize;
  MPI_Comm_split( MPI_COMM_WORLD, gangnum, gangrank, &gang_comm);
  MPI_Comm_split( MPI_COMM_WORLD, gangrank, gangnum, &root_comm);
  if( gangrank == gangroot) printf( "gang %i  : %i maps\n", gangnum, nmaps);


  //-----------------------------------------------------------------------
  // Read parameters
  //-----------------------------------------------------------------------
  maskfile_T       = (char *) malloc( 1024*sizeof(char));
  maskfile_E_spin0 = (char *) malloc( 1024*sizeof(char));
  maskfile_E_spin1 = (char *) malloc( 1024*sizeof(char));
  maskfile_E_spin2 = (char *) malloc( 1024*sizeof(char));
  maskfile_B_spin0 = (char *) malloc( 1024*sizeof(char));
  maskfile_B_spin1 = (char *) malloc( 1024*sizeof(char));
  maskfile_B_spin2 = (char *) malloc( 1024*sizeof(char));

  mllfileTT_TT   = (char *) malloc( 1024*sizeof(char));
  mllfileTE_TE   = (char *) malloc( 1024*sizeof(char));
  mllfileTB_TB   = (char *) malloc( 1024*sizeof(char));
  mllfileEE_EE   = (char *) malloc( 1024*sizeof(char));
  mllfileEE_BB   = (char *) malloc( 1024*sizeof(char));
  mllfileBB_BB   = (char *) malloc( 1024*sizeof(char));
  mllfileBB_EE   = (char *) malloc( 1024*sizeof(char));
  mllfileEB_EB   = (char *) malloc( 1024*sizeof(char));

  nhitfileT = (char *) malloc( 1024*sizeof(char));
  nhitfileP = (char *) malloc( 1024*sizeof(char));

  sprintf( cmd, "mllfile_TT_TT_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTT_TT);
  sprintf( cmd, "mllfile_TE_TE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTE_TE);
  sprintf( cmd, "mllfile_TB_TB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileTB_TB);
  sprintf( cmd, "mllfile_EE_EE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEE_EE);
  sprintf( cmd, "mllfile_EE_BB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEE_BB);
  sprintf( cmd, "mllfile_BB_BB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileBB_BB);
  sprintf( cmd, "mllfile_BB_EE_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileBB_EE);
  sprintf( cmd, "mllfile_EB_EB_%d", gangnum+1);
  get_parameter( pfile, cmd, mllfileEB_EB);

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

  is_bintab = get_parameter( pfile,   "bintab",  binfile);
  get_parameter( pfile, "cellfile", cellfile);
    
  write_pseudo = get_parameter( pfile, "pseudofile", pseudofile);
  is_nhit = 0;
  if( is_noise) {
    is_nhit = get_parameter( pfile, "nhitfileT", nhitfileT);
    is_nhit = is_nhit && get_parameter( pfile, "nhitfileP" , nhitfileP );
  }

  //-----------------------------------------------------------------------
  // Bin tables
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( " - read the binnings...\t");
  int bshift = -1;
  int *binshift, *nbin_per_mask; /* for combining masks */
  double *bintot=NULL;
  double *bintab;

  /* read the bin table */
  if( is_bintab) {
    nbins = get_vect_size( binfile, lmax);
    if( gangrank == gangroot) printf( " (%d) ", nbins);
    bintot = (double *) malloc( (nbins+1)*sizeof(double));
    read_fits_vect( binfile, bintot, nbins, 1, 0);
  } else {
    if( gangrank == gangroot) printf( " fill bintab ");
    nbins = lmax+1;
    bintot = (double *) malloc( (nbins+1)*sizeof(double));
    for( i=0; i<=nbins; i++) bintot[i] = i;
  }
  
  
  /* reduce bintab to lmax */
  b=0;
  while( bintot[b] < 2) b++;
  bintab = &bintot[b];
  nbins -= b;
  while( bintab[nbins-1] >= lmax) nbins--;
  if( gangrank == gangroot) printf( "-> %d \n", nbins);
  bintab[nbins] = lmax+1;
/*   if( gangrank == gangroot) for( i=0; i<nbins; i++) printf( "%d ", (int)bintab[i]); */
/*   if( gangrank == gangroot) printf("\n"); fflush(stdout); */

  if( combine_masks) {
    binshift      = (int *) calloc( nmasks, sizeof(int));
    nbin_per_mask = (int *) calloc( nmasks, sizeof(int));
    BinShifts( nbins, mask_list, binshift, nbin_per_mask, rank);

    if( rank==root) for( b=0; b<nmasks; b++) printf( "mask:%d  binshift=%d  nbin=%d\n", b, binshift[b], nbin_per_mask[b]);
  }


  //-----------------------------------------------------------------------
  // define the S2HAT structures
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( " - Define s2hat structures\n");
  s2hat_pixeltype cpixelization;
  s2hat_scandef cscan;
  s2hat_pixparameters pixpar;

  int nlmax=lmax, nmmax=lmax;
  int lda = nlmax+1;   /* =nlmax (S2HAT) / =ncomp (HEALPIX) */

  /* define the maps parameters and data distribution */
  int pixchoice = PIXCHOICE_HEALPIX;   /* only supported now */
  pixpar.par1 = nside;
  set_pixelization( pixchoice, pixpar, &cpixelization);   /* define the C pixelization structure */

  /* scan definition */
  if( gangrank == gangroot) printf( " - Scan definition\n");
  double *mask;
  int *intmask;
  int fsky=0;
  if( gangrank == gangroot) {

    /* alloc */
    mask = (double *) calloc( npix, sizeof(double));

    /* read mask files */
    printf( " - Read Masks\n");

    printf( "\tTemperature : %s\n", maskfile_T);
    read_fits_map( nside, mask, maskfile_T, 1);

    /* create scan pixelisation */
    intmask = (int *) calloc( npix, sizeof( int));
    for( p=0; p<npix; p++)
      if( mask[p] != 0) {
	intmask[p] = 1;
	fsky++;
      }
    printf( "\t %f%%\n", (double)fsky/(double)npix*100.);
    free( mask);

    mask2scan( intmask, cpixelization, &cscan);
    free( intmask);
  }
  MPI_scanBcast( cpixelization, &cscan, gangroot, gangrank, gang_comm);


  //-----------------------------------------------------------------------
  // get local sizes
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( " - Get local sizes\n");
  int plms=0, nmvals, nmvalsSim, first_ring, last_ring, map_size;
  int *mvals, *mvalsSim;
  long int nplm;
  get_local_data_sizes( plms, cpixelization, cscan, nlmax, nmmax, gangrank, gangsize,
			&nmvals, &first_ring, &last_ring, &map_size, &nplm, gangroot, gang_comm);  

  mvals = (int *) calloc( nmvals, sizeof( int));
  find_mvalues( gangrank, gangsize, nmmax, nmvals, mvals);

  /* and for CMB maps sims if needed */
  if( cmbFromCl) {
    get_local_data_sizes( plms, cpixelization, cscan, nlmaxSim, nmmaxSim, gangrank, gangsize,
			  &nmvalsSim, &first_ring, &last_ring, &map_size, &nplm, gangroot, gang_comm);

    mvalsSim = (int *) calloc( nmvalsSim, sizeof( int));
    find_mvalues( gangrank, gangsize, nmmaxSim, nmvalsSim, mvalsSim);
  }
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // distribute windows
  //-----------------------------------------------------------------------
  double *local_mask_T;
  double *local_mask_E_spin0, *local_mask_E_spin1, *local_mask_E_spin2;
  double *local_mask_B_spin0, *local_mask_B_spin1, *local_mask_B_spin2;
  double *local_nhitT, *nhitT, *local_nhitP, *nhitP;

  if( gangrank == gangroot) printf( " - S2HAT : distribute windows\n");
  local_mask_T       = (double *) calloc(       map_size, sizeof(double));
  local_mask_E_spin0 = (double *) calloc( ncomp*map_size, sizeof(double));
  if( mode > 0) local_mask_E_spin1 = (double *) calloc( ncomp*map_size, sizeof(double));
  if( mode > 0) local_mask_E_spin2 = (double *) calloc( ncomp*map_size, sizeof(double));
  local_mask_B_spin0 = (double *) calloc( ncomp*map_size, sizeof(double));
  if( mode > 0) local_mask_B_spin1 = (double *) calloc( ncomp*map_size, sizeof(double));
  if( mode > 0) local_mask_B_spin2 = (double *) calloc( ncomp*map_size, sizeof(double));

  read_distribute_map( cpixelization, 1, 0, 1, maskfile_T, first_ring, last_ring, map_size, 
		       local_mask_T, gangrank, gangsize, gangroot, gang_comm);

  read_distribute_map( cpixelization, 1, 0, 1, maskfile_E_spin0, first_ring, last_ring, map_size, 
		       local_mask_E_spin0, gangrank, gangsize, gangroot, gang_comm);
  read_distribute_map( cpixelization, 1, 0, 1, maskfile_B_spin0, first_ring, last_ring, map_size, 
		       local_mask_B_spin0, gangrank, gangsize, gangroot, gang_comm);

  if( mode == 1) {
    read_distribute_map( cpixelization, 1, 0, ncomp, maskfile_E_spin1, first_ring, last_ring, map_size, 
			 local_mask_E_spin1, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, 1, 0, ncomp, maskfile_E_spin2, first_ring, last_ring, map_size, 
			 local_mask_E_spin2, gangrank, gangsize, gangroot, gang_comm);

    read_distribute_map( cpixelization, 1, 0, ncomp, maskfile_B_spin1, first_ring, last_ring, map_size, 
			 local_mask_B_spin1, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, 1, 0, ncomp, maskfile_B_spin2, first_ring, last_ring, map_size, 
			 local_mask_B_spin2, gangrank, gangsize, gangroot, gang_comm);
  }

  if( mode == 2) {
    read_distribute_map( cpixelization, 1, 0, ncomp, maskfile_B_spin1, first_ring, last_ring, map_size, 
			 local_mask_B_spin1, gangrank, gangsize, gangroot, gang_comm);
    read_distribute_map( cpixelization, 1, 0, ncomp, maskfile_B_spin2, first_ring, last_ring, map_size, 
			 local_mask_B_spin2, gangrank, gangsize, gangroot, gang_comm);
  }


  /* nhit map */
  if( is_noise && is_nhit) {
	  
    /* root reads the map */
    if( gangrank == gangroot) {
      nhitT = (double *) malloc( npix*sizeof(double));
      nhitP = (double *) malloc( npix*sizeof(double));
    }
    if( rank == root) {
      read_fits_map( nside, nhitT, nhitfileT, 1);
      read_fits_map( nside, nhitP, nhitfileP,  1);
    }

    /* Broadcast to all gangroots */
    if( gangrank == gangroot) {
      MPI_Bcast( nhitT, npix, MPI_DOUBLE, root, root_comm);
      MPI_Bcast( nhitP, npix, MPI_DOUBLE, root, root_comm);
    }
      
    /* distribute inside the gang */
    local_nhitT = (double *) malloc( map_size*sizeof(double));
    local_nhitP  = (double *) malloc( map_size*sizeof(double));
    distribute_map( cpixelization, gangnmaps, 0, 1, first_ring, last_ring, map_size,
		    local_nhitT, nhitT, gangrank, gangsize, gangroot, gang_comm);    
    distribute_map( cpixelization, gangnmaps, 0, 1, first_ring, last_ring, map_size,
		    local_nhitP, nhitP, gangrank, gangsize, gangroot, gang_comm);    

    if( gangrank == gangroot) free( nhitT);
    if( gangrank == gangroot) free( nhitP);
  }
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Compute f_sky
  //-----------------------------------------------------------------------
  double fskyT, fskyspin0_E, fskyspin1_E, fskyspin2_E, fskyspin0_B, fskyspin1_B, fskyspin2_B;
  double local_fskyT=0.;
  double local_fskyspin0_E=0., local_fskyspin1_E=0., local_fskyspin2_E=0.;
  double local_fskyspin0_B=0., local_fskyspin1_B=0., local_fskyspin2_B=0.;

  if( rank == root) printf( " - SKY COVERAGE : compute temperature and polarization noise-weighted effective Fsky\n"); 
 if( is_nhit) {

    for( p=0; p<map_size; p++) {
      if( local_nhitT[p] !=0 || local_nhitP[p] !=0) {
	local_fskyT += local_mask_T[p] * local_mask_T[p]  / local_nhitT[p];
	
	local_fskyspin0_E += local_mask_E_spin0[p] * local_mask_E_spin0[p] / local_nhitP[p];   
        local_fskyspin0_B += local_mask_B_spin0[p] * local_mask_B_spin0[p] / local_nhitP[p];

	if( mode > 0) local_fskyspin1_E += (local_mask_E_spin1[p] * local_mask_E_spin1[p] + local_mask_E_spin1[map_size+p] * local_mask_E_spin1[map_size+p]) / local_nhitP[p];
        if( mode > 0) local_fskyspin1_B += (local_mask_B_spin1[p] * local_mask_B_spin1[p] + local_mask_B_spin1[map_size+p] * local_mask_B_spin1[map_size+p]) / local_nhitP[p];

	if( mode > 0) local_fskyspin2_E += (local_mask_E_spin2[p] * local_mask_E_spin2[p] + local_mask_E_spin2[map_size+p] * local_mask_E_spin2[map_size+p]) / local_nhitP[p];
        if( mode > 0) local_fskyspin2_B += (local_mask_B_spin2[p] * local_mask_B_spin2[p] + local_mask_B_spin2[map_size+p] * local_mask_B_spin2[map_size+p]) / local_nhitP[p];
      }
    }

  } else {

    for( p=0; p<map_size; p++) {
      local_fskyT       += local_mask_T[p] * local_mask_T[p];
      
      local_fskyspin0_E += local_mask_E_spin0[p] * local_mask_E_spin0[p];
      local_fskyspin0_B += local_mask_B_spin0[p] * local_mask_B_spin0[p];

      if( mode > 0) local_fskyspin1_E += local_mask_E_spin1[p] * local_mask_E_spin1[p] + local_mask_E_spin1[map_size+p] * local_mask_E_spin1[map_size+p];
      if( mode > 0) local_fskyspin1_B += local_mask_B_spin1[p] * local_mask_B_spin1[p] + local_mask_B_spin1[map_size+p] * local_mask_B_spin1[map_size+p];

      if( mode > 0) local_fskyspin2_B += local_mask_B_spin2[p] * local_mask_B_spin2[p] + local_mask_B_spin2[map_size+p] * local_mask_B_spin2[map_size+p];
      if( mode > 0) local_fskyspin2_E += local_mask_E_spin2[p] * local_mask_E_spin2[p] + local_mask_E_spin2[map_size+p] * local_mask_E_spin2[map_size+p];
    }

  }
  if( gangrank == gangroot) printf( " - reduce\n");
  MPI_Allreduce( &local_fskyT,       &fskyT,       1, MPI_DOUBLE, MPI_SUM, gang_comm);
  MPI_Allreduce( &local_fskyspin0_E, &fskyspin0_E, 1, MPI_DOUBLE, MPI_SUM, gang_comm);
  MPI_Allreduce( &local_fskyspin0_B, &fskyspin0_B, 1, MPI_DOUBLE, MPI_SUM, gang_comm);
  if( mode > 0) MPI_Allreduce( &local_fskyspin1_E, &fskyspin1_E, 1, MPI_DOUBLE, MPI_SUM, gang_comm);
  if( mode > 0) MPI_Allreduce( &local_fskyspin1_B, &fskyspin1_B, 1, MPI_DOUBLE, MPI_SUM, gang_comm);
  if( mode > 0) MPI_Allreduce( &local_fskyspin2_E, &fskyspin2_E, 1, MPI_DOUBLE, MPI_SUM, gang_comm);
  if( mode > 0) MPI_Allreduce( &local_fskyspin2_B, &fskyspin2_B, 1, MPI_DOUBLE, MPI_SUM, gang_comm);

  if( gangrank == gangroot) {
    printf( "\t fsky = %f%%\tfsky_T = %f%%\n", (double)fsky/(double)npix*100., (double)fskyT/(double)npix*100.);
    printf( "\t E-mode: fsky_0 = %f%%\tfsky_1 = %f%%\tfsky_2 = %f%%\n", (double)fskyspin0_E/(double)npix*100., (double)fskyspin1_E/(double)npix*100., (double)fskyspin2_E/(double)npix*100.);
    printf( "\t B-mode: fsky_0 = %f%%\tfsky_1 = %f%%\tfsky_2 = %f%%\n", (double)fskyspin0_B/(double)npix*100., (double)fskyspin1_B/(double)npix*100., (double)fskyspin2_B/(double)npix*100.);
  }
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Get signal map and distribute
  //-----------------------------------------------------------------------
  int imap, ncmbmap=1;
  double aEE, aTE, aBB, aTB, aEB;
  double *map, *local_map, *local_mapT, *local_mapP;
  double *local_noise, *inputPS, *inputT, *inputG, *inputC, *inputTG, *inputTC, *inputGC, *inputBl;
  s2hat_dcomplex *local_simu_almT, *local_simu_almP;
  int *cmb_random_streamT, *cmb_random_streamP, *noise_random_streamT, *noise_random_streamP;
  int iseed = 985456376;
  // int sprng_type=SPRNG_LFG;
  int nstreamsPerProc, nstreams;
  int cmb_random_streamT_no, cmb_random_streamP_no, noise_random_streamT_no, noise_random_streamP_no;

  local_mapP = (double *) calloc( ncomp*gangnmaps*map_size, sizeof(double));
  local_mapT = (double *) calloc( ncomp*gangnmaps*map_size, sizeof(double));

  /* initialize the random number generator */ 
  // nstreamsPerProc = 0;
  // if( cmbFromCl) nstreamsPerProc += 2;
  // if(  is_noise) nstreamsPerProc += 2;
  // nstreams = gangsize*nstreamsPerProc; /* total number of independant stream */
  // cmb_random_streamT_no   = gangrank*nstreamsPerProc;
  // cmb_random_streamP_no   = gangrank*nstreamsPerProc + 1;
  // noise_random_streamT_no = gangrank*nstreamsPerProc + nstreamsPerProc-2;
  // noise_random_streamP_no = gangrank*nstreamsPerProc + nstreamsPerProc-1;

  // if( rank == root) iseed = make_sprng_seed();
  MPI_Bcast( &iseed, 1, MPI_INT, root, MPI_COMM_WORLD);
  
//   if( cmbFromCl) cmb_random_streamT   = init_sprng( sprng_type, cmb_random_streamT_no,   nstreams, iseed, SPRNG_DEFAULT);
//   if( cmbFromCl) cmb_random_streamP   = init_sprng( sprng_type, cmb_random_streamP_no,   nstreams, iseed, SPRNG_DEFAULT);
//   if(  is_noise) noise_random_streamT = init_sprng( sprng_type, noise_random_streamT_no, nstreams, iseed, SPRNG_DEFAULT);
//   if(  is_noise) noise_random_streamP = init_sprng( sprng_type, noise_random_streamP_no, nstreams, iseed, SPRNG_DEFAULT);

// #if TESTOUT
//   if( cmbFromCl) print_sprng(   cmb_random_streamT);
//   if( cmbFromCl) print_sprng(   cmb_random_streamP);
//   if(  is_noise) print_sprng( noise_random_streamT);
//   if(  is_noise) print_sprng( noise_random_streamP);
// #endif

  /* generate CMB map from input-Cl */
//   if( cmbFromCl) {
//     if( gangrank == gangroot) printf( " - Create Sky signal\n");

//     /* and read the power spectrum */
//     inputPS = (double *) calloc( 6*(nlmaxSim+1), sizeof(double));
//     inputT  = &inputPS[0*(nlmaxSim+1)];
//     inputG  = &inputPS[1*(nlmaxSim+1)];
//     inputC  = &inputPS[2*(nlmaxSim+1)];
//     inputTG = &inputPS[3*(nlmaxSim+1)];
//     inputTC = &inputPS[4*(nlmaxSim+1)];  
//     inputGC = &inputPS[5*(nlmaxSim+1)];
    
//     if( rank == root) {
//       /* read polarization pixwin */
//       pixwin = (double *) malloc( 2*(nlmaxSim+1)*sizeof(double));
//       sprintf( pixfile, "%s/pixel_window_n%04d.fits", healpixdatadir, nside);
//       read_fits_vect( pixfile, &pixwin[(nlmaxSim+1)*0], nlmaxSim+1, 1, 1);
//       read_fits_vect( pixfile, &pixwin[(nlmaxSim+1)*1], nlmaxSim+1, 2, 1);
//       pixwin[0] = pixwin[1] = pixwin[nlmaxSim+1] = pixwin[nlmaxSim+2] = 1.;
// /*       printf( "pix=%e\n", pixwin[10]); */

//       /* read input beam transfer function */
//       inputBl = (double *) malloc( (nlmaxSim+1)*sizeof(double));
//       read_fits_vect( inpBellfile, inputBl, nlmaxSim+1, 1, 1);
// /*       printf( "Bl=%e\n", inputBl[10]); */
      
//       /* reads input spectra */
//       printf( "\tRead Temperature Spectra\n");
//       printf( "\t\t%s\n", inpCellfile);
//       for( i=0; i<6; i++) read_fits_vect( inpCellfile, &inputPS[i*(nlmaxSim+1)], nlmaxSim+1, i+1, 0);
// /*       printf( "TT=%e, EE=%e, BB=%e, TE=%e, TB=%e, EB=%e\n", inputT[10], inputG[10], inputC[10], inputTG[10], inputTC[10], inputGC[10]); */
//       for( l=0; l<=nlmaxSim; l++) {
// 	inputT[l]  *= inputBl[l]*inputBl[l]*pixwin[l]*pixwin[l];
// 	inputG[l]  *= inputBl[l]*inputBl[l]*pixwin[(nlmaxSim+1)+l]*pixwin[(nlmaxSim+1)+l];
// 	inputC[l]  *= inputBl[l]*inputBl[l]*pixwin[(nlmaxSim+1)+l]*pixwin[(nlmaxSim+1)+l];
// 	inputTG[l] *= inputBl[l]*inputBl[l]*pixwin[l]*pixwin[(nlmaxSim+1)+l];
// 	inputTC[l] *= inputBl[l]*inputBl[l]*pixwin[l]*pixwin[(nlmaxSim+1)+l];
// 	inputGC[l] *= inputBl[l]*inputBl[l]*pixwin[(nlmaxSim+1)+l]*pixwin[(nlmaxSim+1)+l];
//       }
      
//       free( inputBl);
//       free( pixwin);
//     }
    
//     MPI_Barrier( MPI_COMM_WORLD);
//     if( gangrank == gangroot) printf("\triri\n");

//     /* broadcast input cls */
//     MPI_Bcast( inputPS, 6*(nlmaxSim+1), MPI_DOUBLE, root, MPI_COMM_WORLD);

//     /* Generate random sky */
//     if( gangrank == gangroot) printf("\tGenerate Random Alm's for Temperature\n");
//     local_simu_almT = (s2hat_dcomplex *) calloc( (nlmaxSim+1)*nmvalsSim*ncomp, sizeof( s2hat_dcomplex));
//     get_random_alms( cmb_random_streamT, nlmaxSim, inputT, ncomp, nmvalsSim, mvalsSim, local_simu_almT);
//     local_simu_almT = (s2hat_dcomplex *) realloc( local_simu_almT, (nlmaxSim+1)*nmvalsSim*sizeof( s2hat_dcomplex));

//     if( gangrank == gangroot) printf("\tGenerate Random Alm's for Polarization\n");
//     local_simu_almP = (s2hat_dcomplex *) calloc( (nlmaxSim+1)*nmvalsSim*ncomp, sizeof( s2hat_dcomplex));
//     get_random_alms( cmb_random_streamP, nlmaxSim, &inputPS[(nlmaxSim+1)], ncomp, nmvalsSim, mvalsSim, local_simu_almP);

//     /* Create TE correlations*/
//     if( gangrank == gangroot) printf("\tGenerate Random Alm's for TE, TB and EB correlations\n");
//     for( l=2; l<nlmaxSim+1; l++) {
//       /*Compute Cholesky coefficients*/
//       aTE = aEE = 0.;
//       aTB = aEB = aBB = 0.;
      
//       if( inputT[l] > 0.) {
// 	aTE = inputTG[l]/inputT[l];
// 	aTB = inputTC[l]/inputT[l];
//       }
//       if( inputG[l] > 0.) {
// 	aEE = sqrt(inputG[l]-inputTG[l]*inputTG[l]/inputT[l])/sqrt(inputG[l]);
// 	aEB = (inputGC[l]-aTE*aTB*inputT[l])/inputG[l]/aEE;
//       }
//       if( inputC[l] > 0.) {
// 	aBB = sqrt(inputC[l]-aTB*aTB*inputT[l]-aEB*aEB*inputG[l])/sqrt(inputC[l]);
//       }
      
//       /*Check consistency*/
//       if( (inputG[l]*inputT[l]-inputTG[l]*inputTG[l]) < 0.0) {
// 	printf("\t\t Incoherent TT, EE and TE spectra: Coefficient G->E is not defined!!!\n");
// 	exit( -1);
//       }
//       //if( (inputGC[l]*inputT[l]-inputTG[l]*inputTC[l]) < 0.0) {
//       //printf("\t\t Incoherent TT, TE, TB and EB spectra: Coefficient G->B is not defined!!!\n");
//       //exit( -1);
//       //}
//       if( (inputC[l]-aTB*aTB*inputT[l]-aEB*aEB*inputG[l]) < 0.0) {
// 	printf("\t\t Incoherent TT, EE, BB, TE, TB and EB spectra: Coefficient C->B is not defined!!!\n");
//         exit( -1);
//       }

//       for( m=0; m<nmvalsSim; m++) {	
// 	/*create TB/EB correlations*/
// 	local_simu_almP[ nmvalsSim*(nlmaxSim+1)+m*(nlmaxSim+1)+l].re *= aBB;
// 	local_simu_almP[ nmvalsSim*(nlmaxSim+1)+m*(nlmaxSim+1)+l].re += aEB*local_simu_almP[ m*(nlmaxSim+1)+l].re;
// 	local_simu_almP[ nmvalsSim*(nlmaxSim+1)+m*(nlmaxSim+1)+l].re += aTB*local_simu_almT[ m*(nlmaxSim+1)+l].re;
	
// 	local_simu_almP[ nmvalsSim*(nlmaxSim+1)+m*(nlmaxSim+1)+l].im *= aBB;
// 	local_simu_almP[ nmvalsSim*(nlmaxSim+1)+m*(nlmaxSim+1)+l].im += aEB*local_simu_almP[ m*(nlmaxSim+1)+l].im;
// 	local_simu_almP[ nmvalsSim*(nlmaxSim+1)+m*(nlmaxSim+1)+l].im += aTB*local_simu_almT[ m*(nlmaxSim+1)+l].im;
	
// 	/*create TE correlations*/
// 	local_simu_almP[ m*(nlmaxSim+1)+l].re *= aEE;
// 	local_simu_almP[ m*(nlmaxSim+1)+l].re += aTE*local_simu_almT[ m*(nlmaxSim+1)+l].re;
	
// 	local_simu_almP[ m*(nlmaxSim+1)+l].im *= aEE;
// 	local_simu_almP[ m*(nlmaxSim+1)+l].im += aTE*local_simu_almT[ m*(nlmaxSim+1)+l].im;
	
//       }
//     }

//     /* free input spectra */
//     free( inputPS);


// #if TESTOUT
//     /* write generated spectra */
//     double *xclSim;
//     xclSim = (double *) malloc( ncomp*ncomp*(nlmaxSim+1)*ncomp*sizeof(double));

//     collect_xls( 1, 0, 1, 0, ncomp, nlmaxSim, nmvalsSim, mvalsSim, nlmaxSim,
// 		 local_simu_almT, local_simu_almT, ncomp, xclSim, gangrank, gangsize, root, gang_comm);
//     if( gangrank == root) sprintf( cmd, "testcl_Temp%d.fits", rank);
//     if( gangrank == root) write_fits_vect( ncomp*(nlmaxSim+1),  xclSim, cmd, 1);
    
//     collect_xls( 1, 0, 1, 0, ncomp, nlmaxSim, nmvalsSim, mvalsSim, nlmaxSim,
// 		 local_simu_almP, local_simu_almP, ncomp*ncomp, xclSim, gangrank, gangsize, root, gang_comm);
//     if( gangrank == root) sprintf( cmd, "testcl_Pol%d.fits", rank);
//     if( gangrank == root) write_fits_vect( ncomp*ncomp*(nlmaxSim+1),  xclSim, cmd, 1);

//     collect_xls( 1, 0, 1, 0, ncomp, nlmaxSim, nmvalsSim, mvalsSim, nlmaxSim,
//                  local_simu_almT, local_simu_almP, ncomp*ncomp, xclSim, gangrank, gangsize, root, gang_comm);
//     if( gangrank == root) sprintf( cmd, "testcl_TempXPol%d.fits", rank);
//     if( gangrank == root) write_fits_vect( ncomp*ncomp*(nlmaxSim+1),  xclSim, cmd, 1);

//     free( xclSim);
// #endif
//     MPI_Barrier( MPI_COMM_WORLD);


//     /* Generate CMB map */
//     if( gangrank == gangroot) printf("\tGenerate input CMB map...");

//     if( gangrank == gangroot) printf("\t(T");
//     s2hat_alm2map( plms, cpixelization, cscan, nlmaxSim, nmmaxSim, nmvalsSim, mvalsSim, ncmbmap, 1, 
// 		   first_ring, last_ring, map_size, local_mapT, lda, local_simu_almT, nplm, NULL, 
// 		   gangsize, gangrank, gang_comm);
// /*     s2hat_alm2map_spin( cpixelization, cscan, 0, nlmaxSim, nmmaxSim, nmvalsSim, mvalsSim, 1, */
// /* 			first_ring, last_ring, map_size, local_mapT, nlmaxSim, local_simu_almT, */
// /* 			gangsize, gangrank, gang_comm); */
//     free( local_simu_almT);
    
//     if( gangrank == gangroot) printf(",P)\n");
//     s2hat_alm2map_spin( cpixelization, cscan, 2, nlmaxSim, nmmaxSim, nmvalsSim, mvalsSim, ncmbmap,
// 			first_ring, last_ring, map_size, local_mapP, nlmaxSim, local_simu_almP,
// 			gangsize, gangrank, gang_comm);
//     free( local_simu_almP);
//     MPI_Barrier( MPI_COMM_WORLD);


//     /* fill CMB maps to all gangnmaps */
//     if( gangrank == gangroot) printf("\tfill all the maps with the same CMB\n");
//     for( imap=1; imap<gangnmaps; imap++) {
//       for( p=0; p<ncomp*map_size; p++) {
//         local_mapT[      map_size*imap+p] = local_mapT[p];
//         local_mapP[ncomp*map_size*imap+p] = local_mapP[p];
//       }
//     }

// #if TESTOUT
//     collect_write_map( cpixelization, 1, 0, 1, "cmbmapT", first_ring, last_ring, map_size,
// 		       &local_mapT[0], gangrank, gangsize, gangroot, gang_comm);
//     collect_write_map( cpixelization, 1, 0, 1, "cmbmapQ", first_ring, last_ring, map_size,
//                        &local_mapP[0], gangrank, gangsize, gangroot, gang_comm);
//     collect_write_map( cpixelization, 1, 0, 1, "cmbmapU", first_ring, last_ring, map_size,
//                        &local_mapP[map_size], gangrank, gangsize, gangroot, gang_comm);
// #endif

//     free( mvalsSim);
//   }
  MPI_Barrier( MPI_COMM_WORLD);


  /* read and distribute CMB map */
  if( cmbFromMap) {
    if( gangrank == gangroot) printf( " - S2HAT : Get and distribute maps\n");
    if( gangrank == gangroot) map = (double *) malloc( 3*npix*sizeof(double));

    for( imap=0; imap<gangnmaps; imap++) {

      /* root reads the map */
      if( rank == root) read_TQU_maps( nside, map, mapfile[imap], 3);

      /* Broadcast to all gangroots */
      if( gangrank == gangroot) MPI_Bcast( map, 3*npix, MPI_DOUBLE, root, root_comm);

      /* distribute inside the gang only T maps (-> map[0,npix-1]) */
      distribute_map( cpixelization, gangnmaps, imap, 1, first_ring, last_ring, map_size,
		      local_mapT, &map[0], gangrank, gangsize, gangroot, gang_comm);

      /* distribute inside the gang only Q, U maps (-> map+npix) */
      distribute_map( cpixelization, gangnmaps, imap, ncomp, first_ring, last_ring, map_size,
		      local_mapP, &map[npix], gangrank, gangsize, gangroot, gang_comm);
   }

    if( gangrank == gangroot) free( map);  
  }
  MPI_Barrier( MPI_COMM_WORLD);
  /***** at this stage each gang has all signal maps *****/


  //-----------------------------------------------------------------------
  // Add white noise
  //-----------------------------------------------------------------------
//   if( is_noise) {

//     for( imap=0; imap<gangnmaps; imap++) {
//       if( gangrank == gangroot) printf( " - Add noise at level = [%4.2e,%4.2e] \n", sigmaT[imap], sigmaP[imap]);
//       local_noise = (double *) calloc( ncomp*map_size, sizeof(double));
      
//       /* Temperature noise */
//       sprng_gaussran( noise_random_streamT, map_size, local_noise);

//       /* generate inhomogeneous noise using nhit map */
//       if( is_nhit) {
// 	for( p=0; p<map_size; p++) {
// 	  if( local_nhitT[p] > 0) {
// 	    local_noise[p] /= sqrt(local_nhitT[p]);
// 	  } else {
// 	    local_noise[p] = 0.0;
// 	  }
// 	}
//       }
	
// #if TESTOUT
//       /*******************/
//       /* check noise map */
//       /*******************/
//       double av, rms;
//       av = rms = 0.0;
//       for( p=0; p<ncomp*map_size; p++) {
// 	av += local_noise[p];
// 	rms += local_noise[p]*local_noise[p];
//       }
//       av /= ncomp*map_size;
//       rms /= ncomp*map_size;
//       printf(" av = %.4e : rms = %.4e \n", av, sqrt( rms-av*av));

//       collect_write_map( cpixelization, 1, 0, ncomp, "noisemap", first_ring, last_ring, map_size,
// 			 local_noise, gangrank, gangsize, gangroot, gang_comm);    
//       /*******************/
// #endif

//       /* add noise with given level */
//       for( p=0; p<ncomp*map_size; p++) local_mapT[map_size*imap+p] += local_noise[p]*sigmaT[imap];


//       /* Polarization noise */
//       sprng_gaussran( noise_random_streamP, ncomp*map_size, local_noise);

//       /* generate inhomogeneous noise using nhit map */
//       if( is_nhit) {
// 	for( p=0; p<map_size; p++) {
// 	  if( local_nhitP[p] > 0) {
// 	    local_noise[p+0*map_size] /= sqrt(local_nhitP[p]);
// 	    local_noise[p+1*map_size] /= sqrt(local_nhitP[p]);
// 	  } else {
// 	    local_noise[p+1*map_size] = local_noise[p+0*map_size] = 0.0;
// 	  }
// 	}
//       }

//       /* add noise with given level */
//       for( p=0; p<ncomp*map_size; p++) local_mapP[ncomp*map_size*imap+p] += local_noise[p]*sigmaP[imap];

//       /* free noise map */
//       free( local_noise);

//     } /* end loop on nmaps */
 
//     if( is_nhit) free( local_nhitT);
//     if( is_nhit) free( local_nhitP);
//   }
  free( sigmaT);
  free( sigmaP );
  // if( cmbFromCl) free_sprng(   cmb_random_streamT);
  // if( cmbFromCl) free_sprng(   cmb_random_streamP);
  // if(  is_noise) free_sprng( noise_random_streamT);
  // if(  is_noise) free_sprng( noise_random_streamP);

#if TESTOUT
  for( imap=0; imap<nmaps; imap++) {
    collect_write_map( cpixelization, gangnmaps, imap, ncomp, "totmap", first_ring, last_ring, map_size,
		       local_map, gangrank, gangsize, gangroot, gang_comm);
  }
#endif
  MPI_Barrier( MPI_COMM_WORLD);



  //-----------------------------------------------------------------------
  // Apply Mask
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( " - S2HAT : Apply Window\n");
  long ishift;
  int winstride = 0; /* 0: one window for all maps */

  /* apply complex windows */
  double *local_apodizedmap_temp;
  double *local_apodizedmap_Espin0, *local_apodizedmap_Espin1, *local_apodizedmap_Espin2;
  double *local_apodizedmap_Bspin0, *local_apodizedmap_Bspin1, *local_apodizedmap_Bspin2;

  local_apodizedmap_temp   = (double *) calloc( ncomp*gangnmaps*map_size, sizeof(double));
  local_apodizedmap_Espin0 = (double *) calloc( ncomp*gangnmaps*map_size, sizeof(double));
  local_apodizedmap_Bspin0 = (double *) calloc( ncomp*gangnmaps*map_size, sizeof(double));
  if( mode > 0) local_apodizedmap_Espin1 = (double *) calloc( gangnmaps*ncomp*map_size, sizeof(double));
  if( mode > 0) local_apodizedmap_Bspin1 = (double *) calloc( gangnmaps*ncomp*map_size, sizeof(double));
  if( mode > 0) local_apodizedmap_Espin2 = (double *) calloc( gangnmaps*ncomp*map_size, sizeof(double));
  if( mode > 0) local_apodizedmap_Bspin2 = (double *) calloc( gangnmaps*ncomp*map_size, sizeof(double));

  apply_maskTT( gangnmaps, map_size, local_mapT, local_mask_T, local_apodizedmap_temp);

  switch( mode) {

  case 0:    
    apply_mask( gangnmaps, map_size, ncomp, local_mapP, local_mask_E_spin0, local_apodizedmap_Espin0);
    apply_mask( gangnmaps, map_size, ncomp, local_mapP, local_mask_B_spin0, local_apodizedmap_Bspin0);
    break;

  case 1:
    apodize_maps_all( gangnmaps, map_size, local_mapP, winstride, local_mask_E_spin0, local_mask_E_spin1, local_mask_E_spin2,
		      local_apodizedmap_Espin0, local_apodizedmap_Espin1, local_apodizedmap_Espin2);
    apodize_maps_all( gangnmaps, map_size, local_mapP, winstride, local_mask_B_spin0, local_mask_B_spin1, local_mask_B_spin2,
                      local_apodizedmap_Bspin0, local_apodizedmap_Bspin1, local_apodizedmap_Bspin2);
    break;

  case 2:
    apply_mask( gangnmaps, map_size, ncomp, local_mapP, local_mask_E_spin0, local_apodizedmap_Espin0);
    apodize_maps_all( gangnmaps, map_size, local_mapP, winstride, local_mask_B_spin0, local_mask_B_spin1, local_mask_B_spin2,
                      local_apodizedmap_Bspin0, local_apodizedmap_Bspin1, local_apodizedmap_Bspin2);
    break;

  }

#if WRITE_MAP
  for( imap=0; imap<1; imap++) {
    collect_write_map( cpixelization, gangnmaps, imap, ncomp, "theapodizedmap", first_ring, last_ring, map_size,
		       local_apodizedmap_spin0, gangrank, gangsize, gangroot, gang_comm);
  }
#endif

  free( local_mask_T);
  free( local_mask_E_spin0);
  free( local_mask_B_spin0);
  if( mode > 0) free( local_mask_E_spin1);
  if( mode > 0) free( local_mask_B_spin1);
  if( mode > 0) free( local_mask_E_spin2);
  if( mode > 0) free( local_mask_B_spin2);
  free( local_mapT);
  free( local_mapP);
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // MAP2ALM transform
  //-----------------------------------------------------------------------
  s2hat_dcomplex *local_Talm;
  s2hat_dcomplex *local_Epurealm, *local_Bpurealm;

  /* compute rings */
  if( gangrank == gangroot) printf( " - S2HAT : compute rings\n");
  long nrings = last_ring-first_ring+1;
  double *local_w8ring;
  local_w8ring = (double *) malloc( nrings*nstokes*sizeof(double));
  for( i=0; i<nrings*nstokes; i++) local_w8ring[i] = 1.;
  MPI_Barrier( MPI_COMM_WORLD);

  local_Talm     = (s2hat_dcomplex *) calloc( gangnmaps*(nlmax+1)*nmvals*nstokes, sizeof( s2hat_dcomplex));
  local_Epurealm = (s2hat_dcomplex *) calloc( gangnmaps*(nlmax+1)*nmvals*nstokes, sizeof( s2hat_dcomplex));
  local_Bpurealm = (s2hat_dcomplex *) calloc( gangnmaps*(nlmax+1)*nmvals*nstokes, sizeof( s2hat_dcomplex));

  if( gangrank == gangroot) printf( " - S2HAT : Temperature : map2alm\n");
/*   s2hat_map2alm( plms, cpixelization, cscan, nlmax, nmmax, nmvals, mvals, gangnmaps, 1, first_ring, last_ring,  */
/* 		 local_w8ring, map_size, local_apodizedmap_temp, lda, local_Talm, nplm, NULL, gangsize, gangrank, gang_comm); */
  s2hat_map2alm_spin( cpixelization, cscan, 0, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
		      local_w8ring, map_size, local_apodizedmap_temp, lda, local_Talm, gangsize, gangrank, gang_comm);

  switch( mode) {

  case 0:
    if( gangrank == gangroot) printf( " - S2HAT : Polarization : Standard Formalism\n");
    s2hat_map2alm_spin( cpixelization, cscan, 2, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
			local_w8ring, map_size, local_apodizedmap_Espin0, lda, local_Epurealm, gangsize, gangrank, gang_comm);
    s2hat_map2alm_spin( cpixelization, cscan, 2, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
                        local_w8ring, map_size, local_apodizedmap_Bspin0, lda, local_Bpurealm, gangsize, gangrank, gang_comm);
    break;
    
  case 1:
    if( gangrank == gangroot) printf( " - S2HAT : Polarization : Pure Formalism\n");
    s2hat_apodizedmaps2purealm( cpixelization, cscan, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring, local_w8ring, map_size,
				local_apodizedmap_Espin0, local_apodizedmap_Espin1, local_apodizedmap_Espin2, lda, local_Epurealm, 
				gangrank, gangsize, gang_comm);
    s2hat_apodizedmaps2purealm( cpixelization, cscan, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring, local_w8ring, map_size,
                                local_apodizedmap_Bspin0, local_apodizedmap_Bspin1, local_apodizedmap_Bspin2, lda, local_Bpurealm,
                                gangrank, gangsize, gang_comm);
    break;

  case 2:
    if( gangrank == gangroot) printf( " - S2HAT : Polarization : Hybrid Formalism\n");
    s2hat_map2alm_spin( cpixelization, cscan, 2, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring,
                        local_w8ring, map_size, local_apodizedmap_Espin0, lda, local_Epurealm, gangsize, gangrank, gang_comm);
    s2hat_apodizedmaps2purealm( cpixelization, cscan, nlmax, nmmax, nmvals, mvals, gangnmaps, first_ring, last_ring, local_w8ring, map_size,
                                local_apodizedmap_Bspin0, local_apodizedmap_Bspin1, local_apodizedmap_Bspin2, lda, local_Bpurealm,
                                gangrank, gangsize, gang_comm);
    break;
    
  }
  
  /* free */
  destroy_pixelization( cpixelization);
  destroy_scan( cscan);
  free( local_w8ring);
  free( local_apodizedmap_temp);
  free( local_apodizedmap_Espin0);
  if( mode > 0) free( local_apodizedmap_Espin1);
  if( mode > 0) free( local_apodizedmap_Espin2);
  free( local_apodizedmap_Bspin0);
  if( mode > 0) free( local_apodizedmap_Bspin1);
  if( mode > 0) free( local_apodizedmap_Bspin2);
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Compute P and Q
  //-----------------------------------------------------------------------
  // double *matp, *matp2, *matq;
  double *matp, *matq;
  if( gangrank == gangroot) printf( " - compute change base matrix (P and Q)...\n");
  int nll = lmax+1;
  int ntt = nbins;
  double true_ell_bin;
  if( gangrank == gangroot) printf( "\t(%d -> %d)\n", nll, ntt);
  matp = (double *) calloc( ntt*nll, sizeof(double));
  // matp2 = (double *) calloc( ntt*nll, sizeof(double));
  matq = (double *) calloc( nll*ntt, sizeof(double));
  for( b=0; b<ntt; b++) {
    for( l=bintab[b]; l<bintab[b+1]; l++) {
      
      // true_ell_bin = (bintab[b]+bintab[b+1]-1)/2.;
      // matp[ b*nll + l] = (double)(l*(double)(l+1.))/(2.*M_PI)/(double)(bintab[b+1]-bintab[b]);
      matp[ b*nll + l] = (double)1./(double)(bintab[b+1]-bintab[b]);
      // matp[ b*nll + l] = (double)(true_ell_bin*(double)(true_ell_bin+1.))/(2.*M_PI)/(double)(bintab[b+1]-bintab[b]);
      
      // matq[ l*ntt + b] = (double)(2*M_PI)/(double)(l*(double)(l+1.));
      // matp2[ b*nll + l] = (double)1./(double)(bintab[b+1]-bintab[b]);
      matq[ l*ntt + b] = (double)1.;
      // matq[ l*ntt + b] = (double)(2*M_PI)/(double)(true_ell_bin*(double)(true_ell_bin+1.));
      
    }
  }
  MPI_Barrier( MPI_COMM_WORLD);

  //  //-----------------------------------------------------------------------
  // // Compute P2 and Q2
  // //-----------------------------------------------------------------------
  // double *matp2;//, *matq2;
  // if( gangrank == gangroot) printf( " - compute change base matrix (P2 and Q2)...\n");
  // int nll2 = nbins;
  // int ntt2 = nbins;
  // if( gangrank == gangroot) printf( "\t(%d -> %d)\n", nll2, ntt2);
  // matp2 = (double *) calloc( ntt2*nll2, sizeof(double));
  // // matq2 = (double *) calloc( nll2*ntt2, sizeof(double));
  // for( b=0; b<ntt2; b++) {
  //   for( l=bintab[b]; l<bintab[b+1]; l++) {
      
  //     matp2[ b*nll2 + l] = (double)(l*(double)(l+1.))/(2.*M_PI)///(double)(bintab[b+1]-bintab[b]);
  //     // matq[ l*ntt2 + b] = (double)(2*M_PI)/(double)(l*(double)(l+1.));
  //     // matp[ b*nll + l] = (double)1./(double)(bintab[b+1]-bintab[b]);
  //     // matq[ l*ntt + b] = (double)1.;
  //   }
  // }
  // MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // Read pixel windows
  //-----------------------------------------------------------------------
  if( gangrank == gangroot) printf( " - read pixel window : ");
  wint = (double *) malloc( (lmax+1)*sizeof(double));
  winp = (double *) malloc( (lmax+1)*sizeof(double));
  sprintf( pixfile, "%s/pixel_window_n%04d.fits", healpixdatadir, nside);
  if( gangrank == gangroot) printf( "%s\n", pixfile);
  read_fits_vect( pixfile, wint, lmax+1, 1, 1);
  read_fits_vect( pixfile, winp, lmax+1, 2, 1);
  wint[0] = wint[1] = 1.; /// ????
  winp[0] = winp[1] = 1.; /// ????
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // get number of cross and spread over gang
  //-----------------------------------------------------------------------
  int ncross = gangnmaps*(gangnmaps+1)/2.;
  int *spec1, *spec2, c=0;
  spec1 = (int *) malloc( ncross*sizeof(int));
  spec2 = (int *) malloc( ncross*sizeof(int));
  for( i=0; i<gangnmaps; i++) for( j=i; j<gangnmaps; j++) {
    spec1[c] = i;
    spec2[c] = j;
    c++;
  }

  int ncross_per_proc = ceil((double)ncross/(double)gangsize);
  int icross1, icross2;
  icross1 = gangrank*ncross_per_proc;
  icross2 = (gangrank+1)*ncross_per_proc;
  if( icross2 > ncross) icross2 = ncross;
  if( icross2 < icross1) { icross1 = 0; icross2 = 0; }
  if( gangnum == 0) {
    printf( "%i - ncross = %d\t(", gangrank, icross2-icross1);
    for( i=icross1; i<icross2; i++) printf( " %dx%d ", spec1[i], spec2[i]);
    printf( ")\n");
  }
  MPI_Barrier( MPI_COMM_WORLD);


  //-----------------------------------------------------------------------
  // De-bias pseudo spectra
  //-----------------------------------------------------------------------
  int imap1, imap2, l1, l2, b1, b2, therank;
  double rcond;
  double *bell1, *bell2;
  double *mllTT_TT, *mbbTT_TT;
  double *mllTE_TE, *mbbTE_TE;
  double *mllTB_TB, *mbbTB_TB;
  double *mllEE_EE, *mllEE_BB, *mbbEE_EE, *mbbEE_BB;
  double *mllBB_BB, *mllBB_EE, *mbbBB_BB, *mbbBB_EE;
  double *mllEB_EB, *mbbEB_EB;
  double *mbb=NULL;
  double *mbb_2=NULL;
  double *mll_2=NULL;
  int icross;
  char myfile[1024], tmp1[1024], tmp2[1024];

//#if WRITE_MBB
//  strncpy( mbbfile1, mllfile1, strlen(mllfile1)-5);
//  strncpy( mbbfile2, mllfile2, strlen(mllfile2)-5);
//  strncpy( mbbfileT, mllfileT, strlen(mllfileT)-5);
//#endif


  /* all spectra spin 2 for combinations */
  int nxtype = nstokes*nstokes; /* compute all cross-spectra */
  double *cell=NULL;
  double *ell, *cell_tt, *cell_ee, *cell_bb, *cell_te, *cell_tb, *cell_eb;
  double *xcl, *pseudocl_tt, *pseudocl_ee, *pseudocl_bb, *pseudocl_te, *pseudocl_tb, *pseudocl_eb;
  xcl = (double *) malloc( (lmax+1)*nxtype*sizeof(double));
  //ell         = (double *) malloc( (lmax+1)*sizeof(double));
  pseudocl_tt = (double *) malloc( (lmax+1)*sizeof(double));
  pseudocl_ee = (double *) malloc( (lmax+1)*sizeof(double));
  pseudocl_bb = (double *) malloc( (lmax+1)*sizeof(double));
  pseudocl_te = (double *) malloc( (lmax+1)*sizeof(double));
  pseudocl_tb = (double *) malloc( (lmax+1)*sizeof(double));
  pseudocl_eb = (double *) malloc( (lmax+1)*sizeof(double));

  ell     = (double *) malloc( nbins*sizeof(double));
  cell_tt = (double *) malloc( nbins*sizeof(double));
  cell_ee = (double *) malloc( nbins*sizeof(double));
  cell_bb = (double *) malloc( nbins*sizeof(double));
  cell_te = (double *) malloc( nbins*sizeof(double));
  cell_tb = (double *) malloc( nbins*sizeof(double));
  cell_eb = (double *) malloc( nbins*sizeof(double));

  //loop over the xspectra
  for( icross=0; icross<ncross; icross++) {
    imap1 = spec1[icross];
    imap2 = spec2[icross];

    /* set the proc that will treat this cross */
    if( combine_masks) therank = gangroot;
    else               therank = icross/ncross_per_proc;

    /* get pseudo */
    if( gangrank == gangroot) printf( "%i\t\tGet pseudo %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);

    if( gangrank == gangroot) printf( "%i\t\t\t\tTT spectra %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);
    collect_xls( gangnmaps, imap1, gangnmaps, imap2, nstokes, nlmax, nmvals, mvals, lda, 
		 local_Talm, local_Talm, nxtype, xcl, gangrank, gangsize, therank, gang_comm);
    if( gangrank == therank) for( l1=0; l1<=lmax; l1++) pseudocl_tt[l1] = xcl[l1];

    if( gangrank == gangroot) printf( "%i\t\t\t\tEE spectra %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);
    collect_xls( gangnmaps, imap1, gangnmaps, imap2, nstokes, nlmax, nmvals, mvals, lda,
                 local_Epurealm, local_Epurealm, nxtype, xcl, gangrank, gangsize, therank, gang_comm);
    if( gangrank == therank) for( l1=0; l1<=lmax; l1++) pseudocl_ee[l1] = xcl[l1];

    if( gangrank == gangroot) printf( "%i\t\t\t\tBB spectra %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);
    collect_xls( gangnmaps, imap1, gangnmaps, imap2, nstokes, nlmax, nmvals, mvals, lda,
                 local_Bpurealm, local_Bpurealm, nxtype, xcl, gangrank, gangsize, therank, gang_comm);
    if( gangrank == therank) for( l1=0; l1<=lmax; l1++) pseudocl_bb[l1] = xcl[(lmax+1)+l1];

    if( gangrank == gangroot) printf( "%i\t\t\t\tTE spectra %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);
    collect_xls( gangnmaps, imap1, gangnmaps, imap2, nstokes, nlmax, nmvals, mvals, lda,
                 local_Talm, local_Epurealm, nxtype, xcl, gangrank, gangsize, therank, gang_comm);
    if( gangrank == therank) for( l1=0; l1<lmax+1; l1++) pseudocl_te[l1] = -1.*xcl[l1];
/*       for( l1=0; l1<=lmax; l1++) pseudocl_te[l1] = xcl[l1]; */

    if( gangrank == gangroot) printf( "%i\t\t\t\tTB spectra %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);
    collect_xls( gangnmaps, imap1, gangnmaps, imap2, nstokes, nlmax, nmvals, mvals, lda,
                 local_Talm, local_Bpurealm, nxtype, xcl, gangrank, gangsize, therank, gang_comm);
    if( gangrank == therank) for( l1=0; l1<lmax+1; l1++) pseudocl_tb[l1] = -1.*xcl[2*(lmax+1)+l1];
/*       for( l1=0; l1<=lmax; l1++) pseudocl_tb[l1] = xcl[2*(lmax+1)+l1]; */

    if( gangrank == gangroot) printf( "%i\t\t\t\tEB spectra %ix%i (%i)...   \n", gangnum, imap1, imap2, therank);
    collect_xls( gangnmaps, imap1, gangnmaps, imap2, nstokes, nlmax, nmvals, mvals, lda,
                 local_Epurealm, local_Bpurealm, nxtype, xcl, gangrank, gangsize, therank, gang_comm);
    if( gangrank == therank) for( l1=0; l1<=lmax; l1++) pseudocl_eb[l1] = xcl[2*(lmax+1)+l1];

    if( gangrank == therank) {
      
      /* ------------------------------------------------------------------------------------------------- */
      /* Write Pseudo Spectra */
      if( write_pseudo) {
        printf( "%i-%i\t\tWrite pseudos...\t", gangnum, gangrank);
        sprintf( myfile, "%s_mask%d_%d_%d.fits", pseudofile, gangnum+1, imap1, imap2);
        write_fits_pseudo( lmax+1, pseudocl_tt, myfile, 1);
        write_fits_pseudo( lmax+1, pseudocl_ee, myfile, 2);
        write_fits_pseudo( lmax+1, pseudocl_bb, myfile, 3);
        write_fits_pseudo( lmax+1, pseudocl_te, myfile, 4);
        write_fits_pseudo( lmax+1, pseudocl_tb, myfile, 5);
        write_fits_pseudo( lmax+1, pseudocl_eb, myfile, 6);
        printf( "%s\n", myfile);
      }
      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* Remove Noise Bias for auto-spectra */
      if( imap1 == imap2) {
        printf( "%i-%i\t\tRemove Noise Bias...   \n", gangnum, gangrank);
        printf( "%d \t %d\t\t Noise Bias...   \n", biasT[imap1], biasP[imap1]);
        double fskytot_E, fskytot_B;
          
        pseudocl_tt[0] -= 4.*M_PI*biasT[imap1]*biasT[imap2]/(double)(npix*npix)*fskyT;
        pseudocl_ee[0] -= 4.*M_PI*biasP[imap1]*biasP[imap2]/(double)(npix*npix)*fskyspin0_E;
        pseudocl_bb[0] -= 4.*M_PI*biasP[imap1]*biasP[imap2]/(double)(npix*npix)*fskyspin0_B;

        pseudocl_tt[1] -= 4.*M_PI*biasT[imap1]*biasT[imap2]/(double)(npix*npix)*fskyT;
        pseudocl_ee[1] -= 4.*M_PI*biasP[imap1]*biasP[imap2]/(double)(npix*npix)*fskyspin0_E;
        pseudocl_bb[1] -= 4.*M_PI*biasP[imap1]*biasP[imap2]/(double)(npix*npix)*fskyspin0_B;

        for( l=2; l<=lmax; l++) {
          fskytot_E = fskyspin0_E;
          fskytot_B = fskyspin0_B;
          if( mode == 1) {
            fskytot_E += 4.*fskyspin1_E/((double)(l)-1.)/((double)(l)+2.) + fskyspin2_E/((double)(l)-1.)/((double)(l)+2.)/((double)(l))/((double)(l)+1.);
            fskytot_B += 4.*fskyspin1_B/((double)(l)-1.)/((double)(l)+2.) + fskyspin2_B/((double)(l)-1.)/((double)(l)+2.)/((double)(l))/((double)(l)+1.);
          }
          if( mode == 2) {
            fskytot_B += 4.*fskyspin1_B/((double)(l)-1.)/((double)(l)+2.) + fskyspin2_B/((double)(l)-1.)/((double)(l)+2.)/((double)(l))/((double)(l)+1.);
          }

          pseudocl_tt[l] -= 4.*M_PI*biasT[imap1]*biasT[imap2]/(double)(npix*npix)*fskyT;
          pseudocl_ee[l] -= 4.*M_PI*biasP[imap1]*biasP[imap2]/(double)(npix*npix)*fskytot_E;
          pseudocl_bb[l] -= 4.*M_PI*biasP[imap1]*biasP[imap2]/(double)(npix*npix)*fskytot_B;

          //pseudocl_ee[l] -= 4.*M_PI*biasP[imap1][imap2]/(double)(npix*npix)*(fskyspin0 + 4.*fskyspin1/((double)(l)-1.)/((double)(l)+2.) + fskyspin2/((double)(l)-1.)/((double)(l)+2.)/((double)(l))/((double)(l)+1.));
          //pseudocl_bb[l] -= 4.*M_PI*biasP[imap1][imap2]/(double)(npix*npix)*(fskyspin0 + 4.*fskyspin1/((double)(l)-1.)/((double)(l)+2.) + fskyspin2/((double)(l)-1.)/((double)(l)+2.)/((double)(l))/((double)(l)+1.));
        }

      }
      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* bin pseudos */
      printf( "%i-%i\t\tBin pseudo...   \n", gangnum, gangrank); fflush(stdout);
      PCl( lmax+1, nbins, matp, pseudocl_tt, cell_tt);
      PCl( lmax+1, nbins, matp, pseudocl_ee, cell_ee);
      PCl( lmax+1, nbins, matp, pseudocl_bb, cell_bb);
      PCl( lmax+1, nbins, matp, pseudocl_te, cell_te);
      PCl( lmax+1, nbins, matp, pseudocl_tb, cell_tb);
      PCl( lmax+1, nbins, matp, pseudocl_eb, cell_eb);

///////// Other attempt to bin pseudos, with bin_space constant
      // int bin_space = bintab[1]-bintab[0];
      // printf( "%d\t\t Bin space, for pseudo-spectra...\n", bin_space); fflush(stdout);
      // bin_pseudo_spectra_not_weighted(nbins, bin_space, pseudocl_tt, cell_tt);
      // bin_pseudo_spectra_not_weighted(nbins, bin_space, pseudocl_ee, cell_ee);
      // bin_pseudo_spectra_not_weighted(nbins, bin_space, pseudocl_bb, cell_bb);
      // bin_pseudo_spectra_not_weighted(nbins, bin_space, pseudocl_te, cell_te);
      // bin_pseudo_spectra_not_weighted(nbins, bin_space, pseudocl_tb, cell_tb);
      // bin_pseudo_spectra_not_weighted(nbins, bin_space, pseudocl_eb, cell_eb);


      // printf( "%i-%i\t\t D_ell pseudo binned ...   \n", gangnum, gangrank); fflush(stdout);
      // PCl( nbins, nbins, matp2, pseudocl_tt, cell_tt);
      // PCl( nbins, nbins, matp2, pseudocl_ee, cell_ee);
      // PCl( nbins, nbins, matp2, pseudocl_bb, cell_bb);
      // PCl( nbins, nbins, matp2, pseudocl_te, cell_te);
      // PCl( nbins, nbins, matp2, pseudocl_tb, cell_tb);
      // PCl( nbins, nbins, matp2, pseudocl_eb, cell_eb);


      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* read Mll                                                                                          */
      /* suppose that TB = EB = 0 so that mllfileTB_TE, mllfileBB_EB, mllfileEB_EE, mllfileEB_BB not read  */
      printf( "%i-%i\t\tRead Mll for mask %d...\n", gangnum, gangrank, gangnum); fflush(stdout);
      mllTT_TT = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));

      mllEE_EE = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));
      mllEE_BB = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));  
      mllBB_BB = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));
      mllBB_EE = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));

      mllTE_TE = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));
      mllTB_TB = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));
      mllEB_EB = (double *) malloc( (lmax+1)*(lmax+1)*sizeof(double));
      
      read_fits_puremll( mllfileTT_TT, lmax+1, mllTT_TT);

      read_fits_puremll( mllfileEE_EE, lmax+1, mllEE_EE);
      read_fits_puremll( mllfileEE_BB, lmax+1, mllEE_BB);
      read_fits_puremll( mllfileBB_BB, lmax+1, mllBB_BB);
      read_fits_puremll( mllfileBB_EE, lmax+1, mllBB_EE);

      read_fits_puremll( mllfileTE_TE, lmax+1, mllTE_TE);
      read_fits_puremll( mllfileTB_TB, lmax+1, mllTB_TB);
      read_fits_puremll( mllfileEB_EB, lmax+1, mllEB_EB);
/*     write_fits_vect( (lmax+1)*(lmax+1), mll1, "mll1.fits", 1); */
/*     write_fits_vect( (lmax+1)*(lmax+1), mll2, "mll2.fits", 1); */
      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* read bell */
/*       if( !strcmp(beamfile[imap1], "") && !strcmp(beamfile[imap2], "") ) */
      printf( "%i-%i\t\tRead bell...\n", gangnum, gangrank);
      bell1 = (double *) malloc( (lmax+1)*sizeof(double));
      bell2 = (double *) malloc( (lmax+1)*sizeof(double));
      read_fits_vect( beamfile[imap1], bell1, lmax+1, 1, 1);
      read_fits_vect( beamfile[imap2], bell2, lmax+1, 1, 1);
      
      /* apply bell */
      printf( "%i-%i\t\tApply TF to Mll...\n", gangnum, gangrank); fflush(stdout);
      for( l1=0; l1<=lmax; l1++) {
	      for( l2=0; l2<=lmax; l2++) {
            mllTT_TT[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*wint[l2]*wint[l2];
            
            mllEE_EE[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*winp[l2];
            mllEE_BB[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*winp[l2];
            mllBB_BB[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*winp[l2];
            mllBB_EE[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*winp[l2];

            mllTE_TE[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*wint[l2];
            mllTB_TB[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*wint[l2];
            mllEB_EB[l1*(lmax+1)+l2] *= bell1[l2]*bell2[l2]*winp[l2]*winp[l2];
          }
      }
      free( bell1);
      free( bell2);
      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* bin Mll */
      printf( "%i-%i\t\tBin Mll...\n", gangnum, gangrank); fflush(stdout);
      mbbTT_TT = (double *) malloc( nbins*nbins * sizeof(double));
      mbbEE_EE = (double *) malloc( nbins*nbins * sizeof(double));
      mbbEE_BB = (double *) malloc( nbins*nbins * sizeof(double));
      mbbBB_BB = (double *) malloc( nbins*nbins * sizeof(double));
      mbbBB_EE = (double *) malloc( nbins*nbins * sizeof(double));
      mbbTE_TE = (double *) malloc( nbins*nbins * sizeof(double));
      mbbTB_TB = (double *) malloc( nbins*nbins * sizeof(double));
      mbbEB_EB = (double *) malloc( nbins*nbins * sizeof(double));

      PMQ( nbins, lmax+1, matp, mllTT_TT, matq, mbbTT_TT);
      PMQ( nbins, lmax+1, matp, mllEE_EE, matq, mbbEE_EE);
      PMQ( nbins, lmax+1, matp, mllEE_BB, matq, mbbEE_BB);
      PMQ( nbins, lmax+1, matp, mllBB_EE, matq, mbbBB_EE);
      PMQ( nbins, lmax+1, matp, mllBB_BB, matq, mbbBB_BB);
      PMQ( nbins, lmax+1, matp, mllTE_TE, matq, mbbTE_TE);
      PMQ( nbins, lmax+1, matp, mllTB_TB, matq, mbbTB_TB);
      PMQ( nbins, lmax+1, matp, mllEB_EB, matq, mbbEB_EB);
////////////////////
      // mll_2 = (double *) malloc( (2*(lmax+1))*(2*(lmax+1))*sizeof( double));
      // for( l1=0; l1<lmax+1; l1++) {
      //   for( l2=0; l2<lmax+1; l2++) {
      //     /*old version*/
      //     mll_2[(l1      )*(2*(lmax+1)) + (l2      )] = mllEE_EE[l1*(lmax+1)+l2];
      //     mll_2[(l1      )*(2*(lmax+1)) + (l2+(lmax+1))] = mllEE_BB[l1*(lmax+1)+l2];
      //     mll_2[(l1+(lmax+1))*(2*(lmax+1)) + (l2      )] = mllBB_EE[l1*(lmax+1)+l2];
      //     mll_2[(l1+(lmax+1))*(2*(lmax+1)) + (l2+(lmax+1))] = mllBB_BB[l1*(lmax+1)+l2];

      //     /*new version (off diagonal blocks are interchanged) WARNING FALSE */
      //     //mll[(l1      )*(2*nbins) + (l2      )] = mllEE_EE[l1*nbins+l2];
      //     //mll[(l1      )*(2*nbins) + (l2+nbins)] = mllBB_EE[l1*nbins+l2];
      //     //mll[(l1+nbins)*(2*nbins) + (l2      )] = mllEE_BB[l1*nbins+l2];
      //     //mll[(l1+nbins)*(2*nbins) + (l2+nbins)] = mllBB_BB[l1*nbins+l2];
      //   }
      // }
      // write_fits_vect( (2*(lmax+1))*(2*(lmax+1)), mll_2, "/global/homes/m/mag/xPure_data/Coupling_matrix_xPure/mllfile_fullmll_v0_version.fits", 1);
      // free(mll_2);



      // mbb_2 = (double *) malloc( (2*nbins)*(2*nbins)*sizeof( double));
      // for( l1=0; l1<nbins; l1++) {
      //   for( l2=0; l2<nbins; l2++) {
      //     /*old version*/
      //     mbb_2[(l1      )*(2*nbins) + (l2      )] = mbbEE_EE[l1*nbins+l2];
      //     mbb_2[(l1      )*(2*nbins) + (l2+nbins)] = mbbEE_BB[l1*nbins+l2];
      //     mbb_2[(l1+nbins)*(2*nbins) + (l2      )] = mbbBB_EE[l1*nbins+l2];
      //     mbb_2[(l1+nbins)*(2*nbins) + (l2+nbins)] = mbbBB_BB[l1*nbins+l2];

      //     /*new version (off diagonal blocks are interchanged) WARNING FALSE */
      //     //mbb[(l1      )*(2*nbins) + (l2      )] = mbbEE_EE[l1*nbins+l2];
      //     //mbb[(l1      )*(2*nbins) + (l2+nbins)] = mbbBB_EE[l1*nbins+l2];
      //     //mbb[(l1+nbins)*(2*nbins) + (l2      )] = mbbEE_BB[l1*nbins+l2];
      //     //mbb[(l1+nbins)*(2*nbins) + (l2+nbins)] = mbbBB_BB[l1*nbins+l2];
      //   }
      // }
      // write_fits_vect( (2*nbins)*(2*nbins), mbb_2, "/global/homes/m/mag/xPure_data/Coupling_matrix_xPure/mllfile_fullmbb_v0_version.fits", 1);
      // free(mbb_2);

      // mbbTT_TT = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbEE_EE = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbEE_BB = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbBB_BB = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbBB_EE = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbTE_TE = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbTB_TB = (double *) calloc( nbins*nbins, sizeof(double));
      // mbbEB_EB = (double *) calloc( nbins*nbins, sizeof(double));


      // // int bin_space = bintab[1]-bintab[0];
      // printf( "%d\t\t Bin space, for coupling matrices...\n", bin_space); fflush(stdout);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllTT_TT, mbbTT_TT);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllEE_EE, mbbEE_EE);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllEE_BB, mbbEE_BB);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllBB_EE, mbbBB_EE);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllBB_BB, mbbBB_BB);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllTE_TE, mbbTE_TE);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllTB_TB, mbbTB_TB);
      // bin_coupling_matrices_not_weighted(nbins, lmax+1, bin_space, mllEB_EB, mbbEB_EB);


      free( mllTT_TT);
      free( mllEE_EE);
      free( mllEE_BB);
      free( mllBB_EE);
      free( mllBB_BB);
      free( mllTE_TE);
      free( mllTB_TB);
      free( mllEB_EB);

/* #if WRITE_MBB */
/* 	char tmpTTTT[1024], tmpEEEE[1024], tmpBBEE[1024], tmpBBBB[1024], tmpTETE[1024], tmpTBTB[1024], tmpEBEB[1024]; */
/* 	sprintf( tmpTTTT, "mask%d_%s_bin_%d_%d.fits", gangnum, "TT_TT", imap1, imap2); */
/* 	sprintf( tmpEEEE, "mask%d_%s_bin_%d_%d.fits", gangnum, "EE_EE", imap1, imap2); */
/* 	sprintf( tmpBBEE, "mask%d_%s_bin_%d_%d.fits", gangnum, "BB_EE", imap1, imap2); */
/* 	sprintf( tmpBBBB, "mask%d_%s_bin_%d_%d.fits", gangnum, "BB_BB", imap1, imap2); */
/* 	sprintf( tmpTETE, "mask%d_%s_bin_%d_%d.fits", gangnum, "TE_TE", imap1, imap2); */
/* 	sprintf( tmpTBTB, "mask%d_%s_bin_%d_%d.fits", gangnum, "TB_TB", imap1, imap2); */
/* 	sprintf( tmpEBEB, "mask%d_%s_bin_%d_%d.fits", gangnum, "EB_EB", imap1, imap2); */

/* 	write_fits_vect( nbins*nbins, mbbTT_TT, tmpTTTT, 1); */
/* 	write_fits_vect( nbins*nbins, mbbEE_EE, tmpEEEE, 1); */
/* 	write_fits_vect( nbins*nbins, mbbBB_EE, tmpBBEE, 1); */
/* 	write_fits_vect( nbins*nbins, mbbBB_BB, tmpBBBB, 1); */
/* 	write_fits_vect( nbins*nbins, mbbTE_TE, tmpTETE, 1); */
/* 	write_fits_vect( nbins*nbins, mbbTB_TB, tmpTBTB, 1); */
/* 	write_fits_vect( nbins*nbins, mbbEB_EB, tmpEBEB, 1); */
/* #endif */
      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* if combine_masks, reduce pseudo and kernel for the                                                */
      /* cross (imap1,imap2) on Master root before inverting                                               */
      int npm;
      if( combine_masks) {
        printf( "%i-%i\t\tCombining masks ...\n", gangnum, gangrank); fflush(stdout);
	bshift = binshift[gangnum]; /* first bin to send for the given gang */
	npm = nbin_per_mask[gangnum];
	double *final_cell_tt, *final_cell_ee, *final_cell_bb, *final_cell_te, *final_cell_tb, *final_cell_eb;
	double *final_mbbTT_TT, *final_mbbEE_EE, *final_mbbEE_BB, *final_mbbBB_BB, *final_mbbBB_EE, *final_mbbTE_TE, *final_mbbTB_TB, *final_mbbEB_EB;

	/* gather binned cells */
	GatherCell( nbins, bshift, npm, cell_tt, root_comm);
	GatherCell( nbins, bshift, npm, cell_ee, root_comm);
	GatherCell( nbins, bshift, npm, cell_bb, root_comm);
	GatherCell( nbins, bshift, npm, cell_te, root_comm);
	GatherCell( nbins, bshift, npm, cell_tb, root_comm);
	GatherCell( nbins, bshift, npm, cell_eb, root_comm);
	
	/* gather mbb */
	GatherMll( nbins, bshift, npm, mbbTT_TT, root_comm);
	GatherMll( nbins, bshift, npm, mbbEE_EE, root_comm);
	GatherMll( nbins, bshift, npm, mbbEE_BB, root_comm);
	GatherMll( nbins, bshift, npm, mbbBB_BB, root_comm);
	GatherMll( nbins, bshift, npm, mbbBB_EE, root_comm);
	GatherMll( nbins, bshift, npm, mbbTE_TE, root_comm);
	GatherMll( nbins, bshift, npm, mbbTB_TB, root_comm);
	GatherMll( nbins, bshift, npm, mbbEB_EB, root_comm);
      }

#if WRITE_MBB
      if( rank == root) {
	char tmpTTTT[1024], tmpEEEE[1024], tmpBBEE[1024], tmpEEBB[1024], tmpBBBB[1024], tmpTETE[1024], tmpTBTB[1024], tmpEBEB[1024];
	sprintf( tmpTTTT, "%s_bin_%d_%d.fits", "mbb_TT_TT", imap1, imap2);
	sprintf( tmpEEEE, "%s_bin_%d_%d.fits", "mbb_EE_EE", imap1, imap2);
	sprintf( tmpEEBB, "%s_bin_%d_%d.fits", "mbb_EE_BB", imap1, imap2);
	sprintf( tmpBBEE, "%s_bin_%d_%d.fits", "mbb_BB_EE", imap1, imap2);
	sprintf( tmpBBBB, "%s_bin_%d_%d.fits", "mbb_BB_BB", imap1, imap2);
	sprintf( tmpTETE, "%s_bin_%d_%d.fits", "mbb_TE_TE", imap1, imap2);
	sprintf( tmpTBTB, "%s_bin_%d_%d.fits", "mbb_TB_TB", imap1, imap2);
	sprintf( tmpEBEB, "%s_bin_%d_%d.fits", "mbb_EB_EB", imap1, imap2);
	
	write_fits_vect( nbins*nbins, mbbTT_TT, tmpTTTT, 1);
	write_fits_vect( nbins*nbins, mbbEE_EE, tmpEEEE, 1);
	write_fits_vect( nbins*nbins, mbbEE_BB, tmpEEBB, 1);
	write_fits_vect( nbins*nbins, mbbBB_EE, tmpBBEE, 1);
	write_fits_vect( nbins*nbins, mbbBB_BB, tmpBBBB, 1);
	write_fits_vect( nbins*nbins, mbbTE_TE, tmpTETE, 1);
	write_fits_vect( nbins*nbins, mbbTB_TB, tmpTBTB, 1);
	write_fits_vect( nbins*nbins, mbbEB_EB, tmpEBEB, 1);
      }
#endif
      /* ------------------------------------------------------------------------------------------------- */


      /* ------------------------------------------------------------------------------------------------- */
      /* correct pseudo                                                                                    */
      if( (combine_masks && rank == root) || (!combine_masks && gangrank == therank) ) {
        printf( "%i-%i\t\tSolve for Temperature\t", gangnum, gangrank);
	rcond = SolveSystem( nbins, mbbTT_TT, cell_tt);
	if( rcond != -1) printf( "\tcn = %e\n", rcond); else printf( "\n");
	
	printf( "%i-%i\t\tSolve for Polarization\t", gangnum, gangrank);
	cell = (double *) malloc( 2*nbins * sizeof(double));
	mbb  = (double *) malloc( (2*nbins)*(2*nbins)*sizeof( double));
	for( l1=0; l1<nbins; l1++) {
	  cell[nbins*0+l1] = cell_ee[l1];
	  cell[nbins*1+l1] = cell_bb[l1];
	  for( l2=0; l2<nbins; l2++) {
	    /*old version*/
	    mbb[(l1      )*(2*nbins) + (l2      )] = mbbEE_EE[l1*nbins+l2];
	    mbb[(l1      )*(2*nbins) + (l2+nbins)] = mbbEE_BB[l1*nbins+l2];
	    mbb[(l1+nbins)*(2*nbins) + (l2      )] = mbbBB_EE[l1*nbins+l2];
	    mbb[(l1+nbins)*(2*nbins) + (l2+nbins)] = mbbBB_BB[l1*nbins+l2];

	    /*new version (off diagonal blocks are interchanged) WARNING FALSE */
	    // mbb[(l1      )*(2*nbins) + (l2      )] = mbbEE_EE[l1*nbins+l2];
	    // mbb[(l1      )*(2*nbins) + (l2+nbins)] = mbbBB_EE[l1*nbins+l2];
	    // mbb[(l1+nbins)*(2*nbins) + (l2      )] = mbbEE_BB[l1*nbins+l2];
	    // mbb[(l1+nbins)*(2*nbins) + (l2+nbins)] = mbbBB_BB[l1*nbins+l2];
	  }
	}
  // for( l1=0; l1<nbins; l1++) {
	//   cell[nbins*0+l1] = cell_ee[l1];
	//   cell[nbins*1+l1] = cell_bb[l1];
	//   for( l2=0; l2<nbins; l2++) {
	//     /*old version*/
	//     // mbb[(l1      )*(2*nbins) + (l2      )] = mbbEE_EE[l1*nbins+l2];
	//     // mbb[(l1      )*(2*nbins) + (l2+nbins)] = 0;//mbbEE_BB[l1*nbins+l2];
	//     // mbb[(l1+nbins)*(2*nbins) + (l2      )] = 0;//mbbBB_EE[l1*nbins+l2];
	//     // mbb[(l1+nbins)*(2*nbins) + (l2+nbins)] = mbbBB_BB[l1*nbins+l2];

	//     /*new version (off diagonal blocks are interchanged) WARNING FALSE */
	//     mbb[(l1      )*(2*nbins) + (l2      )] = mbbEE_EE[l1*nbins+l2];
	//     mbb[(l1      )*(2*nbins) + (l2+nbins)] = 0;//mbbBB_EE[l1*nbins+l2];
	//     mbb[(l1+nbins)*(2*nbins) + (l2      )] = 0;//mbbEE_BB[l1*nbins+l2];
	//     mbb[(l1+nbins)*(2*nbins) + (l2+nbins)] = mbbBB_BB[l1*nbins+l2];
	//   }
	// }
  
 	// write_fits_vect( (2*nbins)*(2*nbins), mbb, "/global/homes/m/mag/xPure_data/Coupling_matrix_xPure/mllfile_fullmbb_version.fits", 1);
 	// write_fits_vect( (2*nbins), cell, "/global/homes/m/mag/xPure_data/Power_spectrum_xPure/cellEE-BB.fits", 1);
	write_fits_vect( (2*nbins)*(2*nbins), mbb, "/global/cscratch1/sd/mag/xPure_data/Coupling_matrix_xPure/mllfile_fullmbb_version.fits", 1);
 	write_fits_vect( (2*nbins), cell, "/global/cscratch1/sd/mag/xPure_data/Power_spectrum_xPure/cellEE-BB.fits", 1);
	rcond = SolveSystem( 2*nbins, mbb, cell);

	for( l1=0; l1<nbins; l1++) {
	  cell_ee[l1] = cell[nbins*0+l1];
	  cell_bb[l1] = cell[nbins*1+l1];
	}
	if( rcond != -1) printf( "\tcn = %e\n", rcond); else printf( "\n");
	free( mbb);
	free( cell);

	printf( "%i-%i\t\tSolve for TE cross-spectrum\t", gangnum, gangrank);
        rcond = SolveSystem( nbins, mbbTE_TE, cell_te);
	if( rcond != -1) printf( "\tcn = %e\n", rcond); else printf( "\n");

	printf( "%i-%i\t\tSolve for TB cross-spectrum\t", gangnum, gangrank);
        rcond = SolveSystem( nbins, mbbTB_TB, cell_tb);
	if( rcond != -1) printf( "\tcn = %e\n", rcond); else printf( "\n");

	printf( "%i-%i\t\tSolve for EB cross-spectrum\t", gangnum, gangrank);
        rcond = SolveSystem( nbins, mbbEB_EB, cell_eb);
	if( rcond != -1) printf( "\tcn = %e\n", rcond); else printf( "\n");


#if WRITE_FULLMBB == 1
	if( combine_masks && rank == root ) {
	  sprintf( myfile, "mll_spin_full_bin_%d_%d.fits", imap1, imap2);
	  write_fits_vect( 2*nbins*2*nbins, mll, myfile, 1); 
	}
	if( !combine_masks && gangrank == therank) {
	  sprintf( myfile, "mll_spin_full_bin_mask%d_%d_%d.fits", gangnum, imap1, imap2);
	  write_fits_vect( (2*nbins)*(2*nbins), mll, myfile, 1); 
	}
#endif

	/* write cell */
	printf( "\n%i-%i\t\tWrite cls...\t", gangnum, gangrank);
	if( combine_masks) sprintf( myfile, "%s_%d_%d.fits", cellfile, imap1, imap2);
	else sprintf( myfile, "%s_mask%d_%d_%d.fits", cellfile, gangnum+1, imap1, imap2);
	for( b=0; b<nbins; b++) ell[b] = (bintab[b]+bintab[b+1]-1)/2.;
	write_fits_pure( nbins, ell, myfile, 1);
	write_fits_pure( nbins, cell_tt, myfile, 2);
	write_fits_pure( nbins, cell_ee, myfile, 3);
	write_fits_pure( nbins, cell_bb, myfile, 4);
  write_fits_pure( nbins, cell_te, myfile, 5);
  write_fits_pure( nbins, cell_tb, myfile, 6);
  write_fits_pure( nbins, cell_eb, myfile, 7);
  printf( "%s\n", myfile);

	/* free */
	free( mbbTT_TT);
	free( mbbEE_EE);
	free( mbbEE_BB);
	free( mbbBB_BB);
	free( mbbBB_EE);
	free( mbbTE_TE);
	free( mbbTB_TB);
	free( mbbEB_EB);
      }
      /* ------------------------------------------------------------------------------------------------- */

    } /* therank */

    fflush( stdout);
    MPI_Barrier( MPI_COMM_WORLD);
  } /* end loop icross */


  //-----------------------------------------------------------------------
  // Free
  //-----------------------------------------------------------------------
  if( rank == root) printf( " - Free...\n");
  free( wint);
  free( winp);
  free( matp);
  free( matq);
  // free(matp2);

  free( spec1);
  free( spec2);
  free( cell_tt);
  free( cell_ee);
  free( cell_bb);
  free( cell_te);
  free( cell_tb);
  free( cell_eb);

  free( bintot);

  free( ell);
  free( pseudocl_tt);
  free( pseudocl_ee);
  free( pseudocl_bb);
  free( pseudocl_te);
  free( pseudocl_tb);
  free( pseudocl_eb);

  free( xcl);
  free( local_Talm    );
  free( local_Epurealm);
  free( local_Bpurealm);
  free( mvals);

  if( combine_masks) free( binshift);
  if( combine_masks) free( nbin_per_mask);

  free( biasT);
  free( biasP);

  for( i=0; i<nmaps; i++) {
    free(  mapfile[i]);
    free( beamfile[i]);
  }
  free(  mapfile);
  free( beamfile);

  free( nhitfileT);
  free( nhitfileP);

  free( maskfile_T);
  free( maskfile_E_spin0);
  free( maskfile_E_spin1);
  free( maskfile_E_spin2);
  free( maskfile_B_spin0);
  free( maskfile_B_spin1);
  free( maskfile_B_spin2);
  free( mllfileTT_TT);
  free( mllfileEE_EE);
  free( mllfileEE_BB);
  free( mllfileBB_BB);
  free( mllfileTE_TE);
  free( mllfileTB_TB);
  free( mllfileEB_EB);

  MPI_Finalize();

  return( 0);
}





void PrintHelp()
{
  printf( "Parameters read by Xpure :\n");
  printf( "\n");

  printf( "nside       (integer)  nside parameter for the maps\n");
  printf( "nmaps       (integer)  number of maps\n");
  printf( "nmasks      (integer)  number of masks files\n");
  printf( "\n");
  printf( "For each map N :\n");
  printf( "\tmapfileN          (string)  input file containing the Healpix map N\n");
  printf( "\tmaskfileN_spin0   (string)  input file containing the weight mask (Healpix map) N\n");
  printf( "\tmaskfileN_spin1   (string)  input file containing the weight mask (Healpix map) N\n");
  printf( "\tmaskfileN_spin2   (string)  input file containing the weight mask (Healpix map) N\n");
  printf( "\tbellfileN    (string)  input file containing the beam transfert fonction N\n");
  printf( "\n");
  printf( "bintab       (string)  input file containing the bintab vector for temperature\n");
  printf( "cellfile     (string)  output file name\n");
  printf( "pseudofile   (string)  output pseudo file name\n");
  printf( "lmax        (integer)  maximum order of l\n");

  fflush( stdout);
  
}









void write_fits_pseudo( int nele, double *vect, char *outfile, int col)
{
  fitsfile *fptr;

  int status = 0;
  char newfilename[1000];
  if( col == 1) sprintf(newfilename,"!%s",outfile);
  else sprintf(newfilename,"%s",outfile);

  char * coltype[6] ={"TT","EE","BB","TE","TB","EB"};
  char * colform[6] ={"1D","1D","1D","1D","1D","1D"};
  
  if( col == 1) ffinit( &fptr, newfilename, &status);
  else ffopen( &fptr, newfilename, 1, &status);

  if( col == 1) fits_create_tbl( fptr, BINARY_TBL, nele, 0, NULL, NULL, NULL, NULL, &status);

  fits_movabs_hdu( fptr, 2, NULL, &status);

  if( col == 1) fits_insert_cols( fptr, 1, 6, coltype, colform, &status);

  fits_write_col( fptr, TDOUBLE, col, 1, 1, nele, vect, &status);

  ffclos(fptr,&status);

}



void write_fits_pure( int nele, double *vect, char *outfile, int col)
{
  fitsfile *fptr;

  int status = 0;
  char newfilename[1000];
  if( col == 1) sprintf(newfilename,"!%s",outfile);
  else sprintf(newfilename,"%s",outfile);

  char * coltype[7] ={"ell","TT","EE","BB","TE","TB","EB"};
  char * colform[7] ={"1D","1D","1D","1D","1D","1D","1D"};
  
  if( col == 1) ffinit( &fptr, newfilename, &status);
  else ffopen( &fptr, newfilename, 1, &status);

  if( col == 1) fits_create_tbl( fptr, BINARY_TBL, nele, 0, NULL, NULL, NULL, NULL, &status);

  fits_movabs_hdu( fptr, 2, NULL, &status);

  if( col == 1) fits_insert_cols( fptr, 1, 7, coltype, colform, &status);

  fits_write_col( fptr, TDOUBLE, col, 1, 1, nele, vect, &status);

  ffclos(fptr,&status);

}




void read_fits_puremll( char *infile, int nele, double *mll)
{
  int l1, l2, status=0, hdutyp, anynul;
  fitsfile *fptr;
  char comment[81];
  long size;
  double *vect=NULL;

  // open file
  ffopen( &fptr, infile, 0, &status);
  if( status) {
    printf( "ERROR when opening file : %s\n", infile);
    exit( -1);
  }

  fits_movabs_hdu( fptr, 2, &hdutyp, &status);
  if( status) {
    printf( "ERROR when opening HDU 2 for file : %s\n", infile);
    exit( -1);
  }

  fits_read_key( fptr, TLONG, "NAXIS2", &size, comment, &status);
  size = sqrt(size);
  if( size < nele) {
    printf( "Mll too short !\n");
    printf( "(%ldx%ld) where we need (%dx%d)\n", size, size, nele, nele);
    exit( -1);
  }
/*   printf( "size = %ld\n", size); */
  
  vect = (double *) malloc( size*size*sizeof( double) );
  if( vect == NULL) {
    printf( "Not enough memory !\n");
    exit( -1);
  }

  /* read col */
  fits_read_col_dbl( fptr, 1, 1, 1, size*size, DBL_MAX, vect, &anynul, &status);
  if( status) {
    printf( "ERROR when reading file : %s\n", infile);
    exit( -1);
  }

  for( l1=0; l1<nele; l1++)
    for( l2=0; l2<nele; l2++)
      mll[l1*nele+l2] = vect[l1*size+l2];
  free( vect);
  
  // close file
  ffclos(fptr, &status);

}





//=========================================================================

// int get_random_alms( int *stream, int lmax, double *cl, int noStokes, int nmvals, int *mvals, s2hat_dcomplex *alms)
// /* noStokes must be equal to 2 ! *
//  * cl contains first EE (lmax+1 - numbers) and then BB (lmax+1 - numbers) */
// {
//   double *revect, *imvect;
//   int i, is, j, l, lmax1, mindx, m, mstart, nlm;

//   lmax1 = lmax + 1;

//   nlm = lmax1*nmvals;

//   revect = (double *)calloc( nlm*noStokes, sizeof( double));
//   sprng_gaussran( stream, nlm*noStokes, revect);

//   imvect = (double *)calloc( nlm*noStokes, sizeof( double));
//   sprng_gaussran( stream, nlm*noStokes, imvect);

//   mstart = (mvals[0] == 0) ? 1 : 0;


//   /* do all the l,m modes now including m = 0 cases */

//   i = 4*noStokes;

//   if( mstart == 0)
//     {
//       for( i = 4*noStokes, l = 2; l < lmax1; l++, i += 4*noStokes)   // leave room for l = 0, 1 modes
//         {

// 	  for( is = 0; is < noStokes; is++)
// 	    {
//               alms[ is*nlm + l].re = sqrt(cl[is*lmax1+l])*revect[i++];    // real
//               alms[ is*nlm + l].im = 0.0;                                 // imaginary
// 	    }
//         }
//     }

//   for( mindx = mstart, l = 2; l < lmax1; l++, i += 4*noStokes, mindx = mstart)   // leave room for l = 0, 1 modes
//     {

//       for( mindx = mindx; (mindx < nmvals) && (mvals[ mindx] <= l); mindx++)
//         {
	  
// 	  for( is = 0; is < noStokes; is++)
// 	    {
// 	      alms[ is*nlm + mindx*lmax1 + l].re = sqrt( 0.5*cl[is*lmax1+l])*revect[i];      // real
// 	      alms[ is*nlm + mindx*lmax1 + l].im = sqrt( 0.5*cl[is*lmax1+l])*imvect[i++];    // imaginary
// 	    }
//         }

//     }

//   free( revect);
//   free( imvect);

// }



// void sprng_gaussran( int *stream, int nvect, double *rvect)

// {
//   double fac, rsq, v1, v2, tvect[2];
//   int i;

//   for( i=0; i< nvect/2; i++) {

//     do
//     {
//        tvect[0] = sprng( stream);
//        tvect[1] = sprng( stream);

//        v1 = 2.0*tvect[0] - 1.0;
//        v2 = 2.0*tvect[1] - 1.0;
//        rsq = v1*v1 + v2*v2;
//     } while (rsq >= 1.0 || rsq == 0.0);

//     fac = sqrt(-2.0*log(rsq)/rsq);

//     rvect[ 2*i]=v1*fac;
//     rvect[ 2*i+1]=v2*fac;
//   }

//   if( nvect%2)
//   {

//     do
//     {
//        tvect[0] = sprng( stream);
//        tvect[1] = sprng( stream);

//        v1 = 2.0*tvect[0] - 1.0;
//        v2 = 2.0*tvect[1] - 1.0;
//        rsq = v1*v1 + v2*v2;
//     }
//     while( rsq>=1.0 || rsq==0.0);

//     fac=sqrt( -2.0*log(rsq)/rsq);

//     rvect[nvect-1]=v1*fac;
//   }

// }

void apply_maskTT( int nmaps, int map_size,
                 double *local_map, double *local_mask, double *local_apodizedmap)
{
  int n, imap, ishift_inmap, ishift_outmap;
  long p;

  for( imap = 0; imap<nmaps; imap++) {
    ishift_inmap = imap*map_size;
    ishift_outmap = 2*imap*map_size;

      for( p=0; p<map_size; p++) {
        local_apodizedmap[ishift_outmap + p] = local_map[ishift_inmap + p] * local_mask[p];
      }
  }

}

void apply_mask( int nmaps, int map_size, int ncomp,
		 double *local_map, double *local_mask, double *local_apodizedmap)
{
  int n, imap, ishift;
  long p;

  for( imap = 0; imap<nmaps; imap++) {
    ishift = imap*ncomp*map_size;
    
    for( n=0; n<ncomp; n++)
      for( p=0; p<map_size; p++) 
	local_apodizedmap[ishift + n*map_size + p] = local_map[ishift + n*map_size + p] * local_mask[p];
  }

}
