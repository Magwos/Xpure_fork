#include "xpure.h"
#include <regex.h>
#include <fitsio.h>
#include <chealpix.h>


#define STRINGSIZE 1024
#define LINESIZE 1024
#define VERBOSE 0

#define EXIT_INFO(Y,Z,args...) { FILE *X=stdout; fprintf( X, "[%s:%d] "Z,__func__, __LINE__, ##args); fflush(X); MPI_Abort( MPI_COMM_WORLD, Y); exit(Y); }
#define INFO(Y,args...)        { FILE *X=stdout; fprintf( X, Y, ##args); fflush(X); }
#define OK 0
#define NOK 1

/*
Table of Content
----------------
int get_parameter(const char *fname, const char *nom, char *value);
int get_vect_size( char *infile, int lmax);
int read_fits_vect( char *infile, double *vect, int lmax, int colnum, int def)
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
*/

int get_parameter( const char *fname, const char *nom, char *value)
{
  char ligne[1024];
  size_t n = 1024;
  int is_set, rank;
  FILE *f;

  char nomex[1024];
  char valstr[1024];

  regex_t re;
  regmatch_t mre[3];
  int regerr;

  f = fopen(fname, "r");
  if( !f ) return( -1);

  if( (regerr=regcomp(&re, "([a-zA-Z0-9_-]+)\\s*=\\s*([a-zA-Z0-9_\\.+-/]+)",
		      REG_EXTENDED)) != 0 )
  {
    fprintf(stderr, "Error compiling regexp\n");
    regerror(regerr, &re, valstr, 1024);
    fprintf(stderr, "Error: %s\n", valstr);
    return( -1);
  }

  is_set = 0;
  while( fgets(ligne, n, f) )
  {
    if( ligne[0] == '#' ) continue;
    if( (regerr=regexec(&re, ligne, 3, mre, 0)) != 0 ) continue;      

    strncpy(nomex, ligne+mre[1].rm_so, mre[1].rm_eo-mre[1].rm_so);
    nomex[mre[1].rm_eo-mre[1].rm_so] = '\0';

    if( strcmp(nom, nomex) )
      continue; /* it is not the right parameter, we continue */
    else
    {
      /* we found it ! */
      strncpy(valstr, ligne+mre[2].rm_so, mre[2].rm_eo-mre[2].rm_so);
      valstr[mre[2].rm_eo-mre[2].rm_so] = '\0';

      /* we want a string */
      strcpy((char*)value, valstr);

      is_set = 1;
      break;
    } /* end of extraction of parameter */

  }
  fclose(f);

  if( !is_set) {
    strcpy( value, "");
  }

  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  if( VERBOSE==1 || rank == 0) {
    printf( "%s = %s", nom, value);
    if( !is_set) printf( "   (not set)\n");
    else printf( "\n");
  }

  return( is_set);
}




//=========================================================================
//Fits tools
//=========================================================================
int get_vect_size( char *infile, int lmax)
{
  fitsfile *fptr;
  long nele;
  int status=0, hdutyp;
  char comment[81];

  if( infile != NULL) {
    // open file
    ffopen( &fptr, infile, 0, &status);
  
    fits_movabs_hdu( fptr, 2, &hdutyp, &status);

    fits_read_key( fptr, TLONG, "NAXIS2", &nele, comment, &status);

    // close file
    ffclos(fptr, &status);

  } else {
    
    nele = lmax+1;
    
  }  

  return( nele);
}


int read_fits_vect( char *infile, double *vect, int nele, int colnum, int def)
{
  int i, status=0, hdutyp, anynul;
  long ncol;
  fitsfile *fptr;
  char errbuf[STRINGSIZE];
  char comment[81];

  // open file
  ffopen( &fptr, infile, 0, &status);
  if( status != 0) {
    printf( "\t\tFill with default = %d\n", def);
    if( def != -1) {
      for( i=0; i<nele; i++) vect[i] = def;
      return( 0);
    } else exit( -1);
  }

  /* move HDU */
  fits_movabs_hdu( fptr, 2, &hdutyp, &status);
  if( status != 0) {
    fits_get_errstatus( status, errbuf);
    EXIT_INFO( status, "%s\n", errbuf);
  }
  
  /* read size of column */
  fits_read_key( fptr, TLONG, "NAXIS2", &ncol, comment, &status);
  if( status != 0) {
    fits_get_errstatus( status, errbuf);
    EXIT_INFO( status, "%s\n", errbuf);
  }    
  if( ncol < nele) {
    EXIT_INFO( NOK, "FITS file too short...\n");
  }  

  /* read */
  fits_read_col_dbl( fptr, colnum, 1, 1, nele, DBL_MAX, vect, &anynul, &status);
  if( status != 0) {
    printf( "\t\tFill with default = %d\n", def);
    if( def != -1) {
      for( i=0; i<nele; i++) vect[i] = def;
      return( 0);
    } else exit( -1);
/*     fits_get_errstatus( status, errbuf); */
/*     EXIT_INFO( status, "%s\n", errbuf); */
  }    

  /* close file */
  ffclos(fptr, &status);  
  if( status != 0) {
    fits_get_errstatus( status, errbuf);
    EXIT_INFO( status, "%s\n", errbuf);
  }

  return( OK);
}



void read_fits_map( int nside, double *map, char *infile, int col)
{
  int status = 0, hdutyp, anynul;
  long nele=0, p, ipix, npix;
  char ordering[80];
  char comment[81];
  fitsfile *fptr;
  double *tmp;
  char errbuf[31];

  npix = nside*nside*(long)12;

  ffopen( &fptr, infile, 0, &status);
  if( status) {
    fits_get_errstatus( status, errbuf); 
    printf("\nFITS ERROR : %s\n", errbuf);
    exit(-1);
  }

  fits_movabs_hdu( fptr, 2, &hdutyp, &status);

  fits_read_key( fptr, TSTRING, "ORDERING", ordering, comment, &status);
  if( status) {
    fits_get_errstatus( status, errbuf); 
    printf("\nFITS ERROR : %s (ORDERING) : assumed RING\n", errbuf);
    sprintf( ordering, "RING");
    status=0;
  }

  tmp = (double *) malloc( npix*sizeof(double));

  fits_read_col_dbl( fptr, col, 1, 1, npix, DBL_MAX, tmp, &anynul, &status);
  if( status) {
    fits_get_errstatus( status, errbuf); 
    printf( "\nFITS ERROR: %s\n", errbuf);
  }

  ffclos(fptr, &status);

  /* revert if NESTED */
  if( !strcmp( ordering, "NESTED")) {
    printf( "NEST -> RING\n");

    for( p=0; p<npix; p++) {
      nest2ring( nside, p, &ipix);
      map[ipix] = (double)tmp[p];
    }
    
  } else {

    for( p=0; p<npix; p++) map[p] = (double)tmp[p];

  }
  free( tmp);
}



void read_TQU_maps( int nside, double *map, char *infile, int nstokes)
{
  long nele = 12*(long)nside*(long)nside;
  double *mapI = map;
  double *mapQ = map + nele;
  double *mapU = map + nele + nele;

  //READ MAPS
  printf( "           data   : %s ", infile);

  read_fits_map( nside, mapI, infile, 1);
  printf( "( T ");
  if( nstokes == 3) {
    read_fits_map( nside, mapQ, infile, 2);
    printf( "Q ");
    read_fits_map( nside, mapU, infile, 3);
    printf( "U ");
    }
  printf( ")\n");
  fflush( stdout);
  
}



void write_fits_vect( int nele, double *vect, char *outfile, int hdu)
{
  fitsfile *fptr;

  int irow,status = 0;
  char newfilename[1000];
  if( hdu == 1) sprintf(newfilename,"!%s",outfile);
  else sprintf(newfilename,"%s",outfile);

  char * coltype[1] ={"VECT"};
  char * colform[1] ={"1D"};
  char *tunit[1] = { "toto"};
  char extname[] = "ARRAY";

  if( hdu == 1) ffinit( &fptr, newfilename, &status);
  else ffopen( &fptr, newfilename, 1, &status);

  fits_create_tbl(fptr, BINARY_TBL, 0, 1, coltype, colform, tunit, extname, &status);

  fits_movabs_hdu( fptr, hdu+1, NULL, &status);
  for(irow=1; irow<=nele; irow++)
  {
    fits_write_col( fptr, TDOUBLE, 1, irow, 1, 1, &vect[irow-1], &status);
  }

  ffclos(fptr,&status);

}


void read_distribute_map( s2hat_pixeltype cpixelization, s2hat_int4 nmaps, s2hat_int4 mapnum, s2hat_int4 ncomp, 
			  char *mapname, s2hat_int4 first_ring, s2hat_int4 last_ring, s2hat_int4 map_size, 
			  s2hat_flt8 *local_map, s2hat_int4 myid, s2hat_int4 numprocs, s2hat_int4 root, MPI_Comm comm)
{
  int n;
  double *map;
  long npix = cpixelization.npixsall;  
  int nside = npix2nside(npix);

  /* read map */
  if( myid == root) {
    map = (double *) calloc( ncomp*npix, sizeof(double));
    for( n=0; n<ncomp; n++)
      read_fits_map( nside, &map[npix*n], mapname, n+1);
  }

  /* distribute */
  distribute_map( cpixelization, nmaps, mapnum, ncomp, first_ring, last_ring, map_size,
		  local_map, map, myid, numprocs, root, comm);

  /* free */
  if( myid == root) free( map);

}


void write_fits_map(int nstokes, int npix, double *map, char *outfile)
{
  int irow,nside, status=0;
  fitsfile *fptr;

  nside = npix2nside(npix);

  char *coltype[3] = {"TEMPERATURE","Q_POLARISATION","U_POLARISATION"};
  char *colform[3] = { "1D", "1D", "1D"};
  char *tunit[3] = { "toto", "tata", "titi"};
  char extname[] = "ARRAY";

  char newfilename[1000];
  sprintf( newfilename, "!%s", outfile);
  ffinit( &fptr, newfilename, &status);

  fits_create_tbl(fptr, BINARY_TBL, 0, nstokes, coltype, colform, tunit,
      extname, &status);

  fits_movabs_hdu( fptr, 1+1, NULL, &status);
  fits_update_key( fptr, TSTRING, "ORDERING", "RING", "", &status);
  fits_update_key( fptr, TLONG,   "NSIDE", &nside, "", &status);

  for(irow=1; irow<=npix; irow++)
  {
    fits_write_col( fptr, TDOUBLE, 1, irow, 1, 1, &map[0*npix + irow-1], &status);
  if( nstokes > 1)
    fits_write_col( fptr, TDOUBLE, 2, irow, 1, 1, &map[1*npix+ irow-1], &status);
  if( nstokes > 2)
    fits_write_col( fptr, TDOUBLE, 3, irow, 1, 1, &map[2*npix+ irow-1], &status);

  }
  ffclos( fptr, &status);

}


void collect_write_map( s2hat_pixeltype cpixelization, s2hat_int4 nmaps, s2hat_int4 mapnum, s2hat_int4 nstokes, 
			char *mapname, s2hat_int4 first_ring, s2hat_int4 last_ring, s2hat_int4 map_size, 
			s2hat_flt8 *local_map, s2hat_int4 myid, s2hat_int4 numprocs, s2hat_int4 root, MPI_Comm comm)
{
  double *map;
  long npix = cpixelization.npixsall;

  if( myid == root) printf( "\tWrite map generated...\n");
  if( myid == root) map = (double *)calloc( nstokes*npix, sizeof( double));

  collect_map( cpixelization, nmaps, mapnum, nstokes, map, first_ring, last_ring, map_size,
	       local_map, myid, numprocs, root, comm);

  if( myid == root) {
    printf( "\t\t%s\n", mapname);
    write_fits_map( nstokes, npix, map, mapname);
  }

  if( myid == root) free( map);

}


void collect_write_mll( int nlmax, char *mllfile, double *local_mll, s2hat_int4 myid, s2hat_int4 root, MPI_Comm comm)
{
  double *mll;
  long nele = (nlmax+1)*(nlmax+1);

/*   if( myid == root) printf( "\tWrite mll generated...\n"); */
  if( myid == root) mll = (double *)calloc( nele, sizeof( double));

  MPI_Reduce( local_mll, mll, nele, MPI_DOUBLE, MPI_SUM, root, comm);

  if( myid == root) {
    printf( "\t\t%s\n", mllfile);
    write_fits_vect( nele, mll, mllfile, 1);
  }

  if( myid == root) free( mll);
}

//=========================================================================





