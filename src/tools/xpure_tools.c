//******************************
//     Xpol tools
//******************************
#include "xpure.h"

#define PRINT_COND_NUM 1

extern double dlamch_(char *), dlange_(char *, int *, int *, double *, int *, double *);

/*
Table of contents
-----------------
void PCl( int nele, int nbin, double *matp, double *pseudocl, double *pseudocb)
void PMQ( int nbin, int nele, double *P, double *V, double *Q, double *M)
double SolveSystem( int nele, double *init_mat, double *cell)
void BinShifts( int nbins, char* mask_list, double *binshift, double *nbin_per_mask)
void GatherCell( int nbins, int *bshift, int *nbin_per_mask, double *cell, int root, MPI_Comm root_comm);
void GatherMll( int nbins, int *bshift, int *nbin_per_mask, double *mbb, int root, MPI_Comm root_comm);
*/


//=========================================================================
// Multipication P*Cl
void PCl( int nele, int nbin, double *matp, double *pseudocl, double *pseudocb)
{

#ifdef CNAG

  old_dgemv( NoTranspose, nbin, nele, 1., matp, nele, pseudocl, 1, 0., pseudocb, 1);
  
#elif defined(CBLAS) || defined(LIBSCI)

  cblas_dgemv( CblasRowMajor, CblasTrans, nele, nbin, 1., matp, nele, pseudocl, 1, 0., pseudocb, 1);
  
#elif ACML

  double dzero=0.0, done=1.0;
  int ione=1;
  char transa = 'T';
  dgemv_( &transa, &nele, &nbin, &done, matp, &nele, pseudocl, &ione, &dzero, pseudocb, &ione);

#elif ESSL

  char transa = 'T';
  dgemv( &transa, nele, nbin, 1., matp, nele, pseudocl, 1, 0., pseudocb, 1);

#elif MKL

  cblas_dgemv( CblasRowMajor, CblasNoTrans, nele, nbin, 1., matp, nele, pseudocl, 1, 0., pseudocb, 1);

#elif GSL

  CBLAS_TRANSPOSE_t TransA = CblasTrans;
  gsl_matrix * A; 
  size_t n1=nele;
  size_t n2=nbin;
  int b, l;
  A = gsl_matrix_alloc (n1, n2);
  for( b=0; b<nbin; b++)
    for( l=0; l<nele; l++) 
      gsl_matrix_set (A, l,b, matp[b*nele+l]);
  
  gsl_vector * V, *Vo;
  V = gsl_vector_alloc (n1);
  for( l=0; l<nele; l++) 
    gsl_vector_set (V, l, pseudocl[l]);   
  Vo = gsl_vector_alloc (n2);
  gsl_vector_set_zero (Vo);
  gsl_blas_dgemv (TransA, 1., A, V, 0., Vo);

  gsl_matrix_free (A);
  gsl_vector_free (V);

  for( b=0; b<nbin; b++) 
    pseudocb[b] = gsl_vector_get (Vo, b);    

    gsl_vector_free(Vo);

#endif
  
}





void PMQ( int nbin, int nele, double *P, double *V, double *Q, double *M)
// multiply matrix two by two
{
  double *R = (double *) calloc( nbin*nele, sizeof(double));

#ifdef CNAG

  old_dgemm( NoTranspose, NoTranspose, nbin, nele, nele, 1., P, nele, V, nele, 0., R, nele);
  old_dgemm( NoTranspose, NoTranspose, nbin, nbin, nele, 1., R, nele, Q, nbin, 0., M, nbin);

#elif defined(CBLAS) || defined(LIBSCI)

  cblas_dgemm( CblasRowMajor, CblasTrans,   CblasTrans, nbin, nele, nele, 1., P, nele, V, nele, 0., R, nele);
  cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans, nbin, nbin, nele, 1., R, nele, Q, nbin, 0., M, nbin);

#elif ACML

  char transa, transb;
  double ione=1, izero=0;

  transa = 'T'; transb = 'T';
  dgemm_( &transa, &transb, &nele, &nbin, &nele, &ione, V, &nele, Q, &nbin, &izero, R, &nele);
  transa = 'T'; transb = 'N';
  dgemm_( &transa, &transb, &nbin, &nbin, &nele, &ione, P, &nele, R, &nele, &izero, M, &nbin);

#elif ESSL

  char transa = 'T', transb = 'T';
  dgemm( &transa, &transb, nbin, nele, nele, 1., P, nele, V, nele, 0., R, nele);
  transa = 'N'; transb = 'T';
  dgemm( &transa, &transb, nbin, nbin, nele, 1., R, nele, Q, nbin, 0., M, nbin);

#elif MKL

  cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, nbin, nele, nele, 1., P, nele, V, nele, 0., R, nele);
  cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, nbin, nbin, nele, 1., R, nele, Q, nbin, 0., M, nbin);

#elif GSL

  CBLAS_TRANSPOSE_t TransA = CblasTrans;
  CBLAS_TRANSPOSE_t TransB = CblasNoTrans;
  gsl_matrix * A, *B, *C, *D, *E;  
  size_t n1=nbin;
  size_t n2=nele;
  int b, l; 
  A = gsl_matrix_alloc (n2, n1);    
  B = gsl_matrix_alloc (n2, n2); 
  C = gsl_matrix_alloc (n1, n2);   
  D = gsl_matrix_alloc (n1, n2);
  E = gsl_matrix_alloc (n1, n1);
  for( b=0; b<nbin; b++)
    for( l=0; l<nele; l++) {
      gsl_matrix_set (A, l,b, P[b*nele+l]);   
      gsl_matrix_set (D, b,l, Q[l*nbin+b]);
    }
  for( b=0; b<nele; b++)   
    for( l=0; l<nele; l++) 
      gsl_matrix_set (B, b,l, V[b*nele+l]);
  gsl_matrix_set_zero (C); 
  gsl_matrix_set_zero (E);
  gsl_blas_dgemm (TransA, TransA, 1., A, B, 0., C);
  gsl_blas_dgemm (TransB, TransA, 1., C, D, 0., E);   

  gsl_matrix_free (A);  
  gsl_matrix_free (B);  
  gsl_matrix_free (C); 

  for( b=0; b<nbin; b++) 
    for( l=0; l<nbin; l++) 
      M[b*nbin+l] = gsl_matrix_get (E, b, l); 
  gsl_matrix_free (D);
  gsl_matrix_free (E);

#endif

  free( R);

}

void bin_pseudo_spectra_not_weighted( int nbins, int bin_space, double *Pseudo_spectra, double *Binned_spectrum)
{
  int j;
  int k1;
  double component_cell;

  for(j=0; j<nbins; j++){
    component_cell = 0;
    for (k1=0; k1<bin_space;k1++){
      component_cell += Pseudo_spectra[ j*bin_space + k1 ];
    }
    Binned_spectrum[j] = component_cell/bin_space; 
    }
}


void bin_coupling_matrices_not_weighted( int nbins, int nele, int bin_space, double *Mll_matrix, double *Mbb_matrix)
{
  int i;
  int j;
  int k1;
  int k2;
  double component_matrix;

  for(i=0; i<nbins; i++){
    for(j=0; j<nbins; j++){
      component_matrix = 0;
      for (k1=0; k1<bin_space;k1++){
        for (k2=0; k2<bin_space;k2++){
          component_matrix += Mll_matrix[ (i*bin_space + k1)*nele + (j*bin_space + k2) ];
        }
      }
      Mbb_matrix[i*nbins + j] = component_matrix/bin_space;
      }
    }
}









double SolveSystem( int nele, double *init_mat, double *cell)
{
  long n;
  double rcond=-1;
  double *mat;

  /* copy input mat */
  mat = (double *) malloc( nele*nele*sizeof(double));
  for( n=0; n<nele*nele; n++) mat[n] = init_mat[n];

#ifdef CNAG
  NagError fail;
  Integer *ipiv;

  ipiv = NAG_ALLOC(nele, Integer);

  nag_dgetrf( Nag_RowMajor, nele, nele, mat, nele, ipiv, &fail);
  if( fail.code != NE_NOERROR) {
    printf( "LU decomp FAILED : \n");
    printf( fail.message);
  }

  nag_dgetrs( Nag_RowMajor, Nag_NoTrans, nele, 1, mat, nele, ipiv, cell, 1, &fail);
  if( fail.code != NE_NOERROR) {
    printf( "Solve FAILED : \n");
    printf( "%s\n", fail.message);
  }

  NAG_FREE( ipiv);

#elif CBLAS

  int info;
  int *ipiv;

  ipiv = (int *)calloc( nele, sizeof( int));

  info = clapack_dgetrf( CblasRowMajor, nele, nele, mat, nele, (int *)ipiv);

  info = clapack_dgetrs( CblasRowMajor, CblasNoTrans, nele, 1, mat, nele, ipiv, cell, nele);

  free( ipiv);
 
#elif LIBSCI

  int info;
  int *ipiv;

  ipiv = (int *)calloc( nele, sizeof( int));

  info = LAPACKE_dgetrf( CblasRowMajor, nele, nele, mat, nele, (int *)ipiv);

  info = LAPACKE_dgetrs( CblasRowMajor, CblasNoTrans, nele, 1, mat, nele, ipiv, cell, nele);

  free( ipiv);

#elif ACML

  int info=0, ione=1;
  int *ipiv;
  char  transa = 'N';

  ipiv = (int *)calloc( nele, sizeof( int));

  double anorm, eps;
  double *dwork = malloc( 4*nele*sizeof( double));
  char norm = '1'; /* '1'=1-norm, 'I'=inf-norm */
  char cmach = 'e'; /* relative machine precision */
  eps = dlamch_( &cmach);
  anorm = dlange_( &norm, &nele, &nele, mat, &nele, dwork);

#if PRINT_COND_NUM == 1
  printf( "\t(norm=%e) (eps=%e)\t", anorm, eps);
#endif

  /* LU decomp */
  dgetrf_( &nele, &nele, mat, &nele, ipiv, &info);
  if( info) {
    if( info<0) printf( "\n\ndgetrf : parameter %i BAD VALUE\n\n", -info);
    if( info>0) printf( "\n\ndgetrf : U(%i,%i) is exactly zero\n\n", info, info);
  }

  /* Condition Number */
  int *iwork = malloc( nele*sizeof( int));
  dgecon_( &norm, &nele, mat, &nele, &anorm, &rcond, dwork, iwork, &info);
  free( iwork);
  free( dwork);

  if( rcond < sqrt( eps) ) printf( "\n **** WARNING : Mll ill-conditionned !  ****");

  /* solve a general system of linear equations */
  dgetrs_( &transa, &nele, &ione, mat, &nele, ipiv, cell, &nele, &info);
  if( info) {
    if( info <0) printf( "dgetrs : parameter %i BAD VALUE\n", -info);
  }

  free( ipiv);

#elif ESSL

  int info;
  int *ipiv;
  char  transa = 'N';

  ipiv = (int *)calloc( nele, sizeof( int));

  dgetrf( nele, nele, mat, nele, ipiv, &info);

  dgetrs( &transa, nele, 1, mat, nele, ipiv, cell, nele, &info);

  free( ipiv);

#elif MKL

  int info;
  int *ipiv;
  char transa = 'N';
  int one=1;

  ipiv = (int *)calloc( nele, sizeof( int));

  dgetrf( &nele, &nele, mat, &nele, ipiv, &info);
  if( info) {
    if( info<0) printf( "dgetrf : parameter %i BAD VALUE\n", -info);
    if( info>0) printf( "dgetrf : U(%i,%i) is exactly zero\n", info, info);
  }

  dgetrs( &transa, &nele, &one, mat, &nele, ipiv, cell, &nele, &info);
  if( info) {
    if( info <0) printf( "dgetrs : parameter %i BAD VALUE\n", -info);
  }

  free( ipiv);

#elif GSL

  gsl_matrix *A, *B;
  gsl_vector *v, *x, *residual;
  int b, l;     
  int sign=0;
  A = gsl_matrix_alloc (nele, nele);
  B = gsl_matrix_alloc (nele, nele);
  v = gsl_vector_alloc (nele);
  x = gsl_vector_alloc (nele);
  residual = gsl_vector_alloc (nele);
  gsl_vector_set_zero (x);
  gsl_vector_set_zero (residual);
  for( b=0; b<nele; b++){   
    for( l=0; l<nele; l++){        
      gsl_matrix_set (A, l,b, mat[b*nele+l]);  
      gsl_matrix_set (B, l,b, mat[b*nele+l]);
    }
  }

  for( l=0; l<nele; l++)
    gsl_vector_set (v, l, cell[l]); 

  gsl_permutation *p;
  p = gsl_permutation_alloc (nele);
  int * signum= &sign;
  int res;
  res = gsl_linalg_LU_decomp (A, p, signum);
  printf ("LU decomp : %d\n", res);
  // gsl_linalg_LU_solve (A, p, v, x);
  gsl_linalg_LU_refine (B, A, p, v, x, residual);
  
  for( l=0; l<nele; l++)
    cell[l] = gsl_vector_get (x, l);
  
  gsl_matrix_free (A);
  gsl_permutation_free(p);
  gsl_vector_free (x);
  gsl_vector_free (v);

#endif

  free( mat);

  return( rcond);
}
//=========================================================================








//=========================================================================
// Compute BinShift and nbin_per_mask when combining masks
void BinShifts( int nbins, char* mask_list, int *binshift, int *nbin_per_mask, int rank)
{
  int b, bshift=-1;
  double *bin2mask = NULL;

  int root = 0;
  //read mask_list
  bin2mask = (double *) malloc( nbins * sizeof( double));
  read_fits_vect( mask_list, bin2mask, nbins, 1, 0);
  if( rank==root) for( b=0; b<nbins; b++) printf( "\t%d", (int)bin2mask[b]);
  if( rank==root) printf( "\n");

  for( b=0; b<nbins; b++) {
    if( bin2mask[b] != bin2mask[binshift[bshift]] || b == 0) {
      bshift++;
      binshift[bshift] = b;
      nbin_per_mask[bshift] = 1;
    } else {
      nbin_per_mask[bshift]++;
    }
  }
  
  free( bin2mask);
}


void GatherCell( int nbins, int bshift, int nbin_per_mask, double *cell, MPI_Comm root_comm)
{
  int b;
  double *final_cell;

  int root=0, rank;
  MPI_Comm_rank( root_comm, &rank);
  

  if( rank == root)
    final_cell = (double *) calloc( nbins, sizeof(double));

  for( b=0; b<bshift; b++) cell[b] = 0.;
  for( b=bshift+nbin_per_mask; b<nbins; b++) cell[b] = 0.;

  MPI_Reduce( cell, final_cell, nbins, MPI_DOUBLE, MPI_SUM, root, root_comm);

  if( rank == root)
    for( b=0; b<nbins; b++)
      cell[b] = final_cell[b];

  if( rank == root)
    free( final_cell);

}

void GatherMll( int nbins, int bshift, int nbin_per_mask, double *mbb, MPI_Comm root_comm)
{
  int b1, b2, b;
  double *final_mbb;
  int nele = nbins*nbins;

  int root=0, rank;
  MPI_Comm_rank( root_comm, &rank);
  
  if( rank == root)
    final_mbb = (double *) calloc( nele, sizeof(double));

  for( b1=0; b1<nbins; b1++) {
    if( b1 >= bshift && b1 < bshift+nbin_per_mask) continue;
    for( b2=0; b2<nbins; b2++) mbb[ b2*nbins + b1] = 0.;
  }

  MPI_Reduce( mbb, final_mbb, nele, MPI_DOUBLE, MPI_SUM, root, root_comm);

  if( rank == root)
    for( b=0; b<nele; b++)
      mbb[b] = final_mbb[b];

  if( rank == root)
    free( final_mbb);


}

//=========================================================================




