/******************************************************
*******************************************************
 *                                                    *
 * Set of routines computing pure alm coefficients    *
 * (   K.M.Smith, PhysRev D, 74, 3002, 2006           *
 * and Grain et al, 2008, in preparation)             *
 *                                                    *
 * - jgrain@apc                             june 2008 *
 *                                                    *
 * minor rewrite/cleaning - rs@apc          july 2008 *
 *                                                    *
 * general restructuring, map2alm driver added        *
 * healpix format capability enabled- rs@apc aug 2008 *
 *                                                    *
 * seems to be working but multi mask/window/maps     *
 * options have not been fully tested yet             *
 *                                - rs@apc 08/07/2008 *
 *                                                    *
 * minor changes/updates          - rs@apc 01/01/2010 *
 *                                                    *
 ******************************************************
 ******************************************************/

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "malloc.h"

#include "mpi.h"
#include "s2hat.h"

/*useful constant*/
#define pi 4.0*atan(1.0)

s2hat_int4 purealm_spin0_update( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);
s2hat_int4 purealm_spin0_update_s2hat( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);
s2hat_int4 purealm_spin0_update_healpix( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);

s2hat_int4 purealm_spin1_update( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);
s2hat_int4 purealm_spin1_update_s2hat( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);
s2hat_int4 purealm_spin1_update_healpix( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);

s2hat_int4 purealm_spin2_update( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);
s2hat_int4 purealm_spin2_update_s2hat( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);
s2hat_int4 purealm_spin2_update_healpix( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*, s2hat_dcomplex*);

s2hat_int4 apodize_maps_complex( s2hat_int4, s2hat_int4, s2hat_flt8*, s2hat_int4, s2hat_flt8*);
s2hat_int4 combine_mask_window( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_int4*, s2hat_int4, s2hat_flt8*);

s2hat_int4 wlm_scalar2vector_s2hat( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*);
s2hat_int4  wlm_vector2tensor_s2hat( s2hat_int4, s2hat_int4, s2hat_int4, s2hat_dcomplex*);

s2hat_int4 window_scalar2all( s2hat_pixeltype pixelisation, s2hat_scandef scan, s2hat_int4 lmax, s2hat_int4 mmax, s2hat_int4 nmvals, s2hat_int4 *mvals, 
								    s2hat_int4 nwins, s2hat_int4 first_ring, s2hat_int4 last_ring, s2hat_flt8 *local_w8ring, s2hat_int4 mapsize, 
								    s2hat_int4 maskstride, s2hat_int4 *local_mask, s2hat_flt8 *local_window_scal, s2hat_flt8 *local_window_vect, 
								    s2hat_flt8 *local_window_tens, s2hat_int4 my_rank, s2hat_int4 numprocs, MPI_Comm my_comm)

/***********************************************************************
 * Computes spin weighted window functions for spin 1 and 2            *
 * given the scalar (spin-0) window on the input.                      *
 *                                                                     *
 * INPUT: distributed real valued scalar window (8byte floats)         *
 *        stored in local_window_scal                                  *
 *        distributed mask (4byte integer)                             *
 *        stored in local_mask.                                        *
 *                                                                     *
 * OUTPUT: distributed spin 1, complex valued, window (s2hat_dcomplex) *
 *         stroed in local_window_vect;                                *
 *         distributed spin 2, complex valued, window (s2hat_dcomplex) *
 *         stored in local_window_tens.                                *
 **********************************************************************/
 /**********************************************************************
 * NB:                                                                 *
 * this routine allocates internally space to accommodate a local set  *
 * of alms coefficients (E and B type) =                               *
 * = sizeof( s2hat_dcomplex) * noSpins * nwins * (lmax+1) * nmvals     *
 **********************************************************************/

{
  s2hat_int4 i, j;
  s2hat_int4 ipix, imap, ishift, ishift1, jshift, mstride;

  s2hat_int4 lda, lmax1, spin;
  s2hat_int4 int_one = 1, nstokes = 2;        /* i.e., spin fields */

  s2hat_flt8  sqrtfact, di;
  s2hat_flt8 *local_w_scal_complex;
  s2hat_dcomplex *local_wlm, *local_alm;

  lda = lmax;            /* S2HAT convention for leading dimension of arrays - internal */
  lmax1 = lmax+1; 
  mstride = (maskstride) ? 1 : 0;

  /* compute Wlm multipoles with s2hat routine */

  local_wlm=(s2hat_dcomplex *)calloc( nwins*nstokes*lmax1*nmvals, sizeof(s2hat_dcomplex));

  local_w_scal_complex = local_window_vect;            /* just reuse the memory -> local_window_vect will be */
													   /* overwritten later with the correct output */
  for( imap=0; imap<nwins; imap++)
    {
      for( ipix = imap*nstokes*mapsize; ipix<(imap*nstokes+1)*mapsize; ipix++)
	{
	    local_w_scal_complex[ ipix]=local_window_scal[ ipix];
	    local_w_scal_complex[ ipix+mapsize]=0.0;
	}
    }

  spin = 0;  
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nwins, first_ring, last_ring, local_w8ring, 
                      mapsize, local_w_scal_complex, lda, local_wlm, numprocs, my_rank, my_comm);

  /* compute the spin=1 components */

  wlm_scalar2vector_s2hat( nwins, nmvals, lmax, local_wlm);

  spin = 1;
  s2hat_alm2map_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nwins, first_ring, last_ring, mapsize, 
		      local_window_vect, lda, local_wlm, numprocs, my_rank, my_comm);

  combine_mask_window( nwins, mapsize, mstride, local_mask, int_one, local_window_vect);

  /*compute the spin=2 components*/

  wlm_vector2tensor_s2hat( nwins, nmvals, lmax, local_wlm);

  spin = 2;
  s2hat_alm2map_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nwins, first_ring, last_ring, mapsize, 
		      local_window_tens, lda, local_wlm, numprocs, my_rank, my_comm);
  
  free(local_wlm);

  combine_mask_window( nwins, mapsize, mstride, local_mask, int_one, local_window_tens);

  return( 1);
    
}

s2hat_int4 s2hat_apodizedmaps2purealm( s2hat_pixeltype pixelisation, s2hat_scandef scan, s2hat_int4 lmax, s2hat_int4 mmax, 
									   s2hat_int4 nmvals, s2hat_int4 *mvals, s2hat_int4 nmaps, s2hat_int4 first_ring, 
									   s2hat_int4 last_ring, s2hat_flt8 *local_w8ring, s2hat_int4 mapsize, s2hat_flt8 *local_pXscal, 
									   s2hat_flt8 *local_pXvect, s2hat_flt8 *local_pXtens, s2hat_int4 lda, s2hat_dcomplex *local_apurelm, 
									   s2hat_int4 my_rank, s2hat_int4 numprocs, MPI_Comm my_comm)

/***********************************************************************
 * Computes the pure alm coefficients for E and B polarization maps    *
 * given the set of the **apodized** Q and U maps on the input.        *
 *                                                                     *
 * INPUT: distributed real valued maps (8byte floats)                  *
 *        stored in                                                    *
 *          local_pXscal = [Wspin0*Q, Wspin0*U]                        *
 *          local_pXvect =                                             *
 *           = [re(Wspin1)*Q-im(Wspin1)*U, im(Wspin1)*Q, re(Wspin1)*U] *
 *          local_pXtens =                                             *
 *           = [re(Wspin2)*Q-im(Wspin2)*U, im(Wspin2)*Q, re(Wspin2)*U] *
 *                                                                     *
 *         where Q and U stand for the (distributed) Q and U maps      *
 *           and Wspin0, Wspin1, Wspin2 are the spin-valued windows.   *
 *                                                                     *
 * OUTPUT: distributed, complex valued Elm and Blm **pure** harmonic   *
 *         coefficients computed for the input apodized maps and       *
 *         stored in:                                                  * 
 *             local_apurelm (s2hat_dcomplex).                         *
 **********************************************************************/
/**********************************************************************
 * NB:                                                                 *
 * this routine allocates internally space to accommodate a local set  *
 * of alms coefficients (E and B type) =                               *
 * = sizeof( s2hat_dcomplex) * noSpins * nmaps * (lmax+1) * nmvals     *
 **********************************************************************/

{
  s2hat_int4 i, j, imap, ishift, lmax1;
  s2hat_int4 nstokes=2;     /* i.e., spin fields */
  s2hat_int4 spin;
  s2hat_flt8 di, sqrtfact;
  
  s2hat_dcomplex *local_alm;

  lmax1 = lmax+1;

  /* compute the different spin-weighted SHT and accumulate the output in local_apurealm */

  local_alm = (s2hat_dcomplex *)calloc( nmaps*nstokes*lmax1*nmvals, sizeof( s2hat_dcomplex));

  spin = 0;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nmaps, first_ring, last_ring, local_w8ring, mapsize, 
		      local_pXtens, lda, local_alm, numprocs, my_rank, my_comm);

  /* add spin0 contribution */
   purealm_spin0_update( nmaps, nmvals, lmax, lda, local_alm, local_apurelm);

  spin = 1;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nmaps, first_ring, last_ring, local_w8ring, mapsize, 
		      local_pXvect, lda, local_alm, numprocs, my_rank, my_comm);
  
  /* add spin1 contribution */
  purealm_spin1_update( nmaps, nmvals, lmax, lda, local_alm, local_apurelm);

  spin = 2;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nmaps, first_ring, last_ring, local_w8ring, mapsize, 
		      local_pXscal, lda, local_alm, numprocs, my_rank, my_comm);

  /* add spin2 contribution */
  purealm_spin2_update( nmaps, nmvals, lmax, lda, local_alm, local_apurelm);

  free(local_alm);
  
}

/* general single call driver */

s2hat_int4 s2hat_map2purealm( s2hat_pixeltype pixelisation, s2hat_scandef scan, s2hat_int4 lmax, s2hat_int4 mmax, s2hat_int4 nmvals, 
							  s2hat_int4 *mvals, s2hat_int4 nmaps, s2hat_int4 first_ring, s2hat_int4 last_ring, s2hat_flt8 *local_w8ring, 
                              s2hat_int4 mapsize, s2hat_int4 maskstride, s2hat_int4 *local_mask, s2hat_int4 winstride, s2hat_flt8 *local_swindow, 
                              s2hat_flt8* local_cmbmap, s2hat_int4 lda, s2hat_dcomplex *local_apurelm, s2hat_int4 my_rank, s2hat_int4 numprocs, 
                              MPI_Comm my_comm)

/***********************************************************************
 * Computes the pure alm coefficients for E and B polarization maps    *
 * given the set of the **unapodized** Q and U maps on the input and   *
 * a scalar apodization plus mask defining the weights and the obser-  *
 * ved sky area.                                                       *
 *                                                                     *
 *     - distributed real value sky map (8byte floats)                 *
 *       stored in local_cmbmap = [Q, U]                               *
 *     - scalar apodization (8byte floats)                             *
 *       stored in local_swindow                                       *
 *     - mask (4byte ints: 0 or 1)                                     *
 *       stored in local_mask;                                         *
 *                                                                     *
 *  All pixel-domain distributed in the same way as defined by the     *
 *  input params                                                       *
 * The routine can process up to nmaps together, which are stored on   *
 * the input in local_cmbmap array one after another i.e.,             *
 *                 [Q0,U0, Q1, U1, ... etc ...]                        *
 * They will be apodized all with the same apodization/mask stored in  *
 * local_swindow and local_mask, if winstride and maskstride are both  *
 * set to 0; alternately if either of those is non-zero then each map  *
 * will be apodized or masked with a different window/mask depending   *
 * on which of those two is non-zero.                                  *
 *                                                                     *
 * Note that in the case with different masks - a care needs to be     *
 * taken to avoid conflicts between the observed sky definition        *
 * provided by the scan structure and the different masks. To ensure   *
 * the consistency the scan structure has to be defined using the      *
 * effective mask caclulated as a sum of all different masks.          *
 *                                                                     * 
 * Note that only a single copy of the apodization/mask is stored on   *
 * the input in the case with winstride or maskstride set to 0.        *
 *                                                                     *
 * OUTPUT: distributed, complex valued Elm and Blm **pure** harmonic   *
 *         coefficients computed for the input apodized maps and       *
 *         stored in:                                                  * 
 *             local_apurelm (s2hat_dcomplex).                         *
 *          if nmaps > 1 then all the coefficients for different maps  *
 *          are concatenated together.                                 *
 **********************************************************************/
/**********************************************************************
 * NB:                                                                 *
 * this routine allocates internally space to accommodate;             *
 *                                                                     *
 * (i)  local sets of alms coefficients (E and B type) one per map     *
 *      =sizeof( s2hat_dcomplex) * nstokes * nmaps * (lmax+1) * nmvals *
 *                                                                     *
 * (ii) local sets of alms coefficients (E and B type) one per window  *
 *      =sizeof( s2hat_dcomplex) * nstokes * nwins * (lmax+1) * nmvals *
 *      where nwins = 1 (winstride = 0) or = nmaps (winstride = 1)     *
 *                                                                     *
 * (iii) pixel domain (complex) window:                                *
 *       = sizeof( s2hat_flt8) * noSpins * nmaps * mapsize             *
 *                                                                     *
 *********************************************************************** 
 ******* This can amount to as much as 1.5 memory needed to store ******
 ****************** all the inputs and outputs ************************* 
 **********************************************************************/

{
  s2hat_int4 i, j;
  s2hat_int4 ipix, imap, ishift, ishift1, jshift, kshift, kshift1, mstride, wstride, wmstride;

  s2hat_int4 lda_wlm, lmax1, spin, nwins;
  s2hat_int4 nstokes = 2;        /* i.e., spin fields */

  s2hat_flt8  sqrtfact, di;
  s2hat_flt8 *local_map;
  s2hat_dcomplex *local_wlm, *local_alm;

  lmax1 = lmax+1; 
  lda_wlm = lmax;     /* internal layout **must be** that of s2hat for the time being ... */

  mstride = (maskstride) ? 1 : 0;
  wstride = (winstride) ? 1 : 0;
  wmstride = (mstride) || (wstride) ? 1 : 0;

  nwins = (wstride) ? nmaps : 1;  /* number of windows : either one per map or just one */

  /* start with the scalar window */

  /* compute Wlm multipoles with s2hat routine */

  local_wlm=(s2hat_dcomplex *)calloc( nwins*nstokes*lmax1*nmvals, sizeof( s2hat_dcomplex));

  /* allocate space for all the maps ... */
	
  local_map=(s2hat_flt8 *)calloc( nmaps*nstokes*mapsize, sizeof( s2hat_flt8));

  for( imap=0; imap<nwins; imap++)
    {
      for( ipix = imap*nstokes*mapsize; ipix<(imap*nstokes+1)*mapsize; ipix++)
 	  {
	    local_map[ ipix]=local_swindow[ ipix];
	  }
    }

  spin = 0;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nwins, first_ring, last_ring, local_w8ring, 
                      mapsize, local_map, lda_wlm, local_wlm, numprocs, my_rank, my_comm);

  /* scalar contribution to pure alm first */

  apodize_maps_complex( nmaps, mapsize, local_cmbmap, wstride, local_map);

  local_alm = (s2hat_dcomplex *)calloc( nmaps*nstokes*lmax1*nmvals, sizeof( s2hat_dcomplex));

  spin = 2;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nmaps, first_ring, last_ring, local_w8ring, mapsize, 
		      local_map, lda, local_alm, numprocs, my_rank, my_comm);
  
  /* add the scalar window contribution to purealm (spin2 updated) */

  purealm_spin2_update( nmaps, nmvals, lmax, lda, local_alm, local_apurelm);

  /* and now the spin = 1 correction */

  /* first compute the spin=1 window harmonic space representation */

  wlm_scalar2vector_s2hat( nwins, nmvals, lmax, local_wlm);

  spin = 1;
  s2hat_alm2map_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nwins, first_ring, last_ring, mapsize, 
		      local_map, lda_wlm, local_wlm, numprocs, my_rank, my_comm);

  /* mask the spin1 window(s) */

  combine_mask_window( nmaps, mapsize, mstride, local_mask, wstride, local_map);

  apodize_maps_complex( nmaps, mapsize, local_cmbmap, wmstride, local_map);

  spin = 1;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nmaps, first_ring, last_ring, local_w8ring, mapsize, 
		      local_map, lda, local_alm, numprocs, my_rank, my_comm);

  /* spin1 update of the pure alms */

  purealm_spin1_update( nmaps, nmvals, lmax, lda, local_alm, local_apurelm);

  /* and now spin2 correction */

  /*compute the spin=2 window components*/

  wlm_vector2tensor_s2hat( nwins, nmvals, lmax, local_wlm);   // conforms with the s2hat format
  
  spin = 2;
  s2hat_alm2map_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nwins, first_ring, last_ring, mapsize, 
		      local_map, lda_wlm, local_wlm, numprocs, my_rank, my_comm);
  
  free(local_wlm);

  /* mask the spin2 window(s) */

  combine_mask_window( nmaps, mapsize, mstride, local_mask, wstride, local_map);

  /* apodize the skies with the spin2 windows */

  apodize_maps_complex( nmaps, mapsize, local_cmbmap, wmstride, local_map);

  /* compute spin2 correction */

  spin = 0;
  s2hat_map2alm_spin( pixelisation, scan, spin, lmax, mmax, nmvals, mvals, nmaps, first_ring, last_ring, local_w8ring, mapsize, 
		      local_map, lda, local_alm, numprocs, my_rank, my_comm);

  free( local_map);

  /* spin0 update of the pure alm */

  purealm_spin0_update( nmaps, nmvals, lmax, lda, local_alm, local_apurelm);

  free( local_alm);

  return( 1);
    
}

/* auxiliary routines for the pure alm calculations */

s2hat_int4 purealm_spin0_update( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, s2hat_int4 lda, 
                                 s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/************************************************************ 
 * performs a spin0 update of the pure alms i.e., adding to *
 * the local_qpurelm array a spin-0 **counterterm**         * 
 * - see Eq. (24) of Grain et al, (2009).                   *
 * Does that for both E and B multipoles stored in the array*
 ************************************************************/  

{
  s2hat_int4 nstokes = 2;

  if( lda == nstokes) return( purealm_spin0_update_healpix( nmaps, nmvals, lmax, local_alm, local_apurelm));
  else return( purealm_spin0_update_s2hat( nmaps, nmvals, lmax, local_alm, local_apurelm));

}

s2hat_int4 purealm_spin0_update_s2hat( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, 
                                       s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/* performs the spin0 update of pure alm in the s2hat format *
 * called by the driver: purealm_spin0_update                */
{
  s2hat_int4 i, imap, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di, sqrtfact;

  lmax1 = lmax+1;

  /* Q - map */

  for( imap=0; imap<nmaps; imap++)
  {

      for( j=0; j<nmvals; j++)
	  {

          ishift = (imap*nstokes*nmvals+j)*lmax1;

          /*  l = 0 and l = 1 modes are zero */

	      local_apurelm[ ishift].re=0.0;
	      local_apurelm[ ishift].im=0.0;
	  
	      local_apurelm[ ishift+1].re=0.0;
	      local_apurelm[ ishift+1].im=0.0;

          ishift += 2;
	  
	      for( i=2; i<lmax1; i++, ishift++)
	      {

			   di = (double)i;

			   sqrtfact = 1.0/sqrt((di-1.0)*di*(di+1.0)*(di+2.0));

	           local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	           local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;
	      }
	   }
    }

  /* U-map */
  
  for( imap=0; imap<nmaps; imap++)
  {
      for( j=0; j<nmvals; j++)
 	  {

          ishift = ((imap*nstokes+1)*nmvals+j)*lmax1;

          /* l = 0 and l = 1 modes are zero */

	      local_apurelm[ ishift].re=0.0;
	      local_apurelm[ ishift].im=0.0;
	  
	      local_apurelm[ ishift+1].re=0.0;
	      local_apurelm[ ishift+1].im=0.0;

          ishift += 2;

	      for( i=2; i<lmax1; i++, ishift++)
	      {
			   di = (double)i;

			   sqrtfact = 1.0/sqrt((di-1.0)*di*(di+1.0)*(di+2.0));

	           local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	           local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;
	      }
	   }
    }

    return( 1);

}

s2hat_int4 purealm_spin0_update_healpix( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, 
                                         s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/* performs the spin0 update of pure alm in the HEALPix format *
 * called by the driver: purealm_spin0_update                  */

{
  s2hat_int4 i, imap, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di, sqrtfact;

  lmax1 = lmax+1;

  for( imap=0; imap<nmaps; imap++)
    {

      for( j=0; j<nmvals; j++)
	{

          ishift = nstokes*(imap*nmvals+j)*lmax1;

          /*  l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;
	  
	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;
	  
	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;

	  for( i=2; i<lmax1; i++)
	    {

  	        /* Q-map */ 

                di = (double)i;

                sqrtfact = 1.0/sqrt((di-1.0)*di*(di+1.0)*(di+2.0));

	        local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	        local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;

                /* U-map */

                ishift++;

	        local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	        local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;

                ishift++;

	    }
	}
    }

    return( 1);
}

/* spin - 1 */

s2hat_int4 purealm_spin1_update( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, s2hat_int4 lda, 
								 s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/************************************************************ 
 * performs a spin1 update of the pure alms i.e., adding to *
 * the local_qpurelm array a spin-1 **counterterm**         * 
 * - see Eq. (24) of Grain et al, (2009).                   *
 * Does that for both E and B multipoles stored in the array*
 ************************************************************/

{
  s2hat_int4 nstokes = 2;

  if( lda == nstokes) return( purealm_spin1_update_healpix( nmaps, nmvals, lmax, local_alm, local_apurelm));
  else return( purealm_spin1_update_s2hat( nmaps, nmvals, lmax, local_alm, local_apurelm));

}

s2hat_int4 purealm_spin1_update_s2hat( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, 
									   s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/* performs the spin1 update of pure alm in the s2hat format *
 * called by the driver: purealm_spin0_update                */

{
  s2hat_int4 i, imap, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di, sqrtfact;

  lmax1 = lmax+1;

  /* Q - map */

  for( imap=0; imap<nmaps; imap++)
    {

      for( j=0; j<nmvals; j++)
	{

          ishift = (imap*nstokes*nmvals+j)*lmax1;

          /*  l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift].im=0.0;
	  
	  local_apurelm[ ishift+1].re=0.0;
	  local_apurelm[ ishift+1].im=0.0;

          ishift += 2;
	  
	  for( i=2; i<lmax1; i++, ishift++)
	    {

                di = (double)i;

                sqrtfact = 2.0/sqrt((di-1.0)*(di+2.0));

	        local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	        local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;
	    }
	}
    }

  /* U-map */
  
  for( imap=0; imap<nmaps; imap++)
    {
      for( j=0; j<nmvals; j++)
	{

          ishift = ((imap*nstokes+1)*nmvals+j)*lmax1;

          /* l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift].im=0.0;
	  
	  local_apurelm[ ishift+1].re=0.0;
	  local_apurelm[ ishift+1].im=0.0;

          ishift += 2;

	  for( i=2; i<lmax1; i++, ishift++)
	    {
                di = (double)i;

                sqrtfact = 2.0/sqrt((di-1.0)*(di+2.0));

	        local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	        local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;
	    }
	}
    }

   return( 1);

}

s2hat_int4 purealm_spin1_update_healpix( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, 
                                         s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/* performs the spin1 update of pure alm in the HEALPix format *
 * called by the driver: purealm_spin0_update                  */

{
  s2hat_int4 i, imap, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di, sqrtfact;

  lmax1 = lmax+1;

  for( imap=0; imap<nmaps; imap++)
    {

      for( j=0; j<nmvals; j++)
	{

          ishift = nstokes*(imap*nmvals+j)*lmax1;

          /*  l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;
	  
	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;
	  
	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;

	  for( i=2; i<lmax1; i++)
	    {

  	        /* Q-map */ 

                di = (double)i;

                sqrtfact = 2.0/sqrt((di-1.0)*(di+2.0));

	        local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	        local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;

                ishift++;

                /* U-map */

	        local_apurelm[ ishift].re += sqrtfact*local_alm[ ishift].re;
	        local_apurelm[ ishift].im += sqrtfact*local_alm[ ishift].im;

                ishift++;

	    }
	}
    }

    return( 1);

}

/* spin-2 */

s2hat_int4 purealm_spin2_update( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, s2hat_int4 lda, 
								 s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/************************************************************ 
 * performs a spin2 update of the pure alms i.e., adding to *
 * the local_qpurelm array a spin-2 **counterterm** equal to*
 * the standard pseudo-alms                                 *
 * - see Eq. (24) of Grain et al, (2009).                   *
 * Does that for both E and B multipoles stored in the array*
 ************************************************************/

{
  s2hat_int4 nstokes = 2;

  if( lda == nstokes) return( purealm_spin2_update_healpix( nmaps, nmvals, lmax, local_alm, local_apurelm));
  else return( purealm_spin2_update_s2hat( nmaps, nmvals, lmax, local_alm, local_apurelm));

}

s2hat_int4 purealm_spin2_update_s2hat( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, 
                                       s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)

/* performs the spin2 update of pure alm in the s2hat format *
 * called by the driver: purealm_spin0_update                */

{
  s2hat_int4 i, imap, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di;

  lmax1 = lmax+1;

  /* Q - map */

  for( imap=0; imap<nmaps; imap++)
    {

      for( j=0; j<nmvals; j++)
	{

          ishift = (imap*nstokes*nmvals+j)*lmax1;

          /*  l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift].im=0.0;
	  
	  local_apurelm[ ishift+1].re=0.0;
	  local_apurelm[ ishift+1].im=0.0;

          ishift += 2;
	  
	  for( i=2; i<lmax1; i++, ishift++)
	    {
	        local_apurelm[ ishift].re += local_alm[ ishift].re;
	        local_apurelm[ ishift].im += local_alm[ ishift].im;
	    }
	}
    }

  /* U-map */
  
  for( imap=0; imap<nmaps; imap++)
    {
      for( j=0; j<nmvals; j++)
	{

          ishift = ((imap*nstokes+1)*nmvals+j)*lmax1;

          /* l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift].im=0.0;
	  
	  local_apurelm[ ishift+1].re=0.0;
	  local_apurelm[ ishift+1].im=0.0;

          ishift += 2;

	  for( i=2; i<lmax1; i++, ishift++)
	    {
	        local_apurelm[ ishift].re += local_alm[ ishift].re;
	        local_apurelm[ ishift].im += local_alm[ ishift].im;
	    }
	}
    }

    return( 1);   
}

s2hat_int4 purealm_spin2_update_healpix( s2hat_int4 nmaps, s2hat_int4 nmvals, s2hat_int4 lmax, 
                                         s2hat_dcomplex *local_alm, s2hat_dcomplex *local_apurelm)


/* performs the spin2 update of pure alm in the s2hat format *
 * called by the driver: purealm_spin0_update                */

{
  s2hat_int4 i, imap, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di;

  lmax1 = lmax+1;

  for( imap=0; imap<nmaps; imap++)
    {

      for( j=0; j<nmvals; j++)
	{

          ishift = nstokes*(imap*nmvals+j)*lmax1;

          /*  l = 0 and l = 1 modes are zero */

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;
	  
	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;

	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;
	  
	  local_apurelm[ ishift].re=0.0;
	  local_apurelm[ ishift++].im=0.0;

	  for( i=2; i<lmax1; i++)
	    {

  	        /* Q-map */ 

	        local_apurelm[ ishift].re += local_alm[ ishift].re;
	        local_apurelm[ ishift].im += local_alm[ ishift].im;

                ishift++;

                /* U-map */

	        local_apurelm[ ishift].re += local_alm[ ishift].re;
	        local_apurelm[ ishift].im += local_alm[ ishift].im;

                ishift++;

	    }
	}
    }

    return( 1);
}

/* map apodization routines */

s2hat_int4 apodize_maps_complex( s2hat_int4 nmaps, s2hat_int4 mapsize, s2hat_flt8 *local_cmbmap, s2hat_int4 wstride, s2hat_flt8 *local_map)

/*  Combines the maps with complex windows :                                                                   *
 *  the window function stored on the input in local_map is overwritten on the output with (Q+iU)*conj(Window) *
 * wstride = 0 if nwindows = 1     *
 * wstride = 1 if nwindows = nmaps */

{

  s2hat_int4 i, imap, ishift, ishift1, kshift, kshift1, nstokes = 2;
  s2hat_flt8 tmp;

  for( imap=nmaps-1; imap>=0; imap--)
    {
      ishift = imap*nstokes*mapsize; ishift1 = ishift+mapsize;
      kshift = ishift*wstride;       kshift1 = kshift+mapsize;

      for( i=0; i<mapsize; i++, ishift++, ishift1++, kshift++, kshift1++)
	{
           tmp = local_map[ kshift];
	   local_map[ ishift] = local_cmbmap[ishift]*local_map[kshift]+local_cmbmap[ishift1]*local_map[ kshift1];       /* real part */
	   local_map[ ishift1] = local_cmbmap[ishift1]*tmp-local_cmbmap[ ishift]*local_map[ kshift1];                   /* imaginary part */
	}
    }

  return( 1);
}

s2hat_int4 apodize_maps_all( s2hat_int4 nmaps, s2hat_int4 mapsize, s2hat_flt8 *local_cmbmap, s2hat_int4 winstride, s2hat_flt8 *local_w_scal, s2hat_flt8 *local_w_vect, s2hat_flt8 *local_w_tens,
		       s2hat_flt8 *local_pXscal, s2hat_flt8 *local_pXvect, s2hat_flt8 *local_pXtens)

     /* apodizes all the Q/U maps with a set of scalar, vector and tensor windows */

{
  int i, imap, ishift, ishift1, jshift, wstride, nstokes = 2;

  wstride = winstride ? 1 : 0;
  
  for( imap = 0; imap < nmaps; imap++)
    {
      ishift = imap*nstokes*mapsize;      jshift = imap*mapsize*wstride;

      for( i = 0; i < mapsize; i++) local_pXscal[ ishift++] = local_w_scal[jshift++];
      for( i = 0; i < mapsize; i++) local_pXscal[ ishift++] = 0.0;
    }

  apodize_maps_complex( nmaps, mapsize, local_cmbmap, wstride, local_pXscal);

  for( imap = 0; imap < nmaps; imap++)
    {
      ishift = imap*nstokes*mapsize; jshift = ishift*wstride;

      for( i = 0; i < nstokes*mapsize; i++) local_pXvect[ ishift++] = local_w_vect[jshift++];
    }

  apodize_maps_complex( nmaps, mapsize, local_cmbmap, wstride, local_pXvect);

  for( imap = 0; imap < nmaps; imap++)
    {
      ishift = imap*nstokes*mapsize; jshift = ishift*wstride;

      for( i = 0; i < nstokes*mapsize; i++) local_pXtens[ ishift++] = local_w_tens[jshift++];
    }

  apodize_maps_complex( nmaps, mapsize, local_cmbmap, wstride, local_pXtens);

  return( 1);

}

s2hat_int4 combine_mask_window( s2hat_int4 nmaps, s2hat_int4 mapsize, s2hat_int4 mstride, s2hat_int4 *local_mask, s2hat_int4 wstride, s2hat_flt8 *local_map)

{
  s2hat_int4 imap, ipix, ishift, ishift1, jshift, kshift, kshift1, wmstride, nstokes = 2;

  wmstride = (wstride) || (mstride) ? 1 : 0;

  for( imap=(nmaps-1)*wmstride; imap>=0; imap--)
    {

      ishift = imap*nstokes*mapsize; 
      ishift1 = ishift+mapsize;

      jshift = mstride*ishift;
     
      kshift = wstride*ishift;
      kshift1 = kshift+mapsize;

      for( ipix=0; ipix<mapsize; ipix++)
        {
	   local_map[ ishift++] = local_map[ kshift++]*local_mask[ jshift];
	   local_map[ ishift1++] = local_map[ kshift1++]*local_mask[ jshift++];
	}

    }

  return( 1);
}

/* windows in harmonic domain */

s2hat_int4 wlm_scalar2vector_s2hat( s2hat_int4 nwins, s2hat_int4 nmvals, s2hat_int4 lmax, s2hat_dcomplex *local_wlm)

{
  s2hat_int4 i, iwin, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di, sqrtfact;

  lmax1 = lmax+1;

  for( iwin=0; iwin<nwins; iwin++)
    {
  
      for( j=0; j<nmvals; j++)
       {

         ishift = (iwin*nstokes*nmvals+j)*lmax1;

         /* set monopole to zero */
	 local_wlm[ ishift].re=0.0;
         local_wlm[ ishift+lmax1*nmvals].re=0.0; 
      
         local_wlm[ ishift].im=0.0;
         local_wlm[ ishift+lmax1*nmvals].im=0.0;
      
         ishift++;

         for( i=1; i<lmax1; i++, ishift++)
	   {

             di = (double)i;

             sqrtfact = sqrt(di*(di+1.0));

	     /*real part of wlm*/
	     local_wlm[ ishift].re *= sqrtfact;

	     local_wlm[ lmax1*nmvals+ishift].re=0.0;  /* because derived from a REAL, SCALAR window function - already set */

	     local_wlm[ lmax1*nmvals+ishift].re *= sqrtfact;

	     /*imaginary part of wlm*/
	     local_wlm[ ishift].im *= sqrtfact;

	     local_wlm[ lmax1*nmvals+ishift].im=0.0;  /* because derived from a REAL, SCALAR window function - already set */

	     local_wlm[ lmax1*nmvals+ishift].im *= sqrtfact;

	  }
       }
    }
}

s2hat_int4  wlm_vector2tensor_s2hat( s2hat_int4 nwins, s2hat_int4 nmvals, s2hat_int4 lmax, s2hat_dcomplex *local_wlm)

{
  s2hat_int4 i, iwin, ishift, j, lmax1, nstokes = 2;
  s2hat_flt8 di, sqrtfact;

  lmax1 = lmax+1;

  for( iwin=0; iwin<nwins; iwin++)
    {

      for( j=0; j<nmvals; j++)
        {

           ishift = (iwin*nstokes*nmvals+j)*lmax1;

           /* l = 0 : */
	   /* real part of wlm */
	   local_wlm[ ishift].re=0.0;
	   local_wlm[ ishift+lmax1*nmvals].re=0.0;
							     
           /* imaginary part of wlm */
	   local_wlm[ ishift].im=0.0;
           local_wlm[ ishift+lmax1*nmvals].im=0.0;

           /* l = 1 : */
           /* real part of wlm */
	   local_wlm[ ishift+1].re=0.0;
	   local_wlm[ ishift+lmax1*nmvals+1].re=0.0;
      
	   /* imaginary part of wlm */
           local_wlm[ ishift+1].im=0.0;
           local_wlm[ ishift+lmax1*nmvals+1].im=0.0;

           ishift += 2;

           for( i=2; i<lmax1; i++, ishift++)
	     {

               di = (double)i;
               sqrtfact = sqrt((di-1.0)*(di+2.0));   // only missing (in the vectorial window) factors - full factor: sqrt((di-1.0)*di*(di+1.0)*(di+2.0))

	           /*real part of wlm*/
	           local_wlm[ ishift].re *= sqrtfact;
               local_wlm[ lmax1*nmvals+ishift].re=0.0; /*because derived from a REAL, SCALAR window function*/

               local_wlm[ lmax1*nmvals+ishift].re *= sqrtfact;
	  
	           /*imaginary part of wlm*/
	           local_wlm[ ishift].im *= sqrtfact;
	           local_wlm[lmax1*nmvals+ishift].im = 0.0; /* because derived from a REAL, SCALAR window function*/

	           local_wlm[lmax1*nmvals+ishift].im *= sqrtfact;
	     }
	}
    }
}
