#include "stdio.h"
#include "stdlib.h"
/* #include "sys/types.h" */
#include "malloc.h"
#include "math.h"
#include "mpi.h"
#include "s2hat.h"

/* #define s2hat_pixeltype pixeltype */
/* #define s2hat_scandef scandef */
/* #define s2hat_pixparameters pixparameters */
/* #define s2hat_int4 int4 */
/* #define s2hat_int8 int8 */
/* #define s2hat_flt8 flt8 */


s2hat_int4 compute_all_xls( s2hat_int4 nmaps, s2hat_int4 noStokes, s2hat_int4 nlmax, s2hat_int4 nmvals, s2hat_int4 *mvals, s2hat_int4 lda,
			    s2hat_dcomplex *local_alm, s2hat_int4 nxtype, s2hat_flt8 *xcl, s2hat_int4 *frstSpec, s2hat_int4 *scndSpec, s2hat_int4 gangSize, 
			    s2hat_int4 n_gangs, s2hat_int4 my_gang_no, s2hat_int4 my_gang_rank, s2hat_int4 gang_root, MPI_Comm my_gang_comm, MPI_Comm global_comm)
     
/* 
  output: xcl: a s2hat_flt8 vector of cross spectra of the length (nlmax+1)*nxtype*(nxspec*nmaps^2-((nmaps-1)*nmaps)/2), where nxspec is implicit 
               and equal to 
                    n_gangs/2+1 for my_gang_no < n_gangs/2 and n_gangs/2 otherwise, if n_gangs is even
               and to
                    n_gangs/2 if n_gangs is odd.
               nxtype is defined on the input and determines how many different types of cross-spectra are to be computed, i.e., 
	        nxtype = 3 => (TT, EE, BB);
		       = 5 => (TT, EE, BB, TE, ET);
		       = 9 => (TT, EE, BB, TE, ET, TB, BT, EB, BE).

		frstSpec, scndSpec: store the ids of the maps used for the cross_spectra computation. Their length is given by nxspec as defined
               above and equal to : nxspec*nmaps^2-((nmaps-1)*nmaps)/2
		   
	        Within each gang the output is stored on the proc "gang_root" only.

  fixed the case with n_gangs == 1 - 16/03/2007 rs@apc

  generalized to accept odd n_gangs - 12/07/2010 rs@apc

*/

{

 s2hat_int4 int_zero = 0, int_one = 1; 
 s2hat_int4 evenOdd, i, ig, im, jm, ip, jg, ic, from_rank, to_rank, my_rank, my_nalm, nbcasts, nMidPoint, numprocs, currentSpec, commLength;
 s2hat_int4 *xls_ranks, *bcast_root;

 s2hat_dcomplex *imported_alm;

 // MPI defs
 MPI_Comm *xls_comm;
 MPI_Group global_group;
 MPI_Group xls_group;
 MPI_Status mpistatus;

 numprocs = gangSize*n_gangs;

 if( n_gangs == 1) {
   currentSpec = 0;
   for( im=0; im<nmaps; im++) {
     for( jm=0; jm<=im; jm++) {
	
	collect_xls( nmaps, jm, nmaps, im, noStokes, nlmax, nmvals, mvals, lda,
		     local_alm, local_alm, nxtype, &xcl[ (nlmax+1)*nxtype*currentSpec],
		     my_gang_rank, numprocs, gang_root, my_gang_comm);
	
	if( my_gang_rank == gang_root) {
	  frstSpec[ currentSpec] = im;
	  scndSpec[ currentSpec] = jm;
	  
	  currentSpec++;
	}

     }      
   }

   return( currentSpec);
 }


 my_nalm = (nlmax+1)*nmvals*noStokes;  /* note that it is **the same** on corresponding procs of each gang */

 /* define the communicators for Bcasts */
 MPI_Comm_group( global_comm, &global_group);
	
 xls_comm = (MPI_Comm *)calloc( n_gangs*gangSize, sizeof( MPI_Comm));
 bcast_root = (s2hat_int4 *) calloc( n_gangs*gangSize, sizeof( s2hat_int4));

 nMidPoint = (n_gangs+1)/2;

 evenOdd = (n_gangs == 2*(n_gangs/2)) ? 1 : 0; 

 for( ig = 0; ig < n_gangs; ig++) {

   for( ip = 0; ip < gangSize; ip++) {
		
	    commLength = (ig < nMidPoint) ? nMidPoint : n_gangs/2;   // changes nothing if n_gangs even

	    xls_ranks  = (s2hat_int4 *) calloc( commLength, sizeof( s2hat_int4));
		
	    for( jg = 0; jg < commLength; jg++)
		xls_ranks[ jg] = ip+gangSize*((ig+jg)%n_gangs);   // global proc numbs

		MPI_Group_incl( global_group, commLength, xls_ranks, &xls_group); 
		MPI_Comm_create( global_comm, xls_group, &xls_comm[ ig*gangSize+ip]);

		if( (my_gang_rank == ip) && ( my_gang_no == ig)) MPI_Comm_rank( xls_comm[ ig*gangSize+ip], &my_rank);

		MPI_Bcast( &my_rank, 1, MPI_INT, ip+ig*gangSize, global_comm); 

		bcast_root[ ig*gangSize+ip] = my_rank;

		free( xls_ranks);		

   }
 }

 nbcasts = (n_gangs+1)/2;                                     // number of (simultaneous) Bcasts = nMidPoint

 currentSpec = 0;  
 for( im = 0; im < nmaps; im++) {  
   for( ic = 0; ic < nbcasts; ic++) {

         imported_alm = (s2hat_dcomplex *)calloc( my_nalm, sizeof( s2hat_dcomplex));

	  if( (my_gang_no == ic) || ((my_gang_no == ic+nMidPoint))) {

	      for( i = 0; i < my_nalm; i++) { 
		   imported_alm[ i].re = local_alm[ my_nalm*im+i].re;
		   imported_alm[ i].im = local_alm[ my_nalm*im+i].im;					
	      }
         } 

        if( (my_gang_no-ic+n_gangs)%n_gangs < nbcasts) {

	      MPI_Bcast( imported_alm, my_nalm, MPI_DOUBLE_COMPLEX, bcast_root[ic*gangSize+my_gang_rank], xls_comm[ ic*gangSize+my_gang_rank]);

        } else {
			 
	      if( (ic != nbcasts-1) || evenOdd)
	          MPI_Bcast( imported_alm, my_nalm, MPI_DOUBLE_COMPLEX, bcast_root[(ic+nMidPoint)*gangSize+my_gang_rank], xls_comm[ (ic+nMidPoint)*gangSize+my_gang_rank]);

        }

        for( jm = 0; jm < nmaps; jm++) {

   	      // and compute x-spectra here ...
	      if( ((my_gang_no != ic) && ((my_gang_no != ic+nMidPoint))) || ( im >= jm)) {                            // avoid recomputing redundant xspectra on the bcasting gangs
		                                                                                                          // computation only on receiving gangs unless im >= jm
	            collect_xls( nmaps, jm, int_one, int_zero, noStokes, nlmax, nmvals, mvals, lda,
	                         local_alm, imported_alm, nxtype, &xcl[ (nlmax+1)*nxtype*currentSpec],
	                         my_gang_rank, numprocs, gang_root, my_gang_comm);

	            // collect xspectra ids (in a global map numbering scheme) as well
	            if( (my_gang_no-ic+n_gangs)%n_gangs < nbcasts) {                         // my gang is a part of the 1st of the two simultaneous bcasts send by gang = ic

			if( my_gang_rank == gang_root) {
			    frstSpec[ currentSpec] = jm+my_gang_no*nmaps;                    // local
			    scndSpec[ currentSpec] = im+ic*nmaps;                            // imported
			}
			currentSpec++;
			   
	            } else {                                                                // my_gang is a part of the 2nd of the two simultaneous bcasts send by gang = ic+nMidPoint

			if( (ic != nbcasts-1) || evenOdd) {
			     if( my_gang_rank == gang_root) {
				 frstSpec[ currentSpec] = jm+my_gang_no*nmaps;              // local
				 scndSpec[ currentSpec] = im+(ic+nMidPoint)*nmaps;          // imported
			     }
			     currentSpec++;
			}
		    }
	      }
	 }

        free( imported_alm);

    }
 }


 free( bcast_root);

 for( ic = 0; ic < nbcasts; ic++)
 {

   if( (my_gang_no-ic+n_gangs)%n_gangs < nbcasts) {                                 // 1st of the pair of bcast communicators

	  MPI_Comm_free( &xls_comm[ ic*gangSize+my_gang_rank]);

   } else {                                                                        // 2nd of the pair 

	  if( (ic != nbcasts-1) || evenOdd)
	      MPI_Comm_free( &xls_comm[ (ic+nMidPoint)*gangSize+my_gang_rank]);

     }
 }

 free( xls_comm);

 // the send/recv part at the end
		   
 for( im = 0; im < nmaps; im++) {

     if( my_gang_no >= nMidPoint) {       // gangs from the second half send to ...

	   my_rank = my_gang_rank+gangSize*my_gang_no;
	   to_rank = my_rank-gangSize*nMidPoint;

	   MPI_Send( local_alm, my_nalm, MPI_DOUBLE_COMPLEX, to_rank, my_rank, global_comm);

     } else {                             // the gangs from the first half

	   if( (my_gang_no != (n_gangs/2)) || evenOdd) {    // the gang in the middle (if exits : n_gangs odd) does nothing

	       imported_alm = (s2hat_dcomplex *)calloc( my_nalm, sizeof( s2hat_dcomplex));
		  
	       from_rank = my_gang_rank+gangSize*(my_gang_no+nMidPoint);

	       MPI_Recv( imported_alm, my_nalm, MPI_DOUBLE_COMPLEX, from_rank, from_rank, global_comm, &mpistatus);
		  		  
	       for( jm = 0; jm < nmaps; jm++) {

	            // and compute x-spectra here ...
	            collect_xls( nmaps, jm, int_one, int_zero, noStokes, nlmax, nmvals, mvals, lda,
		                 local_alm, imported_alm, nxtype, &xcl[ (nlmax+1)*nxtype*currentSpec],
		                 my_gang_rank, numprocs, gang_root, my_gang_comm);

	           // collect xspectra id as well
	           if( my_gang_rank == gang_root) {
	               frstSpec[ currentSpec] = my_gang_no*nmaps+jm;             // local
	               scndSpec[ currentSpec] = (from_rank/gangSize)*nmaps+im;   // imported
		   }
			  
	           currentSpec++;
	       }

              free( imported_alm);
	   }
     }
 }

 return( currentSpec);

}


s2hat_int4 compute_all_xls_numbers( s2hat_int4 nmaps, s2hat_int4 *frstSpec, s2hat_int4 *scndSpec, s2hat_int4 gangSize, s2hat_int4 n_gangs, s2hat_int4 my_gang_no, 
                                   s2hat_int4 my_gang_rank, s2hat_int4 gang_root)

/*
               computes frstSpec, scndSpec, which store the ids of the maps used for the cross_spectra computation. They should be 
               allocated prior to calling this routine and their length is nxspec*nmaps^2-((nmaps-1)*nmaps)/2 where nxspec is
               equal to 
                    n_gangs/2+1 for my_gang_no < n_gangs/2 and n_gangs/2 otherwise, if n_gangs is even
               and to
                    n_gangs/2 if n_gangs is odd.

               The computed values are consistent with what is computed by compute_xls_all routine. 

               - 12/07/2010 rs@apc
*/

{
 s2hat_int4 evenOdd, i, ig, im, jm, ip, jg, ic, from_rank, nbcasts, nMidPoint, numprocs, currentSpec;

 numprocs = gangSize*n_gangs;

 if( n_gangs == 1) {
   currentSpec = 0;
   for( im=0; im<nmaps; im++) {
     for( jm=0; jm<=im; jm++) {
	
	if( my_gang_rank == gang_root) {
	  frstSpec[ currentSpec] = im;
	  scndSpec[ currentSpec] = jm;
	  
	  currentSpec++;
	}

     }      
   }

   return( currentSpec);
 }

 nMidPoint = (n_gangs+1)/2;

 evenOdd = (n_gangs == 2*(n_gangs/2)) ? 1 : 0; 

 nbcasts = (n_gangs+1)/2;                                     // number of (simultaneous) Bcasts = nMidPoint

 currentSpec = 0;  
 for( im = 0; im < nmaps; im++) {  
   for( ic = 0; ic < nbcasts; ic++) {

        for( jm = 0; jm < nmaps; jm++) {

   	      // and compute x-spectra here ...
	      if( ((my_gang_no != ic) && ((my_gang_no != ic+nMidPoint))) || ( im >= jm)) {                            // avoid recomputing redundant xspectra on the bcasting gangs
		                                                                                                      // computation only on receiving gangs unless im >= jm
	            // collect xspectra ids (in a global map numbering scheme) as well
	            if( (my_gang_no-ic+n_gangs)%n_gangs < nbcasts) {                         // my gang is a part of the 1st of the two simultaneous bcasts send by gang = ic

			if( my_gang_rank == gang_root) {
			    frstSpec[ currentSpec] = jm+my_gang_no*nmaps;                    // local
			    scndSpec[ currentSpec] = im+ic*nmaps;                            // imported
			}
			currentSpec++;
			   
	            } else {                                                                // my_gang is a part of the 2nd of the two simultaneous bcasts send by gang = ic+nMidPoint

			if( (ic != nbcasts-1) || evenOdd) {
			     if( my_gang_rank == gang_root) {
				 frstSpec[ currentSpec] = jm+my_gang_no*nmaps;              // local
				 scndSpec[ currentSpec] = im+(ic+nMidPoint)*nmaps;          // imported
			     }
			     currentSpec++;
			}
		    }
	      }
	 }      
    }
 }

 // the send/recv part at the end
		   
 for( im = 0; im < nmaps; im++) {

     if( my_gang_no < nMidPoint) {                          // the gangs from the first half

	   if( (my_gang_no != (n_gangs/2)) || evenOdd) {    // the gang in the middle (if exits : n_gangs odd) does nothing
			 
	       from_rank = my_gang_rank+gangSize*(my_gang_no+nMidPoint);
		  		  
	       for( jm = 0; jm < nmaps; jm++) {

	           // collect xspectra id
	           if( my_gang_rank == gang_root) {
	               frstSpec[ currentSpec] = my_gang_no*nmaps+jm;             // local
	               scndSpec[ currentSpec] = (from_rank/gangSize)*nmaps+im;   // imported
		   }
			  
	           currentSpec++;
	       }
	   }
     }
 }

 return( currentSpec);

}
