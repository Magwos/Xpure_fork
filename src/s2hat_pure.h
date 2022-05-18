
/* version of January, 01, 2010 - rs@apc */

s2hat_int4 window_scalar2all( s2hat_pixeltype, s2hat_scandef, s2hat_int4, s2hat_int4, s2hat_int4, s2hat_int4*, s2hat_int4, s2hat_int4, 
								    s2hat_int4, s2hat_flt8*, s2hat_int4, s2hat_int4, s2hat_int4*, s2hat_flt8*, s2hat_flt8*, s2hat_flt8*, 
								    s2hat_int4, s2hat_int4, MPI_Comm);

s2hat_int4 s2hat_apodizedmaps2purealm( s2hat_pixeltype, s2hat_scandef, s2hat_int4 , s2hat_int4, s2hat_int4, s2hat_int4*, s2hat_int4, s2hat_int4, 
									   s2hat_int4, s2hat_flt8*, s2hat_int4, s2hat_flt8*, s2hat_flt8*, s2hat_flt8*, s2hat_int4, s2hat_dcomplex*, 
									   s2hat_int4, s2hat_int4, MPI_Comm);

s2hat_int4 s2hat_map2purealm( s2hat_pixeltype, s2hat_scandef, s2hat_int4, s2hat_int4, s2hat_int4, s2hat_int4*, s2hat_int4, s2hat_int4, s2hat_int4, 
						      s2hat_flt8*, s2hat_int4, s2hat_int4, s2hat_int4*, s2hat_int4, s2hat_flt8*, s2hat_flt8*, s2hat_int4, s2hat_dcomplex*, 
							  s2hat_int4, s2hat_int4, MPI_Comm);

void apodize_maps_all( s2hat_int4, s2hat_int4, s2hat_flt8*, s2hat_int4, s2hat_flt8*, s2hat_flt8*, s2hat_flt8*, s2hat_flt8*, s2hat_flt8*, s2hat_flt8*); 
