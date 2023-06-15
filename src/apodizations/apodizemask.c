// Copyright (C) 2007, 2008 Marc Betoule

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// report bugs at betoule@apc.univ-paris7.fr


#include <stdlib.h>
#include <math.h>
#include "fitshandle.h"
#include <vector>

#include "healpix_map.h"
#include "healpix_map_fitsio.h"
#include "fitsio.h"

using namespace std;

/*! return the angular distance between 2 points on the sphere*/
inline double angdist(pointing p1, pointing p2){
  /*double gamma = p2.phi - p1.phi; 
    return acos(cos(p1.theta)*cos(p2.theta) + sin(p1.theta)*sin(p2.theta)*cos(gamma));*/
  vec3 v1 = p1.to_vec3();
  vec3 v2 = p2.to_vec3();
  return acos(dotprod(v1,v2));
}

void usage(){
  cout <<
    "\nUsage:\n"
    "  apodizemask <input file> <output file (\"null\" = no output> [-distout <distance map file>] \n"
    " [-radius <int (arcmin, default : 60)>] [-distin <distance map file>][-pixbool <majority pixel value (0 or 1)>][-inside] \n"
    //   " [-spline <int order> <double coef1> <double coef2> ... \n\n";
   " [-spline]\n\n";
  exit(1);
 }
// Apply a spline function to the distance map
void apodize_spline_quintic(const Healpix_Map<double> & distmap, Healpix_Map<float> & mask, double radius,bool pixbool = 0);
/*! Apply a spline function to the distance map*/
//void apodize_spline(const Healpix_Map<double> & distmap, Healpix_Map<float> & mask, double radius, vector<double> & coef);
/*! Apply a cosin function to the distance map*/
void apodize(const Healpix_Map<double> & distmap, Healpix_Map<float> & mask, double radius=60,bool pixbool = 0,bool inside=0);
/*! Compute the actual distance between each pixel centre and the mask, as long as this distance
 * is less than radius
 */
void computeDistance_in(const Healpix_Map<float> & mask, Healpix_Map<double> & distmap, double radius = 60,bool pixbool = 0);
void computeDistance_out(const Healpix_Map<float> & mask, Healpix_Map<double> & distmap, double radius = 60,bool pixbool = 0);

int read_key_from_fits (string file, char* keyname);
void write_key_to_fits   (string file, char* keyname, char* comment ,int value);

int main(int argc, char ** argv){
  string infile      = "";
  string outfile     = "";
  string filedistmap = "";
  bool distout   = false, distin = false;
  double radius  = 60;
  bool spline    = false;
  bool pixbool   = 0;
  bool inside    = false;
  char ** endptr = 0;
  //args processing
  if (argc < 3)    usage();
  

  // update argument with command line inputs
  infile  = argv[1];
  outfile = argv[2];
  if (argc > 3){
    int m = 3;
    while(m<argc){
      string arg = argv[m];
      if      (arg == "-distout") {filedistmap = argv[++m];++m; distout = true;}
      else if (arg == "-distin")  {filedistmap = argv[++m];++m; distin = true;}
      else if (arg == "-radius")  {radius = strtod(argv[++m],endptr);++m;}
      else if (arg == "-spline")  {spline = true;	++m;      }
      else if (arg == "-minpix") {pixbool = atoi(argv[++m]);++m; }
      else if (arg == "-inside")    {inside = true;++m; }
      else{cout << "\nWarning : ignoring argument "<< arg <<" (unknown)\n"; ++m;}
    }
  }
  
  // convert radius from degre to radian
  radius = radius/60*degr2rad;
  printf("radius is %f", radius);
  
  //read input mask
  Healpix_Map <float> mask;
  if(!distin)
    read_Healpix_map_from_fits(infile, mask,1);
  

  //compute distance map
  Healpix_Map <double> distmap;
  if(!distin) { //allocate memory for intermediate results 
    distmap = Healpix_Map<double>(mask.Nside(), mask.Scheme(), SET_NSIDE);
    distmap.fill(2*pi); // distance map is 2pi outside of the mask and 0 inside
    for(int i = 0; i<mask.Npix(); i++){
      if(mask[i] == pixbool) distmap[i] = 0;}
    printf("inside is %i\n", inside);
    if (inside)
      computeDistance_in(mask, distmap, radius, pixbool);  // compute the distance map
    else
      computeDistance_out(mask, distmap, radius, pixbool); 
  }
  else //read the input distance map
    {read_Healpix_map_from_fits(filedistmap, distmap,1);
      // load options of distance computation 
      pixbool = read_key_from_fits (filedistmap,"MINPIX" );
      inside  = read_key_from_fits (filedistmap,"INSIDE" );
    }
  

  // apodize mask wrt distance map
  fitshandle outfits;
  if(outfile != "null"){
    // check that the radius of the input distance map is compatible
    double min, max;
    distmap.minmax(min,max);
    double pixsize = sqrt(pi/(3*distmap.Nside()*distmap.Nside()));
    if(max < radius-pixsize){
      cout << "radius too large for the input dist map\n";
      exit(-1);
    }
    mask = Healpix_Map<float>(distmap.Nside(), distmap.Scheme(), SET_NSIDE);
    // call the right function to apodize
    if(spline) apodize_spline_quintic(distmap, mask, radius, pixbool);    
    else       apodize(distmap, mask, radius, pixbool,inside);
    //writing results
    outfits.create (outfile);
    write_Healpix_map_to_fits (outfits,mask,FITSUTIL<float>::DTYPE);
  }

  // save distance map
  if(distout && !distin){
    // for(int i =0; i<distmap.Npix(); i++)
    //  if(distmap[i] > radius) distmap[i] = Healpix_undef;
    fitshandle outfits2;
    outfits2.create (filedistmap);
    write_Healpix_map_to_fits (outfits2,distmap,FITSUTIL<double>::DTYPE);
    
    // save options of distance computation in fits
    char* keyname = "minpix";
    char* comment = "minority pixel value";
    write_key_to_fits (filedistmap, keyname, comment,pixbool);
    keyname = "inside";
    comment = "apodization direction";
    write_key_to_fits (filedistmap, keyname, comment,(int)inside);

  }
  
  return 0;
}

// Distance map : starting value is 2*pi outside of the mask and 0 inside
// To apodize outside : for each point inside the mask, get all pixels inside a disk of radius "radius"
// If the disk cross the border of the mask, update the distance value of all pixel outside of the mask 
// which are close to the border with the minimum distance value to the border 
void computeDistance_out(const Healpix_Map<float> & mask, Healpix_Map<double> & distmap, double radius,bool pixbool){
  vector<int> listpix;
  for(int i = 0; i<mask.Npix();i++){
    // for each point in the mask
    if(mask[i] == pixbool){
      pointing pixcenter = mask.pix2ang(i);
      // get a disk
      mask.query_disc(pixcenter ,radius, listpix);
      for(vector<int>::iterator p = listpix.begin(); p!=listpix.end(); p++){
	double v = distmap[*p];
	// for each point of the disk outside of the mask
      	if(v != 0){
      	  pointing p2 = mask.pix2ang(*p);
	  // get the distance
      	  double da = angdist(pixcenter,p2);
	  // get the minimum distance
      	  distmap[*p] = v<da?v:da; 
	}
      }
    }
  }
}

// Distance map : starting value is 2*pi outside of the mask and 0 inside
// To apodize inside : for each point inside the mask, get all pixels inside a disk of radius "radius"
// If the disk cross the border of the mask, update the distance value of all pixel inside of the mask 
// which are close to the border with the minimum distance value to the border 
void computeDistance_in(const Healpix_Map<float> & mask, Healpix_Map<double> & distmap, double radius,bool pixbool){
  vector<int> listpix;
  double mind;
  for(int i = 0; i<mask.Npix();i++){
    // for each point in the mask
    if(mask[i] == pixbool){
      pointing pixcenter = mask.pix2ang(i);
      // get a disk
      mask.query_disc(pixcenter ,radius, listpix);
      // maximum value of the distance map is radius
      mind = 2*pi;//radius;
      for(vector<int>::iterator p = listpix.begin(); p!=listpix.end(); p++){
	double v = distmap[*p];
	// for each point of the disk outside of the mask
      	if(v==2*pi){
      	  pointing p2 = mask.pix2ang(*p);
	  // get the distance
      	  double da = angdist(pixcenter,p2);
	  // get the minimum distance
	  if (da<mind) mind = da;
	}
      }
      // update the distance value if a minimum is found
      //      distmap[i] = mind==radius?0:mind;
      distmap[i] = mind>radius?0:mind;
    }
  }
}

void apodize(const Healpix_Map<double> & distmap, Healpix_Map<float> & mask, double radius, bool pixbool,bool inside){
  double omega = pi/radius;
  int    sign_coef ;
 
  if (pixbool) sign_coef = 1;  // Transition from 0 to 1 if pixbool is 1 
  else         sign_coef = -1; // Transition from 1 to 0 if pixbool is 0
  if (inside) sign_coef = - sign_coef; // invert the transition to apodize inside
  double oradius =  radius;//60/60*degr2rad;  
  for(int i = 0; i<distmap.Npix(); i++)
    {
      if ((distmap[i] >= oradius) || (approx(distmap[i],Healpix_undef))) mask[i] = 1-pixbool;
      //if (approx(distmap[i],Healpix_undef)) mask[i] = 1-pixbool;
      else if ((inside) && (distmap[i]>=radius)) mask[i] = pixbool;
      else if ((!inside) && (distmap[i]>=radius)) mask[i] = 1-pixbool;
      else if (distmap[i] ==0) mask[i] = pixbool;
      else{
	mask[i] = 0.5 + sign_coef*(0.5* cos(omega * distmap[i]));
      }
    }
}

/*void apodize_spline(const Healpix_Map<double> & distmap, Healpix_Map<float> & mask, double radius, vector<double> & coef){
  int o = coef.size() -1;
  for(int i = 0; i < distmap.Npix(); i++){
    if(distmap[i] > radius || approx(distmap[i],Healpix_undef)) mask[i] = 1;
    else{
      //Map [0 radius] to [-1 1]
      double u = 1. - 2. * (distmap[i] / radius);
      double u2 = u * u;
      
      //evaluate the spline at distmap[i]
      mask[i] = coef[o] * u;
      for(int j = o-1 ; j == 0; j--){
	u *= u2;
	mask[i] += coef[j] * u;
      }
      
      //rescale to live in [0 1]
      mask[i] = (1. + mask[i]) / 2.;
    }
  }
  }*/
void apodize_spline_quintic(const Healpix_Map<double> & distmap, Healpix_Map<float> & mask, double radius,bool pixbool){
  for(int i = 0; i < distmap.Npix(); i++){
    if(distmap[i] >= radius || approx(distmap[i],Healpix_undef)) mask[i] = 1;
    else
      {
      double x  = 1 - 2*(distmap[i]/radius);// maps [ 0 radius ] to [ -1 1 ] 
      double x2 = x * x ;
      
      double f  =  15 * x ;  x *= x2 ;
      f += -10 * x ;  x *= x2 ;
      f +=  3  * x ;
      
      f = f / 8. ;
      
      f = (1 + f ) / 2. ; //rescaling for f to live in [ 0 1 ]
      mask[i] = 1 - f;
      //f ( (y<0) )  = 1 ; // and now 
      //f ( (y>1) )  = 0 ;
    }
  }
}

int read_key_from_fits (string file, char* keyname)
{
  int value;
  int status = 0;  int hdutype ;
  fitsfile *fptr;
  char comment[FLEN_COMMENT];
  if (fits_open_file(&fptr, file.c_str(), READWRITE, &status))
    {
      fprintf(stderr, "%s (%d): Cannot open the FITS file!\n", __FILE__, __LINE__ );
      fits_report_error(stderr, status);
    }
      
  if (fits_movabs_hdu(fptr, 2, &hdutype, &status))
    {
      fprintf(stderr, "%s (%d): Cannot move to HDU2.\n", __FILE__, __LINE__ );
      fits_report_error(stderr, status);
    }
  
  if (fits_read_key (fptr, TINT, keyname, &value, comment , &status))
    {
      fprintf(stderr, "%s (%d): Cannot read the key %s.\n", __FILE__, __LINE__ ,keyname);
      fits_report_error(stderr, status);
    }

  if (fits_close_file(fptr, &status))
    {  
      fprintf(stderr, "%s (%d): Cannot close the FITS file!\n", __FILE__, __LINE__ );
      fits_report_error(stderr, status);
    }
  return (value);

}

void write_key_to_fits  (string file, char* keyname, char* comment ,int value)
{
    
  int status = 0;  int hdutype ;
  fitsfile *fptr;
  if (fits_open_file(&fptr, file.c_str(), READWRITE, &status))
    {
      fprintf(stderr, "%s (%d): Cannot open the FITS file!\n", __FILE__, __LINE__ );
      fits_report_error(stderr, status);
    }
      
  if (fits_movabs_hdu(fptr, 2, &hdutype, &status))
    {
      fprintf(stderr, "%s (%d): Cannot move to HDU2.\n", __FILE__, __LINE__ );
      fits_report_error(stderr, status);
    }
  
  if (fits_write_key (fptr, TINT, keyname, &value, comment , &status))
    {
      fprintf(stderr, "%s (%d): Cannot write the key %s.\n", __FILE__, __LINE__ ,keyname);
      fits_report_error(stderr, status);
    }

  if (fits_close_file(fptr, &status))
    {  
      fprintf(stderr, "%s (%d): Cannot close the FITS file!\n", __FILE__, __LINE__ );
      fits_report_error(stderr, status);
    }
  
}
