#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
import sys
import waflib.Tools
import os.path as path
APPNAME = 'xpure'
VERSION = '1.0'

top = '.'
out = 'build'


def options(ctx):
    ctx.load('compiler_c')
    ctx.load('compiler_fc')
    ctx.add_option('--platform', action='store', help='Platform name')
    ctx.add_option('--s2hat', action='store', help='s2hat location',default='../s2hat')
    ctx.add_option('--healpix', action='store', help='healpix location',default='../healpix')
    ctx.add_option('--cfitsio', action='store', help='cfitsio location', default='/usr')
    ctx.add_option('--sprng', action='store', help='sprng location', default='/usr')
    ctx.add_option('--fftw', action='store', help='fftw  location', default='/usr')
    ctx.add_option('--gsl', action='store', help='gsl  location', default='/usr')  
def configure(ctx):
    print('→ configuring the project in ' + ctx.path.abspath())
    
    ctx.load('compiler_c')
    ctx.load('compiler_fc')
    if ctx.options.platform in ["hopper", "edison"]:
        ctx.load('gcc')
        gcc_modifier_func = 'waflib.Tools.gcc.gcc_modifier_cray'
        eval(gcc_modifier_func+'(ctx)')

    else:
        ctx.load('mpicc')
    ctx.env.MPI = ctx.check(header_name=['mpi.h'],mandatory=True)
    
    ctx.env['CFLAGS'] += ["-O3"]
    if 0:#ctx.options.platform=='edison':
        ctx.env['CFLAGS'] += ["-march=native"]
        ctx.env['CFLAGS'] += ["-m64 "]
        ctx.env['CFLAGS'] += ["-static"]
        ctx.env['CFLAGS'] += ["-fPIC"]
    
    ctx.check_cc(lib='fftw3',libpath= path.join(ctx.options.fftw,'lib'), mandatory=True, uselib_store='FFTW3') 
    if ctx.options.platform=='hopper':
        ctx.check_cc(lib='s2hat_acml',  libpath= ctx.options.s2hat, mandatory=True, uselib_store='S2HAT')
        ctx.env['CFLAGS'].append('-I/global/u2/r/radek/s2hat/include/hopper/gnu/acml')
        ctx.env['CFLAGS'].append('-DACML')
        ctx.env['CFLAGS'].append('-DFITS')
        ctx.check_cc(lib='acml', mandatory=True, uselib_store='ACML')
    elif  ctx.options.platform=='edison':
        ctx.check_cc(lib='s2hat_fftw-c2r',  libpath= ctx.options.s2hat, mandatory=True, uselib_store='S2HAT')
        ctx.env['CFLAGS'].append('-I/global/u2/r/radek/s2hat/include/edison/gnu/fftw-c2r')
        ctx.env['CFLAGS'].append('-DFFTW3_C2R')
        ctx.env['CFLAGS'].append('-DFITS')
        ctx.env['CFLAGS'].append('-DGSL')
        ctx.check_cc(lib='gsl',libpath= path.join(ctx.options.gsl,'lib'), mandatory=True, uselib_store='GSL')     
        ctx.check_cc(lib='gslcblas',libpath= path.join(ctx.options.gsl,'lib'), mandatory=True, uselib_store='GSL_CBLAS')     
        ctx.env['CFLAGS'].append('-I%s'%path.join(ctx.options.gsl,'include'))
    else:
        ctx.check_cc(lib='s2hat',  libpath= ctx.options.s2hat, mandatory=True, uselib_store='S2HAT') 
    ctx.check_cc(lib='chealpix',  libpath= path.join(ctx.options.healpix, 'lib'), mandatory=True, uselib_store='CHEALPIX') 
    ctx.check_cc(lib='cfitsio',  libpath = path.join(ctx.options.cfitsio, 'lib'), mandatory=True, uselib_store='CFITSIO') 
    ctx.check_cc(lib='gfortran',  mandatory=True, uselib_store='GF') 
    ctx.check_cc(lib='sprng',   libpath = path.join(ctx.options.sprng, 'lib'),  mandatory=True, uselib_store='SPRNG') 
    ctx.check_cc(lib='gmp',   libpath = path.join(ctx.options.sprng, 'lib'),  mandatory=True, uselib_store='GMP')    
    ctx.check_cc(lib='mpi_f90',  mandatory=False, uselib_store='MPI90') 
    ctx.check_cc(lib='mpi_f77',  mandatory=False, uselib_store='MPI77') 

    ctx.env['CFLAGS'].append('-I'+ctx.options.s2hat)
    ctx.env['CFLAGS'].append('-I%s/include'%ctx.options.healpix)
    ctx.env['CFLAGS'].append('-I%s/include'%ctx.options.cfitsio)
    ctx.env['CFLAGS'].append('-I%s/include'%ctx.options.sprng)    
    ctx.env['CFLAGS'].append('-std=gnu')
    ctx.env['CFLAGS'].append('-DHEALPIXDATA=\"%s/data\"'%ctx.options.healpix)
    ctx.check_cc(header_name='s2hat_types.h', mandatory=True)
    ctx.check_cc(header_name='chealpix.h', mandatory=True)
    ctx.check_cc(header_name='fitsio.h', mandatory=True)

    ctx.write_config_header('config.h')


def build(ctx):

    ctx.env['INCLUDES'] += ["./src"]

    sources = 'src/xpure_create_mll.c src/wig3j_f.f src/xpure_tools.c src/compute_all_xls.c src/xpure_io.c src/s2hat_map2purealm.c'

    libs = []
    libs+= ['S2HAT', 'SPRNG', 'GMP', 'GF', 'FFTW3', 'CHEALPIX', 'CFITSIO', 'MPI90', 'MPI77', 'SPRNG']
    if ctx.options.platform=='hopper':
        libs+= ['ACML']
    #elif ctx.options.platform=='edison':   
    libs+= ['GSL', 'GSL_CBLAS']   

    ctx.program( features= 'fc c',
                 source  = sources, 
                 cflags  = ctx.env['CFLAGS'], 
                 ldflags = ctx.env['LDFLAGS'], 
                 target  = 'xpure_create_mll', 
                 use     = libs)
    ctx.program( features= 'fc c',
                 source  = 'src/xpure.c src/wig3j_f.f src/xpure_tools.c src/compute_all_xls.c src/xpure_io.c src/s2hat_map2purealm.c' , 
                 cflags  = ctx.env['CFLAGS'], 
                 ldflags = ctx.env['LDFLAGS'], 
                 target  = 'xpure', 
                 use     = libs)
    ctx.program( features= 'fc c',
                 source  = 'src/wig3j_f.f src/scalar2spin.c src/xpure_io.c',
                 cflags  = ctx.env['CFLAGS'], 
                 ldflags = ctx.env['LDFLAGS'], 
                 target  = 'scalar2spin', 
                 use     = libs)

    



#  build    : executes the build
#  clean    : cleans the project
#  configure: configures the project
#  dist     : makes a tarball for redistributing the sources
#  distcheck: checks if the project compiles (tarball from 'dist')
#  distclean: removes the build directory
#  install  : installs the targets on the system
#  list     : lists the targets to execute
#  step     : executes tasks in a step-by-step fashion, for debugging
#  uninstall: removes the targets installed
