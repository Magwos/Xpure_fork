# Executable for running mappraiser without TOAST workflow

add_executable(optimal_apodization_mask apodizations/optimalmasks_PCG.c)
add_executable(analytical_apodization apodizations/scalar2spin.c)
add_executable(xpure_create_mll coupling_matrix_computation/xpure_create_mll.c)
add_executable(xpure_get_spectra spectra_estimation/xpure.c)

add_subdirectory(apodizations)
add_subdirectory(coupling_matrix_computation)
add_subdirectory(spectra_estimation)
add_subdirectory(tools)

set_property(TARGET optimal_apodization_mask analytical_apodization xpure_create_mll xpure_get_spectra PROPERTY LINKER_LANGUAGE Fortran)

target_include_directories(optimal_apodization_mask PRIVATE
        "../include"
        )

target_include_directories(analytical_apodization PRIVATE
        "../include"
        )

target_include_directories(xpure_create_mll PRIVATE
        "../include"
        )

target_include_directories(xpure_get_spectra PRIVATE
        "../include"
        )


add_compile_options(${MPI_Fortran_COMPILE_FLAGS})

# MPI compile flag for midapack algebra modules
# target_compile_definitions(optimal_apodization_mask PRIVATE W_MPI)

target_include_directories(optimal_apodization_mask PUBLIC "${HEALPIX_INCLUDES}" "${S2HAT_INCLUDES}" "${CFITSIO_INCLUDES}")
target_link_libraries(optimal_apodization_mask PUBLIC "${HEALPIX_LIBRARIES}" "${S2HAT_LIBRARIES}" "${CFITSIO_LIBRARIES}")
target_compile_options(optimal_apodization_mask PUBLIC "${HEALPIX_LFLAGS}" "${S2HAT_LFLAGS}")

target_include_directories(analytical_apodization PUBLIC "${HEALPIX_INCLUDES}" "${S2HAT_INCLUDES}" "${CFITSIO_INCLUDES}")
target_link_libraries(analytical_apodization PUBLIC "${HEALPIX_LIBRARIES}" "${S2HAT_LIBRARIES}" "${CFITSIO_LIBRARIES}")
target_compile_options(analytical_apodization PUBLIC "${HEALPIX_LFLAGS}" "${S2HAT_LFLAGS}")

target_include_directories(xpure_create_mll PUBLIC "${HEALPIX_INCLUDES}" "${S2HAT_INCLUDES}" "${CFITSIO_INCLUDES}")
target_link_libraries(xpure_create_mll PUBLIC "${HEALPIX_LIBRARIES}" "${S2HAT_LIBRARIES}" "${CFITSIO_LIBRARIES}")
target_compile_options(xpure_create_mll PUBLIC "${HEALPIX_LFLAGS}" "${S2HAT_LFLAGS}")

target_include_directories(xpure_get_spectra PUBLIC "${HEALPIX_INCLUDES}" "${S2HAT_INCLUDES}" "${CFITSIO_INCLUDES}")
target_link_libraries(xpure_get_spectra PUBLIC "${HEALPIX_LIBRARIES}" "${S2HAT_LIBRARIES}" "${CFITSIO_LIBRARIES}")
target_compile_options(xpure_get_spectra PUBLIC "${HEALPIX_LFLAGS}" "${S2HAT_LFLAGS}")


# libraries
target_link_libraries(optimal_apodization_mask PRIVATE MPI::MPI_C MPI::MPI_Fortran)
target_link_libraries(analytical_apodization PRIVATE MPI::MPI_C MPI::MPI_Fortran)
target_link_libraries(xpure_create_mll PRIVATE MPI::MPI_C MPI::MPI_Fortran)
target_link_libraries(xpure_get_spectra PRIVATE MPI::MPI_C MPI::MPI_Fortran)

install(TARGETS optimal_apodization_mask 
        DESTINATION ${PROJECT_SOURCE_DIR}/executables/)

install(TARGETS analytical_apodization 
        DESTINATION ${PROJECT_SOURCE_DIR}/executables/)

install(TARGETS xpure_create_mll 
        DESTINATION ${PROJECT_SOURCE_DIR}/executables/)

install(TARGETS xpure_get_spectra 
        DESTINATION ${PROJECT_SOURCE_DIR}/executables/)