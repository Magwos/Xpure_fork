target_sources(optimal_apodization_mask PRIVATE 
                "xpure_io.c"
                "wig3j_f.f")

target_sources(analytical_apodization PRIVATE 
                "xpure_io.c"
                "xpure_tools.c"
                "wig3j_f.f")

target_sources(xpure_create_mll PRIVATE 
                "xpure_io.c"
                "xpure_tools.c"
                "wig3j_f.f")

target_sources(xpure_get_spectra PRIVATE 
                "xpure_io.c"
                "xpure_tools.c"
                "wig3j_f.f")
