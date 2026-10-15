# Build the levisim Fortran reference implementation.
#
# Each target differs only in which parameter file supplies the mechanical frequency
# layout; all of them link the same main program, susceptibilities and spectrum code.
#
#   make pairs    frequencies in pairs, 312 kHz with a 15 kHz inter-pair gap
#                 (this is the variant that produced the committed .dat reference files)
#   make equal    uniform ladder from 320 kHz in 3 kHz steps, two modes parked at 5 MHz
#   make random   ladder from 350 kHz in 7.5 kHz steps with seeded pseudo-random jitter
#
# Each build writes its executable and its output .dat files into build/<variant>/, so the
# variants do not overwrite each other or the reference data at the top level.
#
# LAPACK comes from the system by default. Under conda, point at the environment:
#   make pairs LDFLAGS="-L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib"

# NOTE: plain '=' not '?=' - make predefines FC as f77, so '?=' would never take effect.
# A command-line override (make FC=...) still wins over this.
FC      = gfortran
FFLAGS  = -O2
LDLIBS  ?= -llapack

# -Wall -Wextra is the lint target rather than the default, because the INCLUDE-file
# approach to shared constants makes unused-parameter warnings unavoidable in most
# subprograms. The warning that mattered - cmplx() defaulting to single precision - is fixed.
LINTFLAGS ?= -Wall -Wextra -Wconversion -Wno-unused-parameter -Wno-unused-dummy-argument

COMMON_SOURCES := NParticle.f90 3DEQRom.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90
HEADERS        := Nparticle.h CSCAVITY.h

VARIANTS := pairs equal random
pairs_PARAMS  := ParametersNpair.f90
equal_PARAMS  := ParametersEqual.f90
random_PARAMS := ParametersRandom.f90

.PHONY: all $(VARIANTS) lint clean help

all: $(VARIANTS)

define VARIANT_RULE
$(1): $$(COMMON_SOURCES) $$($(1)_PARAMS) $$(HEADERS)
	@mkdir -p build/$(1)
	@cp $$(HEADERS) build/$(1)/
	$$(FC) $$(FFLAGS) -J build/$(1) -I build/$(1) -o build/$(1)/levisim-$(1).x \
	    $$(COMMON_SOURCES) $$($(1)_PARAMS) $$(LDFLAGS) $$(LDLIBS)
	@echo "built build/$(1)/levisim-$(1).x  (run it from inside build/$(1)/)"
endef
$(foreach v,$(VARIANTS),$(eval $(call VARIANT_RULE,$(v))))

# compile-only pass with warnings enabled, to check the sources stay clean
lint: $(HEADERS)
	@mkdir -p build/lint
	@cp $(HEADERS) build/lint/
	$(FC) $(FFLAGS) $(LINTFLAGS) -J build/lint -I build/lint -c \
	    $(COMMON_SOURCES) $(pairs_PARAMS) $(equal_PARAMS) $(random_PARAMS)
	@rm -f *.o
	@echo "lint pass complete"

clean:
	rm -rf build *.o *.mod

help:
	@echo "targets: $(VARIANTS) lint clean"
