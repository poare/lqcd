# Build the NPR momentum-fraction measurement against the installed Chroma.
#
# Route (a), the default: a standalone binary that registers the measurement
# into a stock Chroma at runtime. No Chroma rebuild, one translation unit.
#
# Route (b), for the exponential clover: compile npr_momfrac_chroma.cc with
# -DNPR_MOMFRAC_NO_MAIN inside the tree that carries the action instead.

CHROMA_CONFIG := $(HOME)/libraries/install/chroma/bin/chroma-config
CXX      := $(shell $(CHROMA_CONFIG) --cxx)
CXXFLAGS := $(shell $(CHROMA_CONFIG) --cxxflags)
LDFLAGS  := $(shell $(CHROMA_CONFIG) --ldflags)
LIBS     := $(shell $(CHROMA_CONFIG) --libs)

npr_momfrac: npr_momfrac_chroma.cc
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -f npr_momfrac

.PHONY: clean
