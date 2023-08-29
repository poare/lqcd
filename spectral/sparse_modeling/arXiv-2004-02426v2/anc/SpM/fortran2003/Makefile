F90 = gfortran

#FFLAGS =-64 -model=M1 -Oss -precise -noparallel
#
#LOGLIST = -loglist -listlang=euc 
#
#LDFLAGS = -64 -model=M1 -Oss -precise -noparallel

MODULE = sparse

SOURCES = \
	main_v2.f90   \
	smac_lib.f90  \
	readfiles.f90    \
	para.f90


OBJS = ${SOURCES:.f90=.o}


.SUFFIXES	:
.SUFFIXES	:.f90 .o
.SUFFIXES	:.mod

all:	$(MODULE)

main_v2.o:smac_lib.o readfiles.o para.o
smac_lib.o:para.o

.f90.o:
	$(F90) $(FFLAGS) $(LOGLIST) $(MONITOR) -c $<

$(MODULE): $(OBJS)
	rm -f $(MODULE)
	$(F90) $(OBJS)  $(MONITOR) $(LDFLAGS) -o $@ $(LIBC)

clean:
	rm -f $(OBJS) $(MODULE) *~ *.bak *.L

## dependencies
$(OBJS):	para.h 


