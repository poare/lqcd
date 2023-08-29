F90 = gfortran

#FFLAGS =-64 -model=M1 -Oss -precise -noparallel
#
#LOGLIST = -loglist -listlang=euc 
#
#LDFLAGS = -64 -model=M1 -Oss -precise -noparallel

MODULE = sparse

SOURCES = \
	main_v2.f   \
	smac_lib.f       

OBJS = ${SOURCES:.f=.o}

all:	$(MODULE)

.f.o:
	$(F90) $(FFLAGS) $(LOGLIST) $(MONITOR) -c $<

$(MODULE): $(OBJS)
	rm -f $(MODULE)
	$(F90) $(OBJS)  $(MONITOR) $(LDFLAGS) -o $@ $(LIBC)

clean:
	rm -f $(OBJS) $(MODULE) *~ *.bak *.L

## dependencies
$(OBJS):	para.h 


