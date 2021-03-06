#
#  Makefile for openMP assignment
#
CMD    = xserial
CC     = icc
CFLAGS = -O2 -align -Zp8 -axP -unroll
#CFLAGS = -O2 -openmp 
LFLAGS = -openmp 
LIBS   = -lm -lrt
INCLUDE = -openmp
OBJS   = serial_hw2.o timing.o 

.c.o:
	$(CC)  $(CFLAGS) $(INCLUDE) -c $<


$(CMD): $(OBJS)
	$(CC)  -o $@ $^ $(LFLAGS) $(LIBS)

# 
.PHONY: clean new

clean:
	-/bin/rm -f *.o *~ $(CMD)

new:
	make clean
	make $(CMD)
