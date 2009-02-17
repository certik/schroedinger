PYVER := $(shell script/config.py python_version)
OPTFLAGS := $(shell script/config.py opt_flags)
NUMPYINCLUDE := $(shell script/config.py numpy_include)
HERMES := $(shell script/config.py hermes2d_path)

CYTHON := cython
CXX := g++
LIBS := -lhermes -lglut -lJudy -lumfpack -lamd -lblas
CFLAGS := -Wfatal-errors -I/usr/include/python$(PYVER) $(OPTFLAGS) -I$(NUMPYINCLUDE)/numpy -I$(HERMES)/../../../include/hermes2d -I$(HERMES)/include
LDFLAGS := -Wfatal-errors -Wl,--rpath=$(HERMES) -L$(HERMES)

all: cschroed.so

cschroed.so: dft.o

%.c: %.pyx
	$(CYTHON) -I$(HERMES)/include $<

%.o : %.c
	$(CXX) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.so: %.o
	$(CXX) $(LDFLAGS) -shared $+ -o $@ $(LIBS)

clean:
	rm -f *.so *.pyd *.o *.pyc cschroed.c

check:
	./test tests
