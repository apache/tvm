export LDFLAGS = -pthread -lm
export CFLAGS =  -std=c++11 -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops\
	 -Iinclude -Idmlc-core/include -fPIC

# specify tensor path
.PHONY: clean all test lint doc

all: lib/libnngraph.so lib/libnngraph.a cli_test

SRC = $(wildcard src/*.cc src/*/*.cc example/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
ALL_DEP = $(filter-out build/test_main.o, $(ALL_OBJ))

include tests/cpp/unittest.mk

test: $(TEST)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnngraph.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libnngraph.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

cli_test: $(ALL_DEP) build/test_main.o
	$(CXX) $(CFLAGS)  -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lint:
	python2 dmlc-core/scripts/lint.py nngraph cpp include src

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
