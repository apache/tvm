export LDFLAGS= -pthread -lm
export CFLAGS=  -std=c++11 -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops\
	 -Iinclude -Idmlc-core/include -fPIC

# specify tensor path
.PHONY: clean all test


all: lib/libnngraph.so test

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC)) $(PLUGIN_OBJS)
ALL_DEP = $(ALL_OBJ)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnngraph.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

test: $(ALL_DEP)
	$(CXX) $(CFLAGS)  -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o xgboost

-include build/*.d
-include build/*/*.d
