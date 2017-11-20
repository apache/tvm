MPS_CONTRIB_SRC = $(wildcard src/contrib/mps/*.mm, src/contrib/mps/*.cc)
MPS_CONTRIB_OBJ = $(patsubst src/%.mm, build/%.o, $(MPS_CONTRIB_SRC))

ifeq ($(USE_MPS), 1)
FRAMEWORKS += -framework MetalPerformanceShaders
CFLAGS += 
ADD_LDFLAGS += 
RUNTIME_DEP += $(MPS_CONTRIB_OBJ)
endif

build/contrib/mps/%.o: src/contrib/mps/%.mm src/contrib/mps/%.cc
	@mkdir -p $(@D)
	$(CXX) $(OBJCFLAGS) $(CFLAGS) -MM -MT build/contrib/mps/$*.o $< >build/contrib/mps/$*.d
	$(CXX) $(OBJCFLAGS) -c $(CFLAGS) -c $< -o $@
