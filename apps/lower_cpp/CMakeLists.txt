

add_executable(sample_lower
        SampleLower.cpp
        )
target_link_libraries(sample_lower
        tvm
        tvm_runtime)