cmake_minimum_required(VERSION 3.4.1)

add_library(
        tvm4j_runtime_packed
     
        SHARED

        IMPORTED
        )

if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set_target_properties(
        tvm4j_runtime_packed

            PROPERTIES IMPORTED_LOCATION

        ${CMAKE_SOURCE_DIR}/../obj/local/arm64-v8a/libtvm4j_runtime_packed.so

)

elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set_target_properties(
            tvm4j_runtime_packed

            PROPERTIES IMPORTED_LOCATION

            ${CMAKE_SOURCE_DIR}/../obj/local/armeabi-v7a/libtvm4j_runtime_packed.so

    )

elseif(${ANDROID_ABI} STREQUAL "x86")
set_target_properties(
        tvm4j_runtime_packed

        PROPERTIES IMPORTED_LOCATION

        ${CMAKE_SOURCE_DIR}/../obj/local/x86/libtvm4j_runtime_packed.so

)    

elseif(${ANDROID_ABI} STREQUAL "x86_64")
    set_target_properties(
            tvm4j_runtime_packed

            PROPERTIES IMPORTED_LOCATION

            ${CMAKE_SOURCE_DIR}/../obj/local/x86_64/libtvm4j_runtime_packed.so

    )

endif()