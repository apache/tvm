# vta software components
file(GLOB TSIM_SW_SRC src/test_driver.cc)
add_library(driver SHARED ${TSIM_SW_SRC})
target_include_directories(driver PRIVATE ${VTA_DIR}/include)

if(APPLE)
  set_target_properties(driver PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)
