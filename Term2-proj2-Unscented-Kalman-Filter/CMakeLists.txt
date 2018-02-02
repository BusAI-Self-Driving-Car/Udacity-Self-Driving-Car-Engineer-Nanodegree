project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/ukf.cpp src/main.cpp src/tools.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 
    include_directories(/usr/local/include)
    include_directories(/usr/local/opt/openssl/include)
    link_directories(/usr/local/lib)
    link_directories(/usr/local/opt/openssl/lib)
    link_directories(/usr/local/Cellar/libuv/1*/lib)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

add_executable(UnscentedKF ${sources})
target_link_libraries(UnscentedKF z ssl uv uWS)


## GTEST
add_subdirectory(googletest-1.8.0)  # project(gtest)
enable_testing()

# Include the gtest library. gtest_SOURCE_DIR is available due to
# 'project(gtest)' above.
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

##############
# Unit Tests
##############
add_library(ukf_lib src/ukf.cpp)
add_executable(test_ukf src/test_ukf.cpp)

# Standard linking to gtest stuff.
target_link_libraries(test_ukf gtest gtest_main)

# Extra linking for the project.
target_link_libraries(test_ukf ukf_lib)

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
#add_test(NAME that-test-I-made COMMAND runUnitTests)

# You can also omit NAME and COMMAND. The second argument could be some other
# test executable.
#add_test(that-other-test-I-made runUnitTests)
