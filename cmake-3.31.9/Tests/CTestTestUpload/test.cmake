cmake_minimum_required(VERSION 3.10)

# Settings:
set(CTEST_DASHBOARD_ROOT                "/home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Tests/CTestTest")
set(CTEST_SITE                          "Schweinshaxe")
set(CTEST_BUILD_NAME                    "CTestTest-Linux-g++-Upload")

set(CTEST_SOURCE_DIRECTORY              "/home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Tests/CTestTestUpload")
set(CTEST_BINARY_DIRECTORY              "/home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Tests/CTestTestUpload")
set(CTEST_CMAKE_GENERATOR               "Unix Makefiles")
set(CTEST_CMAKE_GENERATOR_PLATFORM      "")
set(CTEST_CMAKE_GENERATOR_TOOLSET       "")
set(CTEST_BUILD_CONFIGURATION           "$ENV{CMAKE_CONFIG_TYPE}")

CTEST_START(Experimental)
CTEST_CONFIGURE(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE res)
CTEST_BUILD(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE res)
CTEST_UPLOAD(FILES "${CTEST_SOURCE_DIRECTORY}/sleep.c" "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
CTEST_SUBMIT()
