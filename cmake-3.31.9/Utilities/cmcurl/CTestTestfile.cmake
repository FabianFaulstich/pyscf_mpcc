# CMake generated Testfile for 
# Source directory: /home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Utilities/cmcurl
# Build directory: /home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[curl]=] "curltest" "http://open.cdash.org/user.php")
set_tests_properties([=[curl]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Utilities/cmcurl/CMakeLists.txt;1994;add_test;/home/talha/Documents/RR_MPCC/pyscf_mpcc/cmake-3.31.9/Utilities/cmcurl/CMakeLists.txt;0;")
subdirs("lib")
