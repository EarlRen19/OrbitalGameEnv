#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "oge::oge-lib" for configuration ""
set_property(TARGET oge::oge-lib APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(oge::oge-lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liboge.a"
  )

list(APPEND _cmake_import_check_targets oge::oge-lib )
list(APPEND _cmake_import_check_files_for_oge::oge-lib "${_IMPORT_PREFIX}/lib/liboge.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
