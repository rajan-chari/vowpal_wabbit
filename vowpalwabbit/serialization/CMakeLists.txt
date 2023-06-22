set(vw_serialization_headers
  include/vw/serialization/autofb_schema.h
  include/vw/serialization/autofb_serializer.h
  include/vw/serialization/base.h
  include/vw/serialization/reflection.h
  include/vw/serialization/type_activation.h
  include/vw/serialization/type_builder.h
  include/vw/serialization/type_constructors.h
  include/vw/serialization/type_erase.h
  include/vw/serialization/type_reflection.h
  include/vw/serialization/type_registry.h
  )

set(vw_serialization_sources
  src/autofb.cc
  src/typesys.cc
  #src/parse_example_slates_json.cc
)

vw_add_library(
  NAME "serialization"
  TYPE "STATIC_ONLY"
  SOURCES ${vw_serialization_headers} ${vw_serialization_sources}
  PUBLIC_DEPS 
    vw_config
    flatbuffers::flatbuffers # needed because we depend on the binary schema generation / parsing codebase
                             # for generating the binary schemas and using them to drive full reflection.
  #PRIVATE_DEPS $<BUILD_INTERFACE:RapidJSON>
  DESCRIPTION "[Experimental] Serialization and data binding library."
  EXCEPTION_DESCRIPTION "Yes"
  ENABLE_INSTALL
)
#target_include_directories(vw_fb_parser PUBLIC ${FLATBUFFERS_INCLUDE_DIR})
find_package(Flatbuffers CONFIG REQUIRED)

target_include_directories(vw_serialization 
  PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/src
  #PUBLIC # needed?
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${FLATBUFFERS_INCLUDE_DIR})

# vw_add_library(
#     NAME "core"
#     TYPE "STATIC_ONLY"
#     SOURCES ${vw_core_headers} ${vw_core_sources}
#     PUBLIC_DEPS vw_common vw_explore vw_allreduce vw_config fmt::fmt vw_cache_parser vw_text_parser vw_json_parser
#     PRIVATE_DEPS
#         # Special case for dealing with header-only INTERFACE dependencies of a STATIC library
#         # Use BUILD_INTERFACE to prevent them from being exported, i.e. treat them as PRIVATE
#         # https://gitlab.kitware.com/cmake/cmake/issues/15415
#         ${CMAKE_DL_LIBS} ${LINK_THREADS} vw_io $<BUILD_INTERFACE:${boost_math_target}> $<BUILD_INTERFACE:RapidJSON>
#         $<BUILD_INTERFACE:eigen> $<BUILD_INTERFACE:sse2neon>
#     DESCRIPTION "This contains all remaining VW code, all reduction implementations, driver, option handling"
#     EXCEPTION_DESCRIPTION "Yes"
#     ENABLE_INSTALL
# )

vw_add_test_executable(
  FOR_LIB "serialization"
  SOURCES 
    tests/autofb_schema_test.cc
)
