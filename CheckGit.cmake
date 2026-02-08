set(pre_configure_file ${CMAKE_CURRENT_SOURCE_DIR}/src/MeshField_GetHash.cpp.in)
set(post_configure_file ${CMAKE_CURRENT_BINARY_DIR}/MeshField_GetHash.cpp)

function(CheckGitWrite git_hash)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git-state.txt "${git_hash}\n")
endfunction()

function(CheckGitRead git_hash)
    if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/git-state.txt)
        file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/git-state.txt CONTENT)
        LIST(GET CONTENT 0 var)

        set(${git_hash} ${var} PARENT_SCOPE)
    endif ()
endfunction()

function(CheckGitVersion)
    execute_process(COMMAND git rev-parse HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      OUTPUT_VARIABLE GIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    CheckGitRead(GIT_HASH_CACHE)
    if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif ()

    if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/MeshField_GetHash.hpp)
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/MeshField_GetHash.hpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (NOT DEFINED GIT_HASH_CACHE)
        set(GIT_HASH_CACHE "INVALID")
    endif ()

    if (NOT ${GIT_HASH} STREQUAL ${GIT_HASH_CACHE} OR NOT EXISTS ${post_configure_file})
        CheckGitWrite(${GIT_HASH})

        configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
    endif ()

endfunction()

function(CheckGitSetup)

    add_custom_target(AlwaysCheckGit COMMAND ${CMAKE_COMMAND}
        -DRUN_CHECK_GIT_VERSION=1
        -Dpre_configure_dir=${CMAKE_CURRENT_SOURCE_DIR}
        -Dpost_configure_file=${CMAKE_CURRENT_BINARY_DIR}
        -DGIT_HASH_CACHE=${GIT_HASH_CACHE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/CheckGit.cmake
        BYPRODUCTS ${post_configure_file}
        )


    CheckGitVersion()
endfunction()

if (RUN_CHECK_GIT_VERSION)
  CheckGitVersion()
endif ()
