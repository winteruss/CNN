cmake_minimum_required(VERSION 3.10)

project(CNN)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/wd4267)
endif()

add_executable(${PROJECT_NAME} src/main.cpp)