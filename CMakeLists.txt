cmake_minimum_required(VERSION 3.20)
project(myevs_native LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native_emlb MODULE
    cpp/native_emlb/native_emlb.cpp
)

target_compile_features(_native_emlb PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(_native_emlb PRIVATE /O2 /EHsc)
else()
    target_compile_options(_native_emlb PRIVATE -O3)
endif()

install(TARGETS _native_emlb LIBRARY DESTINATION myevs)
