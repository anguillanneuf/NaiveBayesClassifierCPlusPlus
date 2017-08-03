cmake_minimum_required(VERSION 3.6)
project(T3L3_NaiveBayesC__)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp classifier.h classifier.cpp)
add_executable(T3L3_NaiveBayesC__ ${SOURCE_FILES})