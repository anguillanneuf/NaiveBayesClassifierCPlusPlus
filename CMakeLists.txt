cmake_minimum_required(VERSION 3.6)
project(T3L3_NaiveBayesC__)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp classifier.h classifier.cpp matplotlibcpp.h)
add_executable(T3L3_NaiveBayesC__ ${SOURCE_FILES})

find_package(PythonLibs 2.7)
target_include_directories(T3L3_NaiveBayesC__ PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(T3L3_NaiveBayesC__ ${PYTHON_LIBRARIES})