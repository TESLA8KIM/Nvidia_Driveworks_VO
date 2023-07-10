# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

project(sample_feature_tracker C CXX) #project(sample_feature_tracker C CXX)

#added
#project(sample_feature_tracker C CXX CUDA) 

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(PUBLIC_DOCS
    README.md
)

set(SOURCES
    main.cpp
)

set(LIBRARIES
    samples_framework
    ${Driveworks_LIBRARIES}
)
#add_executable(${PROJECT_NAME} main.cpp)

#----insert cuda compile settings
#set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "<CMAKE_CUDA_COMPILER> <FLAGS> <CMAKE_CUDA_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")

#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#add_executable(${PROJECT_NAME} main.cpp kernels.cu)
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA) # Add this line

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

# ------------------------------------------------------------------------------
# Install target
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})
