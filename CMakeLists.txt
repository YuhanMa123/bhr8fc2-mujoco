cmake_minimum_required(VERSION 3.14)

project(bitbot)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BITBOT_DEPENDENCY_USE_PROXY TRUE)
set(BITBOT_DEPENDENCY_USE_LOCAL_FILE TRUE)
set(BITBOT_DEPENDENCY_LOCAL_FILE_PATH ${CMAKE_SOURCE_DIR}/dependencies) # 指定本地依赖文件的路径

# 可通过设置 CMAKE_PREFIX_PATH 变量为CMake提供额外的查找项目库文件或头文件的路径 
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/mayh/MYH/Library/libtorch/libtorch-cpu/
)
include_directories(${TORCH_PATH}/include)

include(bitbot.cmake)

include_directories("/usr/include/eigen3")

# 使用自定义函数 BitbotAddDependency 添加名为 bitbot_mujoco 的依赖项，指定其文件名和路径，并启用本地文件
BitbotAddDependency(
  NAME           bitbot_mujoco
  FILE_NAME      "bitbot_mujoco-main.zip"
  FILE_PATH      ${BITBOT_DEPENDENCY_LOCAL_FILE_PATH}
  USE_LOCAL_FILE TRUE
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

find_package(Torch REQUIRED)

# 添加可执行文件，并指定3个源文件
add_executable(main_app 
  main.cpp
  user_func.cpp
  leeMatrix.c
)

# 为main_app添加Libtorch的头文件目录
target_include_directories(main_app PUBLIC
  ${TORCH_INCLUDE_DIRS}
)
# 为main_app添加bitbotmujoco和Libtorch的库文件
target_link_libraries(main_app
  BitbotMujoco
  ${TORCH_LIBRARIES}
)
target_compile_options(main_app PUBLIC) # 为main_app设置编译选项
target_compile_features(main_app PUBLIC cxx_std_20) # 指定main_app使用C++20标准
