# 简单使用Visual Studio中进行Python、C++和CUDA混合编程：Windows环境配置

## 项目需求
现有Python项目需部分改用C++提速，其中部分模块需CUDA并行化。仅需支持Windows系统。

## 整体方案
1. 开发CUDA并生成链接库(`.lib`/`.dll`)  
2. 编写CUDA测试单元验证  
3. 用pybind11构建C++ Python库（集成CUDA库）  
4. Python程序整体验证  

## 环境配置
### Visual Studio和CUDA安装
- 先安装VS2019/2022  
- 后安装CUDA 11.6+（勾选*VS Integration*）  

## 构建CUDA程序链接库（文件夹CudaDll）
```cpp
// 头文件需添加导出声明
extern "C" __declspec(dllexport) void functionName(...);
```
**步骤**：  
1. 新建CUDA 11.6 Runtime工程 → 改为动态库  
2. 勾选`CUDA 11.6(.targets,.props)`  
3. 记录生成的`.../x64/Debug/xxx.lib`路径  

## 构建CUDA测试单元（后续源码文件用到）
```cpp
#include "头文件.cuh"  // 相对路径
#pragma comment(lib, "库文件路径.lib")
```

## pybind11集成
### 安装配置
```bash
pip install pybind11
git clone https://github.com/pybind/pybind11.git
```

### CMake配置
```cmake
cmake_minimum_required(VERSION 3.2)
project(工程名)
add_subdirectory(pybind11)
pybind11_add_module(工程名 源码文件.cpp)
```
**构建命令**（生成文件夹cudaWrap，假设工程名为cudaWrap）  
对于VS2019版本，将G参数换成"Visual Studio 16 2019"：
```bash
cmake . -G "Visual Studio 17 2022" -A x64
```  
执行完毕后，即可在文件夹中看到Visual Studio的工程文件，包含：工程名.sln，工程名.vcxproj，ALL_BUILD.vcxproj和ZERO_CHECK.vcxproj。我们只需要工程名.vcxproj，把它添加到原有的那个解决方案（不是这一步生成的解决方案）中即可。

生成项目，在工程名.vcxproj同文件夹下的Debug文件夹中，即可找到生成的.pyd文件。
  
## Python调用（这一步也可以用pycharm，代码一样）
- 用 ***https://learn.microsoft.com/zh-cn/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2022*** 参考资料创建python项目到原先解决方案中，参考目录PythonApp
- 将生成的`.pyd`导入项目  
```
import sys
sys.path.append('D:\\AI\\mycuda\\cudaWrap\\Debug\\')
import addCuda
```
> 提示：文中CUDA 12.3为示例版本，请按实际调整

代码路径：**https://github.com/kennyzhu2013/HelloCuAndPython.git**
