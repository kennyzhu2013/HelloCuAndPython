#include  <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../CudaDll/kernel.cuh"  // ���·��

#pragma comment (lib, "D:\\AI\\mycuda\\CudaDll\\x64\\Debug\\CudaDll.dll")  // ע��˴�����Ҫ��lib��̬���ļ�

namespace py = pybind11;

void addWithCudaWrapper(py::array_t<int> c, py::array_t<int> a, py::array_t<int> b, unsigned int size) {
    // ���������״�ʹ�С
    auto buf_c = c.request();
    auto buf_a = a.request();
    auto buf_b = b.request();

    if (buf_a.size != size || buf_b.size != size || buf_c.size != size) {
        throw std::runtime_error("Array sizes do not match the specified size.");
    }

    // ��ȡָ��
    int* ptr_c = static_cast<int*>(buf_c.ptr);
    const int* ptr_a = static_cast<const int*>(buf_a.ptr);
    const int* ptr_b = static_cast<const int*>(buf_b.ptr);

    // ���� CUDA ����
    addWithCuda(ptr_c, ptr_a, ptr_b, size);
}


// pybind11��cpp�Ĳο��ĵ���https://pybind11.readthedocs.io/en/stable/basics.html
PYBIND11_MODULE(addCuda, m) {
    m.def("addWithCuda", &addWithCudaWrapper, "A function that adds two arrays using CUDA",
        py::arg("c"), py::arg("a"), py::arg("b"), py::arg("size"));
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

// TODO: ����һ��CPythonʵ�ֲο��ĵ���https://learn.microsoft.com/zh-cn/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2022
