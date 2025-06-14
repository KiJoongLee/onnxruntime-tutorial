#include <iostream>
#include <onnxruntime_cxx_api.h> // ONNX Runtime C++ API header file

int main() {
    // Check ONNX Runtime version
    // Ort::GetVersionString() function returns the version of the ONNX Runtime library as a string.
    std::cout << "ONNX Runtime Version: " << Ort::GetVersionString() << std::endl;

    return 0;
}
