#include <iostream> // For standard input/output operations (e.g., std::cout, std::cerr)
#include <vector>   // For std::vector to hold input/output data
#include <string>   // For std::string to handle names and argument parsing
#include <numeric>  // Not strictly needed for this example but good practice
#include <cstdlib>  // For std::stof (string to float) and EXIT_FAILURE/EXIT_SUCCESS
#include <cmath>    // For std::abs

// ONNX Runtime C++ API header file
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_number>" << std::endl;
        return 1;
    }

    try {
        float input_value = std::stof(argv[1]);
        std::vector<float> input_data = {input_value};
        std::vector<int64_t> input_shape = {1, 1};

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "linear_inference");
        Ort::Session session(env, "data/linear/linear.onnx", Ort::SessionOptions());

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        const char* input_name = "input";
        const char* output_name = "output";
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        int rounded_output = std::round(output_data[0]);
        int expected_output = std::round(input_value * 2.0f);

        std::cout << "Input: " << input_value << std::endl;
        std::cout << "Output: " << output_data[0] << std::endl;
        std::cout << "Expected: " << expected_output << std::endl;
        std::cout << "Test " << (rounded_output == expected_output ? "PASSED" : "FAILED") << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
