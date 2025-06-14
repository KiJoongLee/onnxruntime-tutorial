#include <iostream> // For standard input/output operations (e.g., std::cout, std::cerr)
#include <vector>   // For std::vector to hold input/output data
#include <string>   // For std::string to handle names and argument parsing
#include <numeric>  // Not strictly needed for this example but good practice
#include <cstdlib>  // For std::stof (string to float) and EXIT_FAILURE/EXIT_SUCCESS
#include <cmath>    // For std::abs

// ONNX Runtime C++ API header file
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <input_number1> [input_number2] [input_number3] [input_number4] [input_number5]" << std::endl;
        return 1;
    }

    try {
        // Calculate number of inputs (argc - 1)
        size_t num_inputs = argc - 1;
        std::vector<float> input_data;
        input_data.reserve(num_inputs);

        // Parse input values
        for (int i = 1; i < argc; i++) {
            input_data.push_back(std::stof(argv[i]));
        }

        // Set input shape [batch_size, 1]
        std::vector<int64_t> input_shape = {static_cast<int64_t>(num_inputs), 1};

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

        // Process and display results for each input
        for (size_t i = 0; i < num_inputs; i++) {
            int rounded_output = std::round(output_data[i]);
            int expected_output = std::round(input_data[i] * 2.0f);

            std::cout << "Input " << (i + 1) << ": " << input_data[i] << std::endl;
            std::cout << "Output " << (i + 1) << ": " << output_data[i] << std::endl;
            std::cout << "Expected " << (i + 1) << ": " << expected_output << std::endl;
            std::cout << "Test " << (i + 1) << " " << (rounded_output == expected_output ? "PASSED" : "FAILED") << std::endl;
            std::cout << "-------------------" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
