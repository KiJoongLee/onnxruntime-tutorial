#include <iostream> // For standard input/output operations (e.g., std::cout, std::cerr)
#include <vector>   // For std::vector to hold input/output data
#include <string>   // For std::string to handle names and argument parsing
#include <numeric>  // Not strictly needed for this example but good practice
#include <cstdlib>  // For std::stof (string to float) and EXIT_FAILURE/EXIT_SUCCESS
#include <cmath>    // For std::abs

// ONNX Runtime C++ API header file
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    // Check if the correct number of command-line arguments is provided.
    // argc should be 2: 1 for the program name itself, 1 for the input number.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_number>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 3" << std::endl;
        return EXIT_FAILURE; // Indicate an error
    }

    // Parse the input number from the command-line argument.
    // argv[1] contains the first argument (the input number string).
    float input_value;
    try {
        input_value = std::stof(argv[1]); // Convert string to float
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid input number. Please provide a valid floating-point number." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input_number>" << std::endl;
        return EXIT_FAILURE;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error: Input number out of range." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input_number>" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "--- ONNX Runtime Simple Linear Model Inference Example ---" << std::endl;
    std::cout << "Input number received from command line: " << input_value << std::endl;

    // --- 1. Load Model and Prepare Session ---
    // Initialize the ONNX Runtime environment.
    // ORT_LOGGING_LEVEL_WARNING suppresses verbose logs, showing only warnings and errors.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "linear_inference_session");
    // Create session options. Default options are used here.
    Ort::SessionOptions session_options;

    // Define the path to your ONNX model file.
    // IMPORTANT: Adjust this path if your model is not in a 'model' subdirectory.
    const char* model_path = "data/linear/linear.onnx";

    try {
        // Create an ONNX Runtime session by loading the model.
        Ort::Session session(env, model_path, session_options);
        std::cout << "Model loaded successfully from: " << model_path << std::endl;

        // --- 2. Prepare Input Data ---
        // Get the memory allocator, used for managing memory when working with ORT objects.
        Ort::AllocatorWithDefaultOptions allocator;

        // Store the Ort::AllocatedStringPtrs to manage their lifetime,
        // then get the const char* for session.Run().
        // This robust pattern ensures the string pointers remain valid.
        std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
        std::vector<const char*> input_names_c_str;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            input_name_ptrs.push_back(session.GetInputNameAllocated(i, allocator));
            input_names_c_str.push_back(input_name_ptrs.back().get());
        }

        std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
        std::vector<const char*> output_names_c_str;
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            output_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names_c_str.push_back(output_name_ptrs.back().get());
        }

        // Define the input shape for the model.
        // Our linear model expects a single float, so its shape is [1, 1] (Batch size 1, 1 feature).
        std::vector<int64_t> input_shape = {1, 1};

        // Create a standard C++ vector to hold the input data, using the parsed command-line value.
        std::vector<float> input_data = {input_value};
        size_t total_input_elements = input_data.size(); // Total elements in the input tensor.

        // Create memory info for CPU-based tensor creation.
        // This tells ONNX Runtime that the tensor data resides in CPU memory.
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create the ONNX Runtime input tensor (Ort::Value) from our C++ vector.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,            // Memory allocation information
            input_data.data(),      // Raw pointer to the C++ vector's data
            total_input_elements,   // Total number of elements
            input_shape.data(),     // Pointer to the tensor's shape array
            input_shape.size()      // Length of the shape array
        );

        // Ensure the input tensor is valid (optional, good for debugging).
        if (!input_tensor.IsTensor()) {
            std::cerr << "Error: Created input tensor is not valid!" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Input data prepared: " << input_data[0] << std::endl;
        std::cout << "Input tensor shape: [" << input_shape[0] << ", " << input_shape[1] << "]" << std::endl;

        // --- 3. Execute Model Inference ---
        std::cout << "Running inference..." << std::endl;

        // Run the inference! This is the core step.
        // It takes input tensors and returns output tensors.
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr}, // No special run options for this example
            input_names_c_str.data(), // Array of input names (C-style strings)
            &input_tensor,            // Array of input tensors (address of our single input tensor)
            1,                        // Number of input tensors
            output_names_c_str.data(),// Array of output names (C-style strings)
            1                         // Number of output tensors
        );

        std::cout << "Inference completed." << std::endl;

        // --- 4. Process Output Data ---
        // Check if output tensors are valid and not empty.
        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
            std::cerr << "Error: No valid output tensors received!" << std::endl;
            return EXIT_FAILURE;
        }

        // Get a pointer to the raw data of the first output tensor.
        // We expect float values from our linear model.
        float* output_data_ptr = output_tensors[0].GetTensorMutableData<float>();

        // Get the total number of elements in the output tensor.
        size_t total_output_elements = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // Copy the output data from the ORT tensor into a standard C++ vector.
        std::vector<float> output_data(output_data_ptr, output_data_ptr + total_output_elements);

        std::cout << "Inferred output: " << output_data[0] << std::endl; // For our simple model, there's only one element

        // Verify the result by comparing with the expected output (input * 2.0).
        float expected_output = input_value * 2.0f;
        std::cout << "Expected output: " << expected_output << std::endl;

        // Round only the output value to integer
        int rounded_output = std::round(output_data[0]);
        // std::cout << "Rounded output: " << rounded_output << std::endl;

        // Compare rounded output with expected output
        if (rounded_output == expected_output) {
            std::cout << "Inference result matches expected value after rounding. Test PASSED!" << std::endl;
        } else {
            std::cout << "Inference result MISMATCHES expected value after rounding. Test FAILED!" << std::endl;
        }

    } catch (const Ort::Exception& ex) {
        // Catch any ONNX Runtime-specific exceptions for robust error handling.
        std::cerr << "ONNX Runtime Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& ex) {
        // Catch any other standard C++ exceptions.
        std::cerr << "Standard C++ Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "--- Program finished successfully ---" << std::endl;

    return EXIT_SUCCESS; // Indicate successful execution
}
