#include <iostream> // For standard input/output operations
#include <vector>   // For std::vector to store shapes and names
#include <string>   // For std::string to handle tensor names
#include <numeric>  // For std::accumulate (useful for calculating total elements)
#include <cstdlib>  // For EXIT_FAILURE/EXIT_SUCCESS
#include <algorithm> // For std::all_of

// ONNX Runtime C++ API header file
#include <onnxruntime_cxx_api.h>

// Helper function to convert ONNXTensorElementDataType to a readable string
// This function helps in displaying the data type of the tensor.
std::string get_tensor_data_type_string(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED: return "undefined";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "double";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "complex64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
        default: return "unknown";
    }
}

// Custom helper function to check if a shape is static (contains no -1)
// This replaces the HasStaticShape() method for broader compatibility.
bool is_shape_static(const std::vector<int64_t>& shape) {
    for (int64_t dim : shape) {
        if (dim == -1) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Check if the correct number of command-line arguments is provided.
    // argc should be 2: 1 for the program name itself, 1 for the model path.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " model/linear.onnx" << std::endl;
        std::cerr << "Example: " << argv[0] << " linear.onnx" << std::endl;
        return EXIT_FAILURE; // Indicate an error due to incorrect usage
    }

    // The model path is the first argument provided by the user.
    const char* model_path = argv[1];

    std::cout << "--- ONNX Runtime Model Information Example ---" << std::endl;
    std::cout << "Attempting to load model from: " << model_path << std::endl;

    // --- 1. Initialize ONNX Runtime Environment ---
    // Create an ONNX Runtime environment. This is the entry point for ONNX Runtime operations.
    // ORT_LOGGING_LEVEL_WARNING suppresses verbose logs, showing only warnings and errors.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_info_session");
    std::cout << "ONNX Runtime environment initialized." << std::endl;

    // --- 2. Define Session Options ---
    // Create session options. Default options are used here.
    Ort::SessionOptions session_options;

    // --- 3. Load Model and Create Session ---
    // Create an ONNX Runtime session by loading the model.
    // This step parses the model and prepares it for querying information.
    try {
        Ort::Session session(env, model_path, session_options);
        std::cout << "Model loaded successfully." << std::endl;

        // Get a memory allocator for obtaining string names.
        Ort::AllocatorWithDefaultOptions allocator;

        // --- 4. Get Input Tensor Information ---
        std::cout << "\n--- Input Tensor Information ---" << std::endl;

        // Get the number of input nodes in the model.
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "Number of input nodes: " << num_input_nodes << std::endl;

        // Vectors to store smart pointers for names and their C-style string pointers.
        std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
        // std::vector<const char*> input_names_c_str; // No longer needed directly for this logic

        for (size_t i = 0; i < num_input_nodes; ++i) {
            std::cout << "  Input " << i << ":" << std::endl;

            // Get input name
            input_name_ptrs.push_back(session.GetInputNameAllocated(i, allocator));
            std::cout << "    Name: " << input_name_ptrs.back().get() << std::endl;

            // Get input type information
            Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
            // Cast to TensorTypeAndShapeInfo to get tensor-specific details.
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

            // Get data type
            ONNXTensorElementDataType input_data_type = input_tensor_info.GetElementType();
            std::cout << "    Data Type: " << get_tensor_data_type_string(input_data_type) << std::endl;

            // Get shape
            std::vector<int64_t> input_node_dims = input_tensor_info.GetShape();
            std::cout << "    Shape: [";
            for (size_t j = 0; j < input_node_dims.size(); ++j) {
                // -1 indicates a dynamic dimension (e.g., batch size)
                if (input_node_dims[j] == -1) {
                    std::cout << "dynamic";
                } else {
                    std::cout << input_node_dims[j];
                }
                if (j < input_node_dims.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;

            // Get total number of elements (if shape is static)
            // Using our custom is_shape_static helper function for compatibility.
            if (is_shape_static(input_node_dims)) {
                std::cout << "    Total Elements (if static): " << input_tensor_info.GetElementCount() << std::endl;
            } else {
                std::cout << "    Total Elements: Varies (dynamic shape)" << std::endl;
            }
        }

        // --- 5. Get Output Tensor Information ---
        std::cout << "\n--- Output Tensor Information ---" << std::endl;

        // Get the number of output nodes in the model.
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "Number of output nodes: " << num_output_nodes << std::endl;

        // Vectors to store smart pointers for names and their C-style string pointers.
        std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
        // std::vector<const char*> output_names_c_str; // No longer needed directly for this logic

        for (size_t i = 0; i < num_output_nodes; ++i) {
            std::cout << "  Output " << i << ":" << std::endl;

            // Get output name
            output_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
            std::cout << "    Name: " << output_name_ptrs.back().get() << std::endl;

            // Get output type information
            Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

            // Get data type
            ONNXTensorElementDataType output_data_type = output_tensor_info.GetElementType();
            std::cout << "    Data Type: " << get_tensor_data_type_string(output_data_type) << std::endl;

            // Get shape
            std::vector<int64_t> output_node_dims = output_tensor_info.GetShape();
            std::cout << "    Shape: [";
            for (size_t j = 0; j < output_node_dims.size(); ++j) {
                if (output_node_dims[j] == -1) {
                    std::cout << "dynamic";
                } else {
                    std::cout << output_node_dims[j];
                }
                if (j < output_node_dims.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;

            // Get total number of elements (if shape is static)
            // Using our custom is_shape_static helper function for compatibility.
            if (is_shape_static(output_node_dims)) {
                std::cout << "    Total Elements (if static): " << output_tensor_info.GetElementCount() << std::endl;
            } else {
                std::cout << "    Total Elements: Varies (dynamic shape)" << std::endl;
            }
        }

    } catch (const Ort::Exception& ex) {
        // Catch ONNX Runtime-specific exceptions (e.g., model not found, invalid model)
        std::cerr << "ONNX Runtime Error: " << ex.what() << std::endl;
        std::cerr << "Please ensure the model path '" << model_path << "' is correct and the file exists." << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& ex) {
        // Catch any other standard C++ exceptions
        std::cerr << "Standard C++ Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n--- Program finished successfully ---" << std::endl;

    return EXIT_SUCCESS;
}
