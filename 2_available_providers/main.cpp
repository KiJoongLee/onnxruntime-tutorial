#include <iostream> // For standard input/output operations
#include <vector>   // For std::vector
#include <string>   // For std::string
#include <cstdlib>  // For EXIT_SUCCESS/EXIT_FAILURE

// ONNX Runtime C++ API header file
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- ONNX Runtime Execution Provider (EP) Information ---" << std::endl;

    // --- 1. Initialize ONNX Runtime Environment ---
    // Create an ONNX Runtime environment. This is the entry point for ONNX Runtime operations.
    // ORT_LOGGING_LEVEL_WARNING suppresses verbose logs, showing only warnings and errors.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ep_info_session");
    std::cout << "ONNX Runtime environment initialized." << std::endl;

    // --- 2. Get Available Execution Providers ---
    // Ort::GetAvailableProviders() returns a vector of strings, where each string is
    // the name of an available execution provider.
    try {
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();

        std::cout << "\n--- Available Execution Providers ---" << std::endl;
        if (available_providers.empty()) {
            std::cout << "No execution providers found. This is unexpected; CPU should always be available." << std::endl;
        } else {
            for (const auto& provider : available_providers) {
                std::cout << "- " << provider << std::endl;
            }
        }

    } catch (const Ort::Exception& ex) {
        // Catch ONNX Runtime-specific exceptions
        std::cerr << "ONNX Runtime Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& ex) {
        // Catch any other standard C++ exceptions
        std::cerr << "Standard C++ Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n--- Program finished successfully ---" << std::endl;

    return EXIT_SUCCESS;
}
