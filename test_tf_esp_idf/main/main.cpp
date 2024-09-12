#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "flood_prediction_model_quantized.h"  // The model in C array format

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main() {
    // Load the TensorFlow Lite model from the C array
    const tflite::Model* model = tflite::GetModel(flood_prediction_model_quantized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Error: Model schema version %ld does not match supported version %d.\n", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Create MicroMutableOpResolver and register required operations
    tflite::MicroMutableOpResolver<6> resolver;  // Reserve space for 6 operations

    // Add only the operations your model uses
    resolver.AddConv2D();         // If using Conv2D layer
    resolver.AddFullyConnected(); // If using Dense/FullyConnected layer
    resolver.AddReshape();        // If using Reshape
    resolver.AddMaxPool2D();      // If using MaxPool2D
    resolver.AddSoftmax();        // If using Softmax activation
    resolver.AddLogistic();       // Add Logistic (Sigmoid) operation

    // Build an interpreter
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);

    // Allocate memory for the tensors
    interpreter.AllocateTensors();
    TfLiteTensor* input = interpreter.input(0);

    // Fill input tensor with sample data (e.g., rainfall and water level)
    input->data.f[0] = 200.0;  // Example rainfall in mm
    input->data.f[1] = 7.0;    // Example water level in meters

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Error: Model invocation failed.\n");
        return;
    }

    // Get the output tensor (flood probability)
    TfLiteTensor* output = interpreter.output(0);
    printf("Predicted flood probability: %f\n", output->data.f[0]);
}
