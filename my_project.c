#include "arm_nnfunctions.h"
#include "arm_math.h"
#include <stdio.h>

// Define the input dimensions
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
#define INPUT_CHANNELS 1

// Define the convolutional layer parameters
#define CONV1_OUT_CH 32
#define CONV1_FILTER_X 3
#define CONV1_FILTER_Y 3
#define CONV1_PAD_X 1
#define CONV1_PAD_Y 1
#define CONV1_STRIDE_X 1
#define CONV1_STRIDE_Y 1

// Define the pooling layer parameters
#define POOL1_FILTER_X 2
#define POOL1_FILTER_Y 2
#define POOL1_STRIDE_X 2
#define POOL1_STRIDE_Y 2

// Define the fully connected layer parameters
#define FC1_OUT_CH 10

// Allocate buffers for the layers
static q7_t input_data[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
static q7_t conv1_out[(INPUT_HEIGHT * INPUT_WIDTH * CONV1_OUT_CH)];
static q7_t pool1_out[((INPUT_HEIGHT / 2) * (INPUT_WIDTH / 2) * CONV1_OUT_CH)];
static q7_t fc1_out[FC1_OUT_CH];

// Weights and biases for the layers (dummy values for the example)
static q7_t conv1_wt[CONV1_OUT_CH * INPUT_CHANNELS * CONV1_FILTER_X * CONV1_FILTER_Y];
static q7_t conv1_bias[CONV1_OUT_CH];
static q7_t fc1_wt[FC1_OUT_CH * ((INPUT_HEIGHT / 2) * (INPUT_WIDTH / 2) * CONV1_OUT_CH)];
static q7_t fc1_bias[FC1_OUT_CH];

// Define the convolutional layer parameters
cmsis_nn_conv_params conv1_params;
cmsis_nn_per_channel_quant_params conv1_quant_params;
cmsis_nn_dims conv1_input_dims, conv1_filter_dims, conv1_bias_dims, conv1_output_dims;

cmsis_nn_pool_params pool1_params;
cmsis_nn_dims pool1_input_dims, pool1_filter_dims, pool1_output_dims;

cmsis_nn_fc_params fc1_params;
cmsis_nn_per_tensor_quant_params fc1_quant_params;
cmsis_nn_dims fc1_input_dims, fc1_filter_dims, fc1_bias_dims, fc1_output_dims;

void cnn_init() {
    // Initialize convolutional layer parameters
    conv1_params.padding.h = CONV1_PAD_Y;
    conv1_params.padding.w = CONV1_PAD_X;
    conv1_params.stride.h = CONV1_STRIDE_Y;
    conv1_params.stride.w = CONV1_STRIDE_X;
    conv1_params.dilation.h = 1;
    conv1_params.dilation.w = 1;
    conv1_params.input_offset = 0;
    conv1_params.output_offset = 0;
    conv1_params.activation.min = -128;
    conv1_params.activation.max = 127;

    conv1_quant_params.multiplier = NULL;  // Example: Set appropriate multiplier
    conv1_quant_params.shift = NULL;       // Example: Set appropriate shift

    conv1_input_dims.n = 1;
    conv1_input_dims.h = INPUT_HEIGHT;
    conv1_input_dims.w = INPUT_WIDTH;
    conv1_input_dims.c = INPUT_CHANNELS;

    conv1_filter_dims.n = CONV1_OUT_CH;
    conv1_filter_dims.h = CONV1_FILTER_Y;
    conv1_filter_dims.w = CONV1_FILTER_X;
    conv1_filter_dims.c = INPUT_CHANNELS;

    conv1_bias_dims.n = conv1_filter_dims.n;
    conv1_bias_dims.h = conv1_bias_dims.w = conv1_bias_dims.c = 1;

    conv1_output_dims.n = 1;
    conv1_output_dims.h = INPUT_HEIGHT;
    conv1_output_dims.w = INPUT_WIDTH;
    conv1_output_dims.c = CONV1_OUT_CH;

    // Initialize pooling layer parameters
    pool1_params.padding.h = 0;
    pool1_params.padding.w = 0;
    pool1_params.stride.h = POOL1_STRIDE_Y;
    pool1_params.stride.w = POOL1_STRIDE_X;
    pool1_params.activation.min = -128;
    pool1_params.activation.max = 127;

    pool1_input_dims.n = 1;
    pool1_input_dims.h = conv1_output_dims.h;
    pool1_input_dims.w = conv1_output_dims.w;
    pool1_input_dims.c = conv1_output_dims.c;

    pool1_filter_dims.h = POOL1_FILTER_Y;
    pool1_filter_dims.w = POOL1_FILTER_X;
    pool1_filter_dims.n = pool1_filter_dims.c = 1;

    pool1_output_dims.n = 1;
    pool1_output_dims.h = conv1_output_dims.h / 2;
    pool1_output_dims.w = conv1_output_dims.w / 2;
    pool1_output_dims.c = conv1_output_dims.c;

    // Initialize fully connected layer parameters
    fc1_params.input_offset = 0;
    fc1_params.filter_offset = 0;
    fc1_params.output_offset = 0;
    fc1_params.activation.min = -128;
    fc1_params.activation.max = 127;

    fc1_quant_params.multiplier = 1; // Example: Set appropriate multiplier
    fc1_quant_params.shift = 0;      // Example: Set appropriate shift

    fc1_input_dims.n = 1;
    fc1_input_dims.h = 1;
    fc1_input_dims.w = 1;
    fc1_input_dims.c = (pool1_output_dims.h * pool1_output_dims.w * pool1_output_dims.c);

    fc1_filter_dims.n = FC1_OUT_CH;
    fc1_filter_dims.h = 1;
    fc1_filter_dims.w = 1;
    fc1_filter_dims.c = fc1_input_dims.c;

    fc1_bias_dims.n = FC1_OUT_CH;
    fc1_bias_dims.h = fc1_bias_dims.w = fc1_bias_dims.c = 1;

    fc1_output_dims.n = 1;
    fc1_output_dims.h = 1;
    fc1_output_dims.w = 1;
    fc1_output_dims.c = FC1_OUT_CH;
}

void cnn_forward(q7_t *input_data) {
    // Convolutional layer
    cmsis_nn_context conv1_ctx;
    int32_t conv1_buffer_size = arm_convolve_s8_get_buffer_size(&conv1_input_dims, &conv1_filter_dims);
    q7_t conv1_buffer[conv1_buffer_size];
    conv1_ctx.buf = conv1_buffer;
    conv1_ctx.size = sizeof(conv1_buffer);

    arm_status status = arm_convolve_s8(&conv1_ctx,
                                        &conv1_params,
                                        &conv1_quant_params,
                                        &conv1_input_dims,
                                        input_data,
                                        &conv1_filter_dims,
                                        conv1_wt,
                                        &conv1_bias_dims,
                                        (int32_t*)conv1_bias,
                                        &conv1_output_dims,
                                        conv1_out);

    // Check for status of convolution
    if (status != ARM_MATH_SUCCESS) {
        printf("Convolution layer failed\n");
        return;
    }

    // Pooling layer
    cmsis_nn_context pool1_ctx;
    int32_t pool1_buffer_size = arm_avgpool_s8_get_buffer_size(pool1_output_dims.w, pool1_input_dims.c);
    q7_t pool1_buffer[pool1_buffer_size];
    pool1_ctx.buf = pool1_buffer;
    pool1_ctx.size = sizeof(pool1_buffer);

    status = arm_avgpool_s8(&pool1_ctx,
                            &pool1_params,
                            &pool1_input_dims,
                            conv1_out,
                            &pool1_filter_dims,
                            &pool1_output_dims,
                            pool1_out);

    if (status != ARM_MATH_SUCCESS) {
        printf("Pooling layer failed\n");
        return;
    }

    // Fully connected layer
    cmsis_nn_context fc1_ctx;
    int32_t fc1_buffer_size = arm_fully_connected_s8_get_buffer_size(&fc1_filter_dims);
    q7_t fc1_buffer[fc1_buffer_size];
    fc1_ctx.buf = fc1_buffer;
    fc1_ctx.size = sizeof(fc1_buffer);

    status = arm_fully_connected_s8(&fc1_ctx,
                                    &fc1_params,
                                    &fc1_quant_params,
                                    &fc1_input_dims,
                                    pool1_out,
                                    &fc1_filter_dims,
                                    fc1_wt,
                                    &fc1_bias_dims,
                                    (int32_t*)fc1_bias,
                                    &fc1_output_dims,
                                    fc1_out);

    if (status != ARM_MATH_SUCCESS) {
        printf("Fully connected layer failed\n");
        return;
    }

    // Softmax layer
    arm_softmax_s8(fc1_out, 1, FC1_OUT_CH, 1, 0, -128, fc1_out);
}

int main() {
    cnn_init();
    cnn_forward(input_data);
    for(int i = 0; i < FC1_OUT_CH; i++) {
        printf("%d ", fc1_out[i]);
    }
    return 0;
}
