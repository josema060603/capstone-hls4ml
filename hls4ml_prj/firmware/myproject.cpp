#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t in_x[N_INPUT_1_1],
    result_t layer9_out[N_LAYER_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=in_x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=in_x,layer9_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 1536>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 64>(b2, "b2.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale10_t, 64>(s10, "s10.txt");
        nnet::load_weights_from_txt<bias10_t, 64>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight5_t, 2048>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 32>(b5, "b5.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale11_t, 32>(s11, "s11.txt");
        nnet::load_weights_from_txt<bias11_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<y_weight_t, 32>(w8, "w8.txt");
        nnet::load_weights_from_txt<y_bias_t, 1>(b8, "b8.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(in_x, layer2_out, w2, b2); // qdense1

    qdense1_alpha_result_t layer10_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::normalize<layer2_t, qdense1_alpha_result_t, config10>(layer2_out, layer10_out, s10, b10); // qdense1_alpha

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<qdense1_alpha_result_t, layer4_t, relu_config4>(layer10_out, layer4_out); // qact1

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // qdense2

    qdense2_alpha_result_t layer11_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::normalize<layer5_t, qdense2_alpha_result_t, config11>(layer5_out, layer11_out, s11, b11); // qdense2_alpha

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<qdense2_alpha_result_t, layer7_t, relu_config7>(layer11_out, layer7_out); // qact2

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // y

    nnet::linear<layer8_t, result_t, linear_config9>(layer8_out, layer9_out); // y_linear

}

