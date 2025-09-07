#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 24
#define N_LAYER_2 64
#define N_LAYER_2 64
#define N_LAYER_2 64
#define N_LAYER_5 32
#define N_LAYER_5 32
#define N_LAYER_5 32
#define N_LAYER_8 1
#define N_LAYER_8 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<24,12> input_t;
typedef ap_fixed<24,12> qdense1_accum_t;
typedef ap_fixed<24,12> layer2_t;
typedef ap_fixed<8,4> weight2_t;
typedef ap_fixed<8,4> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,6> qdense1_alpha_result_t;
typedef struct exponent_scale10_t {ap_uint<1> sign;ap_int<5> weight; } exponent_scale10_t;
typedef ap_fixed<8,4> bias10_t;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> qact1_table_t;
typedef ap_fixed<24,12> qdense2_accum_t;
typedef ap_fixed<24,12> layer5_t;
typedef ap_fixed<8,3> weight5_t;
typedef ap_fixed<8,4> bias5_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<16,6> qdense2_alpha_result_t;
typedef struct exponent_scale11_t {ap_uint<1> sign;ap_int<4> weight; } exponent_scale11_t;
typedef ap_fixed<8,4> bias11_t;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> qact2_table_t;
typedef ap_fixed<24,12> y_accum_t;
typedef ap_fixed<24,12> layer8_t;
typedef ap_fixed<16,6> y_weight_t;
typedef ap_fixed<16,6> y_bias_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> y_linear_table_t;


#endif
