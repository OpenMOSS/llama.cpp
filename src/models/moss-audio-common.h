#pragma once

#include "models.h"

#include <cstdint>
#include <string>
#include <vector>

namespace moss_audio {

constexpr float LAYER_NORM_EPS = 1e-5f;

enum class module_type {
    PATCHED_PRETRANSFORM,
    TRANSFORMER,
};

struct transformer_layer {
    ggml_tensor * attn_in  = nullptr;
    ggml_tensor * attn_out = nullptr;
    ggml_tensor * linear1  = nullptr;
    ggml_tensor * linear2  = nullptr;
    ggml_tensor * norm1_w  = nullptr;
    ggml_tensor * norm1_b  = nullptr;
    ggml_tensor * norm2_w  = nullptr;
    ggml_tensor * norm2_b  = nullptr;
    ggml_tensor * scale1   = nullptr;
    ggml_tensor * scale2   = nullptr;
};

struct transformer_block {
    int input_dimension  = 0;
    int output_dimension = 0;
    int d_model          = 0;
    int num_heads        = 0;
    int num_layers       = 0;
    int context          = 0;
    float max_period     = 10000.0f;

    ggml_tensor * input_proj  = nullptr;
    ggml_tensor * output_proj = nullptr;

    std::vector<transformer_layer> layers;
};

struct module {
    module_type type = module_type::PATCHED_PRETRANSFORM;
    int patch_size = 1;
    transformer_block transformer;
};

struct quantizer_meta {
    int input_dim       = 0;
    int rvq_dim         = 0;
    int output_dim      = 0;
    int num_quantizers  = 0;
    int codebook_size   = 0;
    int codebook_dim    = 0;
};

std::string unquote(std::string value);
std::string meta_str(const llama_model & model, const std::string & key);
uint32_t meta_u32(const llama_model & model, const std::string & key);
float meta_f32(const llama_model & model, const std::string & key, float def = 0.0f, bool required = true);

ggml_tensor * require_tensor(const llama_model & model, const std::string & name);
ggml_tensor * optional_tensor(const llama_model & model, const std::string & name);
ggml_tensor * as_matrix(ggml_context * ctx0, ggml_tensor * tensor);
ggml_tensor * as_f32_matrix(ggml_context * ctx0, ggml_tensor * tensor);
ggml_tensor * as_f32_vector(ggml_context * ctx0, ggml_tensor * tensor);
ggml_tensor * linear_f32(ggml_context * ctx0, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias);

class graph_input_embd : public llm_graph_input_i {
public:
    explicit graph_input_embd(int64_t n_embd);

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * embd = nullptr;

private:
    int64_t n_embd;
};

class graph_input_channel : public llm_graph_input_i {
public:
    graph_input_channel(uint32_t channel, uint32_t n_channels);

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr;

private:
    uint32_t channel;
    uint32_t n_channels;
    std::vector<int32_t> data;
};

class graph_input_i32 : public llm_graph_input_i {
public:
    explicit graph_input_i32(std::vector<int32_t> data);

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tensor = nullptr;

private:
    std::vector<int32_t> data;
};

class graph_input_f32 : public llm_graph_input_i {
public:
    explicit graph_input_f32(std::vector<float> data);

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tensor = nullptr;

private:
    std::vector<float> data;
};

std::vector<int32_t> make_positions(size_t n_tokens);
int64_t align_up(int64_t value, int64_t multiple);
std::vector<float> make_causal_mask(size_t n_tokens, int context);

ggml_tensor * build_layer_norm(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias);
ggml_tensor * build_attention(
        ggml_context * ctx0,
        ggml_tensor * wo,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale);
ggml_tensor * patch_encode(ggml_context * ctx0, ggml_tensor * cur, int channels, int64_t n_frames, int patch_size);
ggml_tensor * patch_decode(ggml_context * ctx0, ggml_tensor * cur, int channels, int64_t n_frames, int patch_size);

quantizer_meta load_quantizer_meta(const llama_model & model, const std::string & arch_name);
std::vector<module> load_modules(
        const llama_model & model,
        ggml_context * ctx0,
        const std::string & arch_name,
        const std::string & section_name);

ggml_tensor * build_transformer_block(
        llm_graph_context & llm,
        ggml_tensor * cur,
        const transformer_block & block,
        int64_t n_frames,
        int module_index);

ggml_tensor * build_decoder_quantizer(
        llm_graph_context & llm,
        const llama_model & model,
        ggml_context * ctx0,
        const quantizer_meta & quantizer,
        int64_t n_frames);

ggml_tensor * build_l2_normalize(ggml_context * ctx0, ggml_tensor * cur);
ggml_tensor * build_encoder_quantizer_codes(
        const llama_model & model,
        ggml_context * ctx0,
        const quantizer_meta & quantizer,
        ggml_tensor * cur);

} // namespace moss_audio
