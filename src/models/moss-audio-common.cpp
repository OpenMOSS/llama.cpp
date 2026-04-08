#include "moss-audio-common.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace moss_audio {

std::string unquote(std::string value) {
    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
        return value.substr(1, value.size() - 2);
    }
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

std::string meta_str(const llama_model & model, const std::string & key) {
    const auto it = model.gguf_kv.find(key);
    if (it == model.gguf_kv.end()) {
        throw std::runtime_error("missing GGUF key: " + key);
    }
    return unquote(it->second);
}

uint32_t meta_u32(const llama_model & model, const std::string & key) {
    return (uint32_t) std::stoul(meta_str(model, key));
}

float meta_f32(const llama_model & model, const std::string & key, float def, bool required) {
    const auto it = model.gguf_kv.find(key);
    if (it == model.gguf_kv.end()) {
        if (!required) {
            return def;
        }
        throw std::runtime_error("missing GGUF key: " + key);
    }
    return std::stof(unquote(it->second));
}

ggml_tensor * require_tensor(const llama_model & model, const std::string & name) {
    auto * tensor = const_cast<ggml_tensor *>(model.get_tensor(name.c_str()));
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

ggml_tensor * optional_tensor(const llama_model & model, const std::string & name) {
    return const_cast<ggml_tensor *>(model.get_tensor(name.c_str()));
}

ggml_tensor * as_matrix(ggml_context * ctx0, ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }

    const int n_dims = ggml_n_dims(tensor);
    if (n_dims == 2) {
        return tensor;
    }
    if (n_dims == 3 && tensor->ne[0] == 1) {
        return ggml_reshape_2d(ctx0, tensor, tensor->ne[1], tensor->ne[2]);
    }
    if (n_dims == 4 && tensor->ne[0] == 1 && tensor->ne[1] == 1) {
        return ggml_reshape_2d(ctx0, tensor, tensor->ne[2], tensor->ne[3]);
    }

    throw std::runtime_error("unsupported tensor rank for linear projection: " + std::string(ggml_get_name(tensor)));
}

ggml_tensor * as_f32_matrix(ggml_context * ctx0, ggml_tensor * tensor) {
    return tensor != nullptr ? ggml_cast(ctx0, as_matrix(ctx0, tensor), GGML_TYPE_F32) : nullptr;
}

ggml_tensor * as_f32_vector(ggml_context * ctx0, ggml_tensor * tensor) {
    return tensor != nullptr ? ggml_cast(ctx0, tensor, GGML_TYPE_F32) : nullptr;
}

ggml_tensor * linear_f32(
        ggml_context * ctx0,
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * bias) {
    ggml_tensor * cur = ggml_cast(ctx0, input, GGML_TYPE_F32);

    if (weight != nullptr) {
        cur = ggml_mul_mat(ctx0, as_f32_matrix(ctx0, weight), cur);
    }
    if (bias != nullptr) {
        cur = ggml_add(ctx0, cur, as_f32_vector(ctx0, bias));
    }

    return cur;
}

graph_input_embd::graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}

void graph_input_embd::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ubatch->embd != nullptr);
    GGML_ASSERT(embd != nullptr);
    ggml_backend_tensor_set(embd, ubatch->embd, 0, (size_t) ubatch->n_tokens * (size_t) n_embd * sizeof(float));
}

bool graph_input_embd::can_reuse(const llm_graph_params & params) {
    return params.ubatch.embd != nullptr && embd != nullptr && embd->ne[0] == n_embd && embd->ne[1] == params.ubatch.n_tokens;
}

graph_input_channel::graph_input_channel(uint32_t channel, uint32_t n_channels) : channel(channel), n_channels(n_channels) {}

void graph_input_channel::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(tokens != nullptr);
    data.resize(ubatch->n_tokens, 0);

    if (ubatch->token_audio != nullptr) {
        GGML_ASSERT(ubatch->n_token_audio == n_channels);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = ubatch->token_audio[(size_t) i * n_channels + channel];
        }
    }

    ggml_backend_tensor_set(tokens, data.data(), 0, data.size() * sizeof(int32_t));
}

bool graph_input_channel::can_reuse(const llm_graph_params & params) {
    return tokens != nullptr &&
            tokens->ne[0] == params.ubatch.n_tokens &&
            ((params.ubatch.token_audio == nullptr && params.ubatch.n_token_audio == 0) ||
             (params.ubatch.token_audio != nullptr && params.ubatch.n_token_audio == n_channels));
}

graph_input_i32::graph_input_i32(std::vector<int32_t> data) : data(std::move(data)) {}

void graph_input_i32::set_input(const llama_ubatch *) {
    GGML_ASSERT(tensor != nullptr);
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(int32_t));
}

bool graph_input_i32::can_reuse(const llm_graph_params &) {
    return tensor != nullptr && tensor->ne[0] == (int64_t) data.size();
}

graph_input_f32::graph_input_f32(std::vector<float> data) : data(std::move(data)) {}

void graph_input_f32::set_input(const llama_ubatch *) {
    GGML_ASSERT(tensor != nullptr);
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

bool graph_input_f32::can_reuse(const llm_graph_params &) {
    return tensor != nullptr && ggml_nelements(tensor) == (int64_t) data.size();
}

std::vector<int32_t> make_positions(size_t n_tokens) {
    std::vector<int32_t> positions(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        positions[i] = (int32_t) i;
    }
    return positions;
}

int64_t align_up(int64_t value, int64_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

std::vector<float> make_causal_mask(size_t n_tokens, int context) {
    std::vector<float> mask(n_tokens * n_tokens, -std::numeric_limits<float>::infinity());

    for (size_t iq = 0; iq < n_tokens; ++iq) {
        for (size_t ik = 0; ik < n_tokens; ++ik) {
            if (ik > iq) {
                continue;
            }
            if (context > 0 && (int) (iq - ik) >= context) {
                continue;
            }
            mask[iq * n_tokens + ik] = 0.0f;
        }
    }

    return mask;
}

ggml_tensor * build_layer_norm(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * bias) {
    cur = ggml_norm(ctx0, cur, LAYER_NORM_EPS);
    cur = ggml_mul(ctx0, cur, weight);
    cur = ggml_add(ctx0, cur, bias);
    return cur;
}

ggml_tensor * build_attention(
        ggml_context * ctx0,
        ggml_tensor * wo,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale) {
    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
    v = ggml_cont(ctx0, v);

    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
    ggml_tensor * cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);

    if (wo != nullptr) {
        cur = ggml_mul_mat(ctx0, as_matrix(ctx0, wo), cur);
    }

    return cur;
}

ggml_tensor * patch_encode(
        ggml_context * ctx0,
        ggml_tensor * cur,
        int channels,
        int64_t n_frames,
        int patch_size) {
    GGML_ASSERT(patch_size > 0);
    GGML_ASSERT(n_frames % patch_size == 0);
    GGML_ASSERT(cur->ne[0] == channels);
    GGML_ASSERT(cur->ne[1] == n_frames);

    cur = ggml_reshape_3d(ctx0, cur, channels, patch_size, n_frames / patch_size);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, channels * patch_size, n_frames / patch_size);
    return cur;
}

ggml_tensor * patch_decode(
        ggml_context * ctx0,
        ggml_tensor * cur,
        int channels,
        int64_t n_frames,
        int patch_size) {
    GGML_ASSERT(patch_size > 0);
    GGML_ASSERT(channels % patch_size == 0);
    GGML_ASSERT(cur->ne[0] == channels);
    GGML_ASSERT(cur->ne[1] == n_frames);

    const int out_channels = channels / patch_size;
    cur = ggml_reshape_3d(ctx0, cur, patch_size, out_channels, n_frames);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, out_channels, n_frames * patch_size);
    return cur;
}

quantizer_meta load_quantizer_meta(const llama_model & model, const std::string & arch_name) {
    quantizer_meta meta;
    meta.input_dim = (int) meta_u32(model, arch_name + ".quantizer.input_dim");
    meta.rvq_dim = (int) meta_u32(model, arch_name + ".quantizer.rvq_dim");
    meta.output_dim = (int) meta_u32(model, arch_name + ".quantizer.output_dim");
    meta.num_quantizers = (int) meta_u32(model, arch_name + ".quantizer.num_quantizers");
    meta.codebook_size = (int) meta_u32(model, arch_name + ".quantizer.codebook_size");
    meta.codebook_dim = (int) meta_u32(model, arch_name + ".quantizer.codebook_dim");
    return meta;
}

std::vector<module> load_modules(
        const llama_model & model,
        ggml_context * ctx0,
        const std::string & arch_name,
        const std::string & section_name) {
    const auto tn = LLM_TN(model.arch);
    const uint32_t block_count = meta_u32(model, arch_name + "." + section_name + ".block_count");
    std::vector<module> modules(block_count);
    int tensor_block = 0;

    for (uint32_t ib = 0; ib < block_count; ++ib) {
        const std::string block_prefix = arch_name + "." + section_name + "." + std::to_string(ib);
        auto & block = modules[ib];
        const std::string current_type = meta_str(model, block_prefix + ".module_type");

        if (current_type == "PatchedPretransform") {
            block.type = module_type::PATCHED_PRETRANSFORM;
            block.patch_size = (int) meta_u32(model, block_prefix + ".patch_size");
            continue;
        }

        if (current_type != "Transformer") {
            throw std::runtime_error("unsupported MOSS audio module type: " + current_type);
        }

        block.type = module_type::TRANSFORMER;

        auto & tr = block.transformer;
        tr.input_dimension = (int) meta_u32(model, block_prefix + ".input_dimension");
        tr.output_dimension = (int) meta_u32(model, block_prefix + ".output_dimension");
        tr.d_model = (int) meta_u32(model, block_prefix + ".d_model");
        tr.num_heads = (int) meta_u32(model, block_prefix + ".num_heads");
        tr.num_layers = (int) meta_u32(model, block_prefix + ".num_layers");
        tr.context = (int) meta_u32(model, block_prefix + ".context");
        tr.max_period = meta_f32(model, block_prefix + ".max_period", 10000.0f, false);

        tr.input_proj = as_matrix(ctx0, optional_tensor(model,
                tn(LLM_TENSOR_MOSS_AUDIO_BLOCK_INPUT_PROJ, "weight", tensor_block).str()));
        tr.output_proj = as_matrix(ctx0, optional_tensor(model,
                tn(LLM_TENSOR_MOSS_AUDIO_BLOCK_OUTPUT_PROJ, "weight", tensor_block).str()));

        tr.layers.resize(tr.num_layers);
        for (int il = 0; il < tr.num_layers; ++il) {
            auto & layer = tr.layers[il];
            layer.attn_in  = as_matrix(ctx0, require_tensor(model,
                    tn(LLM_TENSOR_MOSS_AUDIO_ATTN_QKV, "weight", tensor_block, il).str()));
            layer.attn_out = as_matrix(ctx0, require_tensor(model,
                    tn(LLM_TENSOR_MOSS_AUDIO_ATTN_OUT, "weight", tensor_block, il).str()));
            layer.linear1  = as_matrix(ctx0, require_tensor(model,
                    tn(LLM_TENSOR_MOSS_AUDIO_FFN_UP, "weight", tensor_block, il).str()));
            layer.linear2  = as_matrix(ctx0, require_tensor(model,
                    tn(LLM_TENSOR_MOSS_AUDIO_FFN_DOWN, "weight", tensor_block, il).str()));
            layer.norm1_w  = require_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_ATTN_NORM, "weight", tensor_block, il).str());
            layer.norm1_b  = require_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_ATTN_NORM, "bias", tensor_block, il).str());
            layer.norm2_w  = require_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_FFN_NORM, "weight", tensor_block, il).str());
            layer.norm2_b  = require_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_FFN_NORM, "bias", tensor_block, il).str());
            layer.scale1   = optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_ATTN_SCALE, "scale", tensor_block, il).str());
            layer.scale2   = optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_FFN_SCALE, "scale", tensor_block, il).str());
        }

        tensor_block++;
    }

    return modules;
}

ggml_tensor * build_transformer_block(
        llm_graph_context & llm,
        ggml_tensor * cur,
        const transformer_block & block,
        int64_t n_frames,
        int module_index) {
    GGML_ASSERT(cur->ne[1] == n_frames);

    auto inp_pos = std::make_unique<graph_input_i32>(make_positions((size_t) n_frames));
    inp_pos->tensor = ggml_new_tensor_1d(llm.ctx0, GGML_TYPE_I32, n_frames);
    llm.cb(inp_pos->tensor, "moss_audio_pos", module_index);
    ggml_set_input(inp_pos->tensor);
    ggml_tensor * positions = inp_pos->tensor;
    llm.res->add_input(std::move(inp_pos));

    auto inp_mask = std::make_unique<graph_input_f32>(make_causal_mask((size_t) n_frames, block.context));
    inp_mask->tensor = ggml_new_tensor_4d(llm.ctx0, GGML_TYPE_F32, n_frames, n_frames, 1, 1);
    llm.cb(inp_mask->tensor, "moss_audio_mask", module_index);
    ggml_set_input(inp_mask->tensor);
    ggml_tensor * mask = inp_mask->tensor;
    llm.res->add_input(std::move(inp_mask));

    if (block.input_proj != nullptr) {
        cur = ggml_mul_mat(llm.ctx0, block.input_proj, cur);
    }

    const int d_head = block.d_model / block.num_heads;
    const float attn_scale = 1.0f / std::sqrt((float) d_head);

    for (int il = 0; il < block.num_layers; ++il) {
        const auto & layer = block.layers[il];

        ggml_tensor * inp_sa = cur;
        ggml_tensor * x = build_layer_norm(llm.ctx0, cur, layer.norm1_w, layer.norm1_b);
        ggml_tensor * qkv = ggml_mul_mat(llm.ctx0, layer.attn_in, x);

        ggml_tensor * q = ggml_view_3d(llm.ctx0, qkv, d_head, block.num_heads, n_frames,
                ggml_row_size(qkv->type, d_head), qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_3d(llm.ctx0, qkv, d_head, block.num_heads, n_frames,
                ggml_row_size(qkv->type, d_head), qkv->nb[1], ggml_row_size(qkv->type, block.d_model));
        ggml_tensor * v = ggml_view_3d(llm.ctx0, qkv, d_head, block.num_heads, n_frames,
                ggml_row_size(qkv->type, d_head), qkv->nb[1], ggml_row_size(qkv->type, 2 * block.d_model));

        q = ggml_rope_ext(llm.ctx0, q, positions, nullptr, d_head, 0, 0,
                block.max_period, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(llm.ctx0, k, positions, nullptr, d_head, 0, 0,
                block.max_period, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * attn = build_attention(llm.ctx0, layer.attn_out, q, k, v, mask, attn_scale);
        if (layer.scale1 != nullptr) {
            attn = ggml_mul(llm.ctx0, attn, layer.scale1);
        }
        cur = ggml_add(llm.ctx0, inp_sa, attn);

        ggml_tensor * inp_ff = cur;
        x = build_layer_norm(llm.ctx0, cur, layer.norm2_w, layer.norm2_b);
        x = ggml_mul_mat(llm.ctx0, layer.linear1, x);
        x = ggml_gelu(llm.ctx0, x);
        x = ggml_mul_mat(llm.ctx0, layer.linear2, x);
        if (layer.scale2 != nullptr) {
            x = ggml_mul(llm.ctx0, x, layer.scale2);
        }
        cur = ggml_add(llm.ctx0, inp_ff, x);
    }

    if (block.output_proj != nullptr) {
        cur = ggml_mul_mat(llm.ctx0, block.output_proj, cur);
    }

    return cur;
}

ggml_tensor * build_decoder_quantizer(
        llm_graph_context & llm,
        const llama_model & model,
        ggml_context * ctx0,
        const quantizer_meta & quantizer,
        int64_t n_frames) {
    const auto tn = LLM_TN(model.arch);

    ggml_tensor * cur = nullptr;
    for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
        auto inp_code = std::make_unique<graph_input_channel>((uint32_t) iq, (uint32_t) quantizer.num_quantizers);
        inp_code->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        llm.cb(inp_code->tokens, "moss_audio_code", iq);
        ggml_set_input(inp_code->tokens);

        ggml_tensor * codebook = ggml_cast(ctx0,
                require_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_CODEBOOK, "weight", -1, iq).str()),
                GGML_TYPE_F32);
        ggml_tensor * emb = ggml_get_rows(ctx0, codebook, inp_code->tokens);
        emb = linear_f32(
                ctx0,
                emb,
                optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_OUT_PROJ, "weight", -1, iq).str()),
                optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_OUT_PROJ, "bias", -1, iq).str()));

        cur = cur != nullptr ? ggml_add(ctx0, cur, emb) : emb;
        llm.res->add_input(std::move(inp_code));
    }

    cur = linear_f32(
            ctx0,
            cur,
            optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_OUTPUT_PROJ, "weight").str()),
            optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_OUTPUT_PROJ, "bias").str()));

    GGML_ASSERT(cur != nullptr);
    return cur;
}

ggml_tensor * build_l2_normalize(ggml_context * ctx0, ggml_tensor * cur) {
    constexpr float L2_NORM_EPS = 3.4526698e-4f;

    ggml_tensor * norm = ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur));
    norm = ggml_sqrt(ctx0, norm);
    norm = ggml_clamp(ctx0, norm, L2_NORM_EPS, INFINITY);
    return ggml_div(ctx0, cur, ggml_repeat(ctx0, norm, cur));
}

ggml_tensor * build_encoder_quantizer_codes(
        const llama_model & model,
        ggml_context * ctx0,
        const quantizer_meta & quantizer,
        ggml_tensor * cur) {
    const auto tn = LLM_TN(model.arch);

    cur = linear_f32(
            ctx0,
            cur,
            optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_INPUT_PROJ, "weight").str()),
            optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_INPUT_PROJ, "bias").str()));

    ggml_tensor * codes = nullptr;
    ggml_tensor * residual = cur;

    for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
        ggml_tensor * latent = linear_f32(
                ctx0,
                residual,
                optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_IN_PROJ, "weight", -1, iq).str()),
                optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_IN_PROJ, "bias", -1, iq).str()));

        ggml_tensor * latent_unit = build_l2_normalize(ctx0, latent);
        ggml_tensor * codebook = ggml_cast(ctx0,
                require_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_CODEBOOK, "weight", -1, iq).str()),
                GGML_TYPE_F32);
        ggml_tensor * codebook_unit = build_l2_normalize(ctx0, codebook);
        ggml_tensor * scores = ggml_mul_mat(ctx0, codebook_unit, latent_unit);
        ggml_tensor * code_i = ggml_argmax(ctx0, scores);

        ggml_tensor * code_i_row = ggml_reshape_2d(ctx0, code_i, 1, code_i->ne[0]);
        codes = codes != nullptr ? ggml_concat(ctx0, codes, code_i_row, 0) : code_i_row;

        ggml_tensor * decoded = ggml_get_rows(ctx0, codebook, code_i);
        decoded = linear_f32(
                ctx0,
                decoded,
                optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_OUT_PROJ, "weight", -1, iq).str()),
                optional_tensor(model, tn(LLM_TENSOR_MOSS_AUDIO_QUANT_OUT_PROJ, "bias", -1, iq).str()));
        residual = ggml_sub(ctx0, residual, decoded);
    }

    GGML_ASSERT(codes != nullptr);
    return codes;
}

} // namespace moss_audio
