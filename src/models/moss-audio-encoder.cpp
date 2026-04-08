#include "moss-audio-common.h"

using namespace moss_audio;

llm_build_moss_tts_audio_encoder::llm_build_moss_tts_audio_encoder(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const std::string arch_name = llm_arch_name(model.arch);
    const auto quantizer = load_quantizer_meta(model, arch_name);
    const std::vector<module> modules = load_modules(model, ctx0, arch_name, "encoder");
    const int64_t downsample_rate = (int64_t) meta_u32(model, arch_name + ".downsample_rate");
    const int64_t reserve_frames = ubatch.embd != nullptr ? ubatch.n_tokens : align_up(ubatch.n_tokens, downsample_rate);

    auto inp = std::make_unique<graph_input_embd>(1);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, reserve_frames);
    cb(inp->embd, "moss_audio_waveform", -1);
    ggml_set_input(inp->embd);
    ggml_tensor * cur = inp->embd;
    res->add_input(std::move(inp));

    int channels = 1;
    int64_t frames = reserve_frames;
    int tensor_block = 0;

    for (size_t i = 0; i < modules.size(); ++i) {
        const auto & current = modules[i];
        switch (current.type) {
            case module_type::PATCHED_PRETRANSFORM:
                cur = patch_encode(ctx0, cur, channels, frames, current.patch_size);
                channels *= current.patch_size;
                frames /= current.patch_size;
                break;
            case module_type::TRANSFORMER:
                cur = build_transformer_block(*this, cur, current.transformer, frames, tensor_block);
                channels = current.transformer.output_dimension;
                tensor_block++;
                break;
        }
    }

    GGML_ASSERT(channels == quantizer.input_dim);

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    ggml_tensor * codes = build_encoder_quantizer_codes(model, ctx0, quantizer, cur);
    cb(codes, "result_out_i32", -1);
    res->t_out_i32 = codes;
    ggml_build_forward_expand(gf, codes);
}
