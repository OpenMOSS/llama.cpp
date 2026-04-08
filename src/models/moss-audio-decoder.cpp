#include "moss-audio-common.h"

using namespace moss_audio;

llm_build_moss_tts_audio_decoder::llm_build_moss_tts_audio_decoder(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const std::string arch_name = llm_arch_name(model.arch);
    const auto quantizer = load_quantizer_meta(model, arch_name);
    const std::vector<module> modules = load_modules(model, ctx0, arch_name, "decoder");

    int64_t frames = ubatch.n_tokens;
    ggml_tensor * cur = build_decoder_quantizer(*this, model, ctx0, quantizer, frames);
    int channels = quantizer.output_dim;
    int tensor_block = 0;

    for (size_t i = 0; i < modules.size(); ++i) {
        const auto & current = modules[i];
        switch (current.type) {
            case module_type::TRANSFORMER:
                cur = build_transformer_block(*this, cur, current.transformer, frames, tensor_block);
                channels = current.transformer.output_dimension;
                tensor_block++;
                break;
            case module_type::PATCHED_PRETRANSFORM:
                cur = patch_decode(ctx0, cur, channels, frames, current.patch_size);
                channels /= current.patch_size;
                frames *= current.patch_size;
                break;
        }
    }

    GGML_ASSERT(channels == 1);

    cb(cur, "result_embd", -1);
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
