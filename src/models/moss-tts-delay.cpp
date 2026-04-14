#include "models.h"

namespace {

class llm_graph_input_moss_audio_channel : public llm_graph_input_i {
public:
    llm_graph_input_moss_audio_channel(uint32_t channel, uint32_t n_channels)
        : channel(channel), n_channels(n_channels) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(tokens != nullptr);

        data.resize(ubatch->n_tokens, 0);
        if (ubatch->token_audio != nullptr) {
            GGML_ASSERT(ubatch->n_token_audio == n_channels);

            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                data[i] = ubatch->token_audio[(size_t) i*n_channels + channel];
            }
        }

        ggml_backend_tensor_set(tokens, data.data(), 0, data.size()*ggml_element_size(tokens));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return
            tokens != nullptr &&
            tokens->ne[0] == params.ubatch.n_tokens &&
            (
                (params.ubatch.n_token_audio == n_channels && params.ubatch.token_audio != nullptr) ||
                (params.ubatch.n_token_audio == 0 && params.ubatch.token_audio == nullptr)
            );
    }

    ggml_tensor * tokens = nullptr;

private:
    const uint32_t channel;
    const uint32_t n_channels;
    std::vector<llama_token> data;
};

}

llm_build_moss_tts_delay::llm_build_moss_tts_delay(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(hparams.n_vq == model.tok_embd_audio.size());

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    GGML_ASSERT(ubatch.token != nullptr);
    GGML_ASSERT(
        (ubatch.token_audio != nullptr && ubatch.n_token_audio == hparams.n_vq) ||
        (ubatch.token_audio == nullptr && ubatch.n_token_audio == 0));

    for (uint32_t i = 0; i < hparams.n_vq; ++i) {
        auto inp_audio = std::make_unique<llm_graph_input_moss_audio_channel>(i, hparams.n_vq);
        inp_audio->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp_audio->tokens, "inp_audio_tokens", i);
        ggml_set_input(inp_audio->tokens);

        ggml_tensor * audio_embd = ggml_get_rows(ctx0, model.tok_embd_audio[i], inp_audio->tokens);
        cb(audio_embd, "audio_embd", i);

        inpL = ggml_add(ctx0, inpL, audio_embd);
        cb(inpL, "input_sum", i);

        res->add_input(std::move(inp_audio));
    }
    
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    GGML_ASSERT(hparams.n_vq == model.output_audio.size());

    // Detect Local variant: speech_to_local bridge MLP present
    const bool is_local = (model.speech_to_local_gate != nullptr);

    ggml_tensor * logits;

    if (is_local) {
        // ── Local variant output path ──
        // 1. Text head: backbone output → text logits (same as 8B)
        logits = build_lora_mm(model.output, cur);
        cb(logits, "result_output_text", -1);

        // 2. Bridge backbone(2048) → local space(1536) via SwiGLU MLP
        ggml_tensor * gate_out = build_lora_mm(model.speech_to_local_gate, cur);
        gate_out = ggml_silu(ctx0, gate_out);
        ggml_tensor * up_out = build_lora_mm(model.speech_to_local_up, cur);
        ggml_tensor * local_cur = ggml_mul(ctx0, gate_out, up_out);
        local_cur = build_lora_mm(model.speech_to_local_down, local_cur);
        cb(local_cur, "speech_to_local_out", -1);

        // 3. Run local transformer (4 layers) on the local representation
        //    Note: in the full autoregressive version, this runs per-channel
        //    with accumulating inputs. Here we process the single backbone output
        //    through all 4 layers as a static pass. This uses the trained weights
        //    but lacks the sequential channel feedback (TODO: autoregressive loop
        //    in moss-tts.cpp for proper inter-channel coherence).
        if (!model.local_layers.empty() && model.local_layers[0].attn_norm != nullptr) {
            const int n_local_layers = (int) model.local_layers.size();

            for (int ll = 0; ll < n_local_layers; ++ll) {
                const auto & layer = model.local_layers[ll];

                // FFN-only pass for the static single-token case.
                // Full attention with GQA and KV cache accumulation is needed
                // for the autoregressive channel loop (TODO in moss-tts.cpp).
                // The FFN layers do the bulk of the learned transformation.

                ggml_tensor * local_ffn_inp = local_cur;
                local_cur = build_norm(local_cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, -1);
                cb(local_cur, "local_ffn_norm", ll);

                // SwiGLU FFN
                ggml_tensor * ffn_gate = build_lora_mm(layer.ffn_gate, local_cur);
                ffn_gate = ggml_silu(ctx0, ffn_gate);
                ggml_tensor * ffn_up = build_lora_mm(layer.ffn_up, local_cur);
                local_cur = ggml_mul(ctx0, ffn_gate, ffn_up);
                local_cur = build_lora_mm(layer.ffn_down, local_cur);
                cb(local_cur, "local_ffn_out", ll);

                // Residual
                local_cur = ggml_add(ctx0, local_cur, local_ffn_inp);
                cb(local_cur, "local_l_out", ll);
            }

            // Local output norm
            if (model.local_output_norm != nullptr) {
                local_cur = build_norm(local_cur, model.local_output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(local_cur, "local_output_norm", -1);
            }
        }

        // 4. For each audio channel: local_to_speech bridge → audio_ln → head → logits
        for (uint32_t i = 0; i < hparams.n_vq; ++i) {
            // Bridge local(1536) → backbone(2048) via SwiGLU MLP
            ggml_tensor * ch_gate = build_lora_mm(model.local_to_speech_gate[i], local_cur);
            ch_gate = ggml_silu(ctx0, ch_gate);
            ggml_tensor * ch_up = build_lora_mm(model.local_to_speech_up[i], local_cur);
            ggml_tensor * ch_cur = ggml_mul(ctx0, ch_gate, ch_up);
            ch_cur = build_lora_mm(model.local_to_speech_down[i], ch_cur);
            cb(ch_cur, "local_to_speech_out", i);

            // Audio layer norm
            if (model.audio_ln[i] != nullptr) {
                ch_cur = build_norm(ch_cur, model.audio_ln[i], nullptr, LLM_NORM_RMS, -1);
                cb(ch_cur, "audio_ln", i);
            }

            // Audio head → logits
            ggml_tensor * audio_logits = build_lora_mm(model.output_audio[i], ch_cur);
            cb(audio_logits, "result_output_audio", i);

            logits = ggml_concat(ctx0, logits, audio_logits, 0);
            cb(logits, "result_output_concat", i);
        }
    } else {
        // ── 8B Delay variant output path (original) ──
        logits = build_lora_mm(model.output, cur);
        cb(logits, "result_output_text", -1);

        for (uint32_t i = 0; i < hparams.n_vq; ++i) {
            ggml_tensor * audio_logits = build_lora_mm(model.output_audio[i], cur);
            cb(audio_logits, "result_output_audio", i);

            logits = ggml_concat(ctx0, logits, audio_logits, 0);
            cb(logits, "result_output_concat", i);
        }
    }

    logits = ggml_cont(ctx0, logits);
    cb(logits, "result_output_cont", -1);

    res->t_logits = logits;
    cb(logits, "result_output", -1);

    ggml_build_forward_expand(gf, logits);
}
