#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ref_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tokens;
    uint32_t n_vq;
    uint32_t n_embd;
    uint32_t n_logits;
};

constexpr uint32_t REF_MAGIC = 0x4d545452; // "RTTM"
constexpr uint32_t REF_VERSION = 1;

template <typename T>
void read_exact(std::ifstream & in, T * data, size_t count, const char * what) {
    in.read(reinterpret_cast<char *>(data), sizeof(T) * count);
    if (!in) {
        throw std::runtime_error(std::string("failed to read ") + what);
    }
}

float max_abs_diff(const float * got, const std::vector<float> & ref) {
    float out = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i])) {
            continue;
        }
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        out = std::max(out, std::fabs(got[i] - ref[i]));
    }
    return out;
}

float max_abs_diff_span(const float * got, const float * ref, size_t count) {
    float out = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(ref[i])) {
            continue;
        }
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        out = std::max(out, std::fabs(got[i] - ref[i]));
    }
    return out;
}

}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "error: failed to open reference '%s'\n", argv[2]);
        return EXIT_FAILURE;
    }

    ref_header hdr{};
    read_exact(in, &hdr, 1, "header");
    if (hdr.magic != REF_MAGIC || hdr.version != REF_VERSION) {
        std::fprintf(stderr, "error: unexpected reference format\n");
        return EXIT_FAILURE;
    }

    std::vector<llama_token> text(hdr.n_tokens);
    std::vector<llama_token> audio((size_t) hdr.n_tokens * hdr.n_vq);
    std::vector<float> ref_embd(hdr.n_embd);
    std::vector<float> ref_logits(hdr.n_logits);

    read_exact(in, text.data(),      text.size(),      "text tokens");
    read_exact(in, audio.data(),     audio.size(),     "audio tokens");
    read_exact(in, ref_embd.data(),  ref_embd.size(),  "reference embeddings");
    read_exact(in, ref_logits.data(), ref_logits.size(), "reference logits");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    // Keep this parity test deterministic and avoid multi-backend split-input limits.
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", argv[1]);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max<uint32_t>(hdr.n_tokens + 8, 64);
    cparams.n_batch = hdr.n_tokens;
    cparams.n_ubatch = hdr.n_tokens;
    cparams.n_seq_max = 1;
    cparams.embeddings = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_set_warmup(ctx, false);
    llama_set_embeddings(ctx, true);
    llama_set_causal_attn(ctx, true);

    llama_batch batch = llama_batch_init(hdr.n_tokens, 0, 1);
    batch.n_tokens = hdr.n_tokens;
    batch.n_token_audio = hdr.n_vq;
    batch.token_audio = (llama_token *) std::malloc(sizeof(llama_token) * audio.size());
    if (batch.token_audio == nullptr) {
        std::fprintf(stderr, "error: failed to allocate token_audio\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    for (uint32_t i = 0; i < hdr.n_tokens; ++i) {
        batch.token[i] = text[i];
        std::memcpy(batch.token_audio + (size_t) i * hdr.n_vq, audio.data() + (size_t) i * hdr.n_vq, sizeof(llama_token) * hdr.n_vq);
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == hdr.n_tokens;
    }

    const int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        std::fprintf(stderr, "error: llama_decode failed: %d\n", ret);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_audio_logits = (int32_t) (hdr.n_logits - n_vocab) / (int32_t) hdr.n_vq;
    const int32_t out_idx = (int32_t) hdr.n_tokens - 1;
    const float * got_embd = llama_get_embeddings_ith(ctx, out_idx);
    const float * got_logits = llama_get_logits_ith(ctx, out_idx);

    if (got_embd == nullptr || got_logits == nullptr) {
        std::fprintf(stderr, "error: missing outputs from context\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const float embd_max_abs = max_abs_diff(got_embd, ref_embd);
    const float logits_max_abs = max_abs_diff(got_logits, ref_logits);
    const float text_logits_max_abs = max_abs_diff_span(got_logits, ref_logits.data(), n_vocab);
    const float audio_logits_max_abs = max_abs_diff_span(got_logits + n_vocab, ref_logits.data() + n_vocab, hdr.n_logits - n_vocab);

    std::fprintf(stderr,
            "moss-tts-delay forward parity: out_idx=%d embd_max_abs=%g logits_max_abs=%g text_logits_max_abs=%g audio_logits_max_abs=%g n_audio_logits=%d\n",
            out_idx, embd_max_abs, logits_max_abs, text_logits_max_abs, audio_logits_max_abs, n_audio_logits);

    const bool ok = embd_max_abs < 1e-4f && logits_max_abs < 1e-4f;

    if (!ok) {
        for (uint32_t i = 0; i < hdr.n_tokens; ++i) {
            const float * got_embd_i = llama_get_embeddings_ith(ctx, (int32_t) i);
            if (got_embd_i != nullptr) {
                std::fprintf(stderr, "  embd_max_abs[out=%u]=%g\n", i, max_abs_diff(got_embd_i, ref_embd));
            }
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
