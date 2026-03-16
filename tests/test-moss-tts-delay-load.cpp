#include "llama.h"
#include "../src/llama-arch.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void check(bool cond, const std::string & msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

static void check_tensor_2d(const ggml_tensor * tensor, const char * name, int64_t ne0, int64_t ne1) {
    check(tensor != nullptr, std::string("missing tensor: ") + name);
    check(tensor->ne[0] == ne0, std::string(name) + " ne[0] mismatch");
    check(tensor->ne[1] == ne1, std::string(name) + " ne[1] mismatch");
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return EXIT_FAILURE;
    }

    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    params.use_mmap = false;

    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", argv[1]);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    try {
        check(model->arch == LLM_ARCH_MOSS_TTS_DELAY, "unexpected architecture");
        check(model->hparams.n_vq > 0, "n_vq must be > 0");
        check(model->hparams.audio_vocab_size > 0, "audio_vocab_size must be > 0");

        const int64_t n_embd = model->hparams.n_embd;
        const int64_t n_vocab = model->vocab.n_tokens();
        const int64_t n_audio_vocab = std::max<int64_t>(model->hparams.audio_vocab_size + 1, model->hparams.audio_pad_code + 1);

        check_tensor_2d(model->tok_embd, "token_embd.weight", n_embd, n_vocab);
        check_tensor_2d(model->output, "output.weight", n_embd, n_vocab);

        check(model->tok_embd_audio.size() == model->hparams.n_vq, "token_embd_audio size mismatch");
        check(model->output_audio.size() == model->hparams.n_vq, "output_audio size mismatch");

        for (uint32_t i = 0; i < model->hparams.n_vq; ++i) {
            check_tensor_2d(model->tok_embd_audio.at(i), "token_embd_audio", n_embd, n_audio_vocab);
            check_tensor_2d(model->output_audio.at(i), "output_audio", n_embd, n_audio_vocab);
        }

        std::fprintf(stderr,
            "loaded MOSS-TTS-Delay: n_layer=%u n_embd=%u n_vq=%u audio_vocab=%u tensors_ok=1\n",
            model->hparams.n_layer,
            model->hparams.n_embd,
            model->hparams.n_vq,
            model->hparams.audio_vocab_size);
    } catch (const std::exception & err) {
        std::fprintf(stderr, "validation failed: %s\n", err.what());
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
