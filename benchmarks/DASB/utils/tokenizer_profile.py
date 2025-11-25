import torch
import time
import argparse
import torchaudio
from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from tokenizer_interface import (
    EncodecTokenizer,
    DACTokenizer,
    SpeechTokenizerWrapper,
    DiscreteSSLTokenizer,
    MimiTokenizer,
    WavTokenizerWrapper,
    SQCodecTokenizer,
)

def profile_model(model, input_tensor, device='cuda', encode_kwargs=None, decode_kwargs=None, num_trials=5):
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if encode_kwargs is None:
        encode_kwargs = {}
    if decode_kwargs is None:
        decode_kwargs = {}

    # Warm-up
    print(encode_kwargs)
    with torch.no_grad():
        tokens = model.sig_to_tokens(input_tensor, **encode_kwargs)
        print(tokens.shape)
        signal = model.tokens_to_sig(tokens, **decode_kwargs)
        print(signal.shape)

    # ---------------------
    # Measure Encode
    # ---------------------
    encode_times = []
    encode_mems = []
    for _ in range(num_trials):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            tokens = model.sig_to_tokens(input_tensor, **encode_kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB

        encode_times.append(elapsed)
        encode_mems.append(peak_mem)

    mean_encode_time = sum(encode_times) / len(encode_times)
    std_encode_time = (sum((t - mean_encode_time) ** 2 for t in encode_times) / len(encode_times)) ** 0.5
    mean_encode_mem = sum(encode_mems) / len(encode_mems)
    std_encode_mem = (sum((m - mean_encode_mem) ** 2 for m in encode_mems) / len(encode_mems)) ** 0.5

    # ---------------------
    # Measure Decode
    # ---------------------
    decode_times = []
    decode_mems = []
    for _ in range(num_trials):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model.tokens_to_sig(tokens, **decode_kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB

        decode_times.append(elapsed)
        decode_mems.append(peak_mem)

    mean_decode_time = sum(decode_times) / len(decode_times)
    std_decode_time = (sum((t - mean_decode_time) ** 2 for t in decode_times) / len(decode_times)) ** 0.5
    mean_decode_mem = sum(decode_mems) / len(decode_mems)
    std_decode_mem = (sum((m - mean_decode_mem) ** 2 for m in decode_mems) / len(decode_mems)) ** 0.5

    return (mean_encode_time, std_encode_time, mean_encode_mem, std_encode_mem), (mean_decode_time, std_decode_time, mean_decode_mem, std_decode_mem)

def resample_audio(audio, orig_sr, target_sr):
    """Resample audio tensor to a new sampling rate."""
    if orig_sr == target_sr:
        return audio
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a tokenizer encoder/decoder.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        choices=["Encodec", "DAC", "SpeechTokenizer", "DiscreteSSL", "Mimi", "WavTokenizer", "SQCodec"],
        help="Name of the tokenizer to profile.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        # nargs="+",   # <-- Accept one or more ints
        default=None,
        help="Number of codebooks to use (single int or list of ints). Leave empty to use default."
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Sampling rate for the input audio (default: 16000)."
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Number of trials for averaging timing and memory (default: 5)."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Always generate a fixed 16s of 16kHz fake audio
    fixed_sr = 16000
    torch.manual_seed(42)  # Ensure same audio every time
    fixed_audio = torch.randn(1, fixed_sr * 16)  # [B, T]

    # Resample to target sampling rate if needed
    input_audio = resample_audio(fixed_audio, fixed_sr, args.sampling_rate)
    print(input_audio.shape)
    # Load SSL model for DiscreteSSL
    # model_hub = "microsoft/wavlm-large"
    model_hub = "facebook/wav2vec2-large"
    save_path = "/network/scratch/a/ali.parviz/savedir"

    # ssl_model = Wav2Vec2(model_hub, save_path, output_all_hiddens=True).to(device)

    tokenizer_classes = {
        "Encodec": lambda: EncodecTokenizer("facebook/encodec_24khz", save_path,bandwidth=24.0),
        "DAC": lambda: DACTokenizer(load_pretrained=True, model_type="24KHz", model_bitrate="8kbps", tag="latest"),
        "SpeechTokenizer": lambda: SpeechTokenizerWrapper("fnlp/SpeechTokenizer", save_path),
        "DiscreteSSL": lambda: DiscreteSSLTokenizer(
            save_path,
            ssl_model="ssl_model",
            # vocoder_repo_id="speechbrain/hifigan-wavlm-k1000-LibriTTS",
            # vocoder_repo_id="speechbrain/hifigan-hubert-k1000-LibriTTS",
            vocoder_repo_id="speechbrain/hifigan-wav2vec2-k1000-LibriTTS",
            kmeans_dataset="LibriSpeech",
            num_clusters=1000,
            device=device
        ).to(device),
        "Mimi": lambda: MimiTokenizer("kyutai/mimi", save_path,num_codebooks=8),
        "WavTokenizer": lambda: WavTokenizerWrapper(
            "novateur/WavTokenizer-medium-music-audio-75token",
           save_path,
            config="wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
            checkpoint="wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
        ),
        "SQCodec": lambda: SQCodecTokenizer(
            "/network/scratch/a/ali.parviz/SQ-Codec",
            "config.yaml",
            "ckpt_00190000.pth"
        ),
    }

    tokenizer_name = args.tokenizer
    tokenizer_fn = tokenizer_classes[tokenizer_name]

    encode_kwargs = {}
    decode_kwargs = {}
    if args.num_codebooks is not None:
        encode_kwargs["num_codebooks"] = args.num_codebooks
    if tokenizer_name in ["Mimi","Encodec" ]:
         encode_kwargs["lengths"] = torch.tensor([1.0])
    
         


    tokenizer = tokenizer_fn()
    if tokenizer_name in ["DiscreteSSL" ]:
        tokenizer.codec_vocoder.device = device
        decode_kwargs["SSL_layers"] = args.num_codebooks
        
    # audio = torch.randn(4, 1000)
    # length = torch.tensor([1.0, .5, .75, 1.0])
    # model_hub = "facebook/encodec_24khz"
    # tokens= tokenizer.sig_to_tokens(input_audio, torch.tensor([1.0]))
    # print(tokens.shape)
    # rec = tokenizer.tokens_to_sig(tokens, lenght=torch.tensor([1.0]))
    # print(rec.shape)
    try:
        (mean_enc, std_enc, mean_mem_enc, std_mem_enc), (mean_dec, std_dec, mean_mem_dec, std_mem_dec) = profile_model(
            tokenizer, input_audio, device=device,
            encode_kwargs=encode_kwargs,
            decode_kwargs=decode_kwargs,
            num_trials=args.num_trials,
        )

        print("\n=== Profiling Result ===")
        print(f"Tokenizer: {tokenizer_name}")
        print(f"Sampling Rate: {args.sampling_rate}")
        print(f"Num Codebooks: {args.num_codebooks if args.num_codebooks else 'Default'}")
        print(f"Trials: {args.num_trials}")
        print(f"Encode Time: {mean_enc:.4f} ± {std_enc:.4f} s")
        print(f"Encode Memory: {mean_mem_enc:.4f} ± {std_mem_enc:.4f} GB")
        print(f"Decode Time: {mean_dec:.4f} ± {std_dec:.4f} s")
        print(f"Decode Memory: {mean_mem_dec:.4f} ± {std_mem_dec:.4f} GB")

    except Exception as e:
        print(f"Failed to profile {tokenizer_name}. Error: {e}")
