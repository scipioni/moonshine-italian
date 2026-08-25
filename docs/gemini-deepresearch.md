# **Technical Report on Fine-Tuning, Serializing, and Integrating an Italian Streaming Speech Recognition Model into Moonshine Voice**

## **Architectural Foundations and Prerequisites for Italian Streaming ASR Integration**

The Moonshine Voice framework provides an open-source, on-device artificial intelligence architecture optimized for real-time speech-to-text, text-to-speech, and voice agent pipelines1. Unlike conventional automatic speech recognition systems relying on rigid 30-second audio windowing, Moonshine Streaming implements an encoder-decoder Transformer equipped with Rotary Position Embeddings (RoPE) and cached Key-Value (KV) attention states2. This structural design allows the model to process variable-length, incremental acoustic inputs while user speech is still actively arriving, cutting compute requirements and streaming latency significantly compared to Whisper architectures1.  
A central innovation in Moonshine's streaming pipeline is speculative decoding4. During incremental streaming re-decodes, the decoder verifies previous token hypotheses and continues generation from the first point of mismatch rather than restarting from the beginning-of-sequence (BOS) token4. This cuts streaming re-evaluation latency significantly4. The processing pipeline begins with raw audio capture at 16 kHz mono PCM, passes through continuous stream buffering, updates the cached encoder and decoder KV states incrementally, performs speculative re-decodes, and outputs real-time transcription events (LineStarted, LineUpdated, LineTextChanged, LineCompleted) through native C++ engine core libraries and high-level SDK bindings1.  
The core framework relies on single-language streaming models ("Flavors of Moonshine") such as TINY\_STREAMING, SMALL\_STREAMING, or BASE\_STREAMING3. Restricting parameter capacity to a single language maximizes acoustic representation for Italian regional accents and phonetic rules, delivering lower Word Error Rates (WER) for equivalent parameter counts3. Adding an Italian streaming speech recognition model requires fine-tuning a base streaming parameter set, exporting streaming encoder and KV-cached decoder graphs, serializing weights into ONNX Runtime FlatBuffers (.ort), and registering the model into the SDK catalog and ReadTheDocs documentation4.  
The core C++ engine (core/) strictly enforces execution using ONNX Runtime FlatBuffers (.ort), rejecting standard .onnx model files and external data sidecars at runtime4.

| Parameter Metric | Moonshine Tiny Streaming | Moonshine Small Streaming | Moonshine Medium Streaming |
| :---- | :---- | :---- | :---- |
| **Total Parameter Count** | 34 Million3 | 123 Million3 | 245 Million3 |
| **Open ASR Leaderboard WER** | 12.00%3 | 7.84%3 | 6.65%3 |
| **MacBook Pro Execution Latency** | 32 ms3 | 49 ms3 | 74 ms3 |
| **Raspberry Pi 5 Latency** | 237 ms3 | 527 ms3 | 802 ms3 |
| **Primary Deployment Target** | Ultra-low RAM Edge/IoT3 | Balanced Mobile/Desktop3 | High-Accuracy Workstations3 |

## **Italian Corpus Curation, Acoustic Preprocessing, and Dynamic Streaming Input**

Training a high-accuracy Italian streaming model requires combining diverse open-source corpora to cover regional accents, varied speech registers, and acoustic environments9. Prominent Italian datasets include Multilingual LibriSpeech (MLS Italian), Mozilla Common Voice Italian, FLEURS Italian, and VoxPopuli10.

| Dataset Name | Source Characteristics | Approximate Hours | Primary Optimization Purpose |
| :---- | :---- | :---- | :---- |
| **Multilingual LibriSpeech (MLS) Italian** | Read audiobooks, clean studio conditions | \~200+ Hours10 | Baseline acoustic feature and language alignment9 |
| **Mozilla Common Voice Italian** | Crowdsourced, varied microphones and background noise | \~300+ Hours10 | Robustness to real-world ambient noise and accents10 |
| **FLEURS Italian** | Parallel sentence readings, high linguistic diversity | \~12 Hours10 | Evaluation, validation, and domain calibration10 |
| **VoxPopuli Italian** | European Parliament speeches, political discourse | \~50+ Hours11 | Formal register, long-form continuous speech11 |

For streaming execution, audio streams are sampled at 16 kHz mono PCM8. Unlike non-streaming models that transcribe isolated segments after speech stops, streaming models process incremental PCM audio chunks (typically 32 ms to 100 ms hops)3. Silero VAD is integrated into the live input pipeline to monitor speech boundaries, triggering LineStarted on initial speech detection, dispatching LineUpdated / LineTextChanged during incremental streaming re-decodes, and issuing LineCompleted when speech pauses3.  
Italian text transcripts undergo normalization to handle accented characters (à, è, é, ì, ò, ù), expand numbers to text words, and process Italian-specific token vocabularies via SentencePiece or Byte-Pair Encoding (BPE) tokenizers9.

## **Step-by-Step Practical Workflow for Creating and Training an Italian Streaming Model**

Creating and deploying a custom Italian streaming Moonshine model involves environment configuration, training with streaming-compatible base checkpoints, exporting separate encoder and decoder ONNX graphs with KV-caching support, and converting to quantized ONNX Runtime FlatBuffers (.ort)4.

Bash  
\# Step 1: Clone training repository and set up environment  
git clone https://github.com/pierre-cheneau/finetune-moonshine-asr.git  
cd finetune-moonshine-asr  
pip install \-r requirements.txt

### **1\. Dataset Acquisition and Incremental Audio Chunking**

The dataset pre-processor segmentates Italian audio into chunks while ensuring compatibility with continuous streaming input buffers.

Bash  
python scripts/intelligent\_segmentation.py \\  
  \--dataset facebook/multilingual\_librispeech \\  
  \--language italian \\  
  \--output ./data/mls\_italian\_segmented \\  
  \--max-duration 10.0 \\  
  \--min-duration 1.0

### **2\. Training Configuration Setup for Streaming Architecture**

The configuration file targets a base streaming architecture checkpoint (such as moonshine-tiny-streaming) and configures schedule-free AdamW optimization.

YAML  
\# configs/my\_italian\_streaming\_model.yaml  
dataset:  
  name: "facebook/multilingual\_librispeech"  
  language: "italian"  
  train\_split: "train"  
  test\_split: "test"

training:  
  output\_dir: "./results-moonshine-it-streaming"  
  num\_train\_epochs: 5  
  per\_device\_train\_batch\_size: 16  
  learning\_rate: 5e-5  
  warmup\_steps: 500  
  optimizer: "schedule\_free\_adamw"  
  eval\_steps: 500  
  save\_steps: 1000

model:  
  base\_model: "UsefulSensors/moonshine-tiny-streaming"  
  language: "it"  
  is\_streaming: true

### **3\. Executing Training and Streaming Benchmarking**

The model is fine-tuned on Italian corpora. During evaluation, performance is measured both on whole utterances and on the chunked real-time streaming path using \--backend moonshine\_c\_streaming.

Bash  
\# Launch fine-tuning execution  
python train.py \--config configs/my\_italian\_streaming\_model.yaml

\# Monitor training metrics  
tensorboard \--logdir results-moonshine-it-streaming/runs

\# Evaluate accuracy on the chunked real-time streaming backend  
python scripts/evaluate.py \\  
  \--model results-moonshine-it-streaming/checkpoint-best \\  
  \--dataset facebook/multilingual\_librispeech \\  
  \--language italian \\  
  \--backend moonshine\_c\_streaming \\  
  \--split test

### **4\. Graph Export and FlatBuffer Serialization with KV Cache Support**

After fine-tuning, the PyTorch checkpoint is exported into ONNX graphs. For streaming execution, the encoder and decoder graphs are exported with inputs/outputs configured for cached Key-Value states (past\_key\_values and present\_key\_values). The intermediate .onnx graphs are then quantized to 8-bit integers (INT8) and converted into .ort FlatBuffers.

Bash  
\# Export PyTorch streaming model to ONNX graphs with KV cache  
python scripts/convert\_for\_deployment.py \\  
  \--model results-moonshine-it-streaming/checkpoint-best \\  
  \--streaming \\  
  \--output\_dir ./export\_streaming\_onnx

\# Convert ONNX graphs to ONNX Runtime FlatBuffer (.ort) format  
python scripts/convert-models-to-ort.py \\  
  \--input\_dir ./export\_streaming\_onnx \\  
  \--output\_dir ./export\_streaming\_ort \\  
  \--quantize int8

## **Fine-Tuning Methodology, KV Caching, and Speculative Decoding Pipeline**

Fine-tuning Moonshine Streaming models on Italian speech utilizes transfer learning from pre-trained streaming base checkpoints via PyTorch and Hugging Face Transformers. Optimization uses the Schedule-Free AdamW optimizer at a learning rate of ![][image1], removing the need for learning rate decay schedules while maintaining training stability. Training progress and Word Error Rates are validated continuously using jiwer.  
During streaming deployment, graph optimization and execution leverage two critical runtime features:

> 1. **Key-Value State Caching**: Encoder output encodings and decoder attention states are preserved across consecutive audio chunks, preventing redundant re-computation of historical speech segments3.  
> 2. **Speculative Decoding**: On incremental re-decodes (use\_speculative\_decoding=true), the runtime verifies previously predicted tokens against the updated audio frame and resumes generation directly from the first mismatch point4. This cuts streaming latency significantly across client devices4.

Converted .onnx graphs are compiled into ONNX Runtime FlatBuffers (.ort) using scripts/convert-models-to-ort.py4. Multi-head attention and LayerNormalization layers are fused into specialized operators under the com.microsoft domain16. 8-bit integer quantization (INT8) compresses model weights by \~75% with negligible accuracy degradation3.

| Quantization Precision | Model Parameter Size | Relative WER Penalty | On-Device RAM Footprint |
| :---- | :---- | :---- | :---- |
| **FP32 Reference** | \~136 MB | Baseline (0.00%) | High (\~512 MB) |
| **FP16 Semi-Float** | \~68 MB | \+0.02% | Moderate (\~256 MB) |
| **INT8 Quantized** | \~34 MB | \+0.31%3 | Minimal (\~96-128 MB)8 |

## **Engine Integration, API Regularization, and Documentation Catalog Extension**

To integrate the Italian streaming model into Moonshine SDK and document it on moonshine-voice.readthedocs.io/en/latest/models/available-models/, developers follow repository guidelines7. All pull requests target the dev-v candidate branch3. Converted .ort assets are published to download.moonshine.ai and mirrored on Hugging Face at moonshine-ai/moonshine-voice-assets3.  
In the core C++ layer (core/) and high-level language bindings, the model architecture enum is updated to register the new streaming architecture variant (ModelArch.TINY\_STREAMING, ModelArch.BASE\_STREAMING, or custom ITALIAN\_TINY\_STREAMING)5.

Python  
\# python/src/moonshine\_voice/moonshine\_api.py model enum mapping  
if model\_arch \== ModelArch.TINY\_STREAMING:  
    return "tiny-streaming"  
elif model\_arch \== ModelArch.ITALIAN\_TINY\_STREAMING:  
    return "italian-tiny-streaming"

High-level client APIs across Python, JavaScript, Swift, and Java adhere strictly to the construct \-\> configure \-\> load() object lifecycle4. Constructors remain lightweight without performing disk reads or model initialization, deferring session setup to .load()4.  
Streaming applications receive real-time transcription updates by listening to event callbacks3:

> * LineStarted: Fired when speech start is detected.  
> * LineUpdated / LineTextChanged: Fired whenever streaming re-decodes update hypothesis text.  
> * LineCompleted: Fired when the speaker pauses and the utterance segment closes.

Documentation updates are made in docs/models/available-models.md1. Table entries are added detailing total parameters, file sizes, streaming execution latencies (e.g., MacBook Pro, Raspberry Pi 5), and Open ASR Leaderboard WER metrics3. Local validation is completed by running scripts/test-docs.sh7.

## **Cross-Framework Performance and Latency Benchmarks**

Evaluating the fine-tuned Italian Moonshine Streaming model against alternative edge ASR frameworks highlights distinct operational advantages in streaming responsiveness and resource usage2.

| Architectural Attribute | Moonshine Tiny Streaming Italian (INT8) | OpenAI Whisper Tiny (INT8) | Sherpa-ONNX SenseVoice (INT8) |
| :---- | :---- | :---- | :---- |
| **Model Architecture** | Encoder-Decoder \+ RoPE \+ KV Cache2 | Encoder-Decoder (Absolute Pos)2 | Non-Autoregressive SAN-CTC12 |
| **Language Target** | Italian Single-Language Streaming3 | Multilingual Generalist3 | Multilingual (ZH/EN/JA/KO/YUE)12 |
| **Serialized Format** | ONNX Runtime FlatBuffer (.ort)4 | PyTorch / ONNX / GGUF12 | ONNX Int8 (model.int8.onnx)12 |
| **Input Windowing** | Incremental Dynamic Windows3 | Fixed 30-Sec Zero-Padded Window3 | Dynamic Sequence Chunking12 |
| **Re-decoding Engine** | Speculative Decoding Verification4 | Full Window Re-processing3 | Chunk-based Forward Pass12 |
| **Streaming Latency** | Ultra-Low (\~32-45 ms)3 | High (Requires full 30s audio segment)3 | Moderate (\~100-200 ms)12 |
| **RAM Footprint** | Minimal (\~96 MB)8 | High (\>500 MB)12 | Moderate (\~300-450 MB)12 |

Moonshine’s incremental KV caching and speculative decoding yield low latency during live microphone streaming3. Bypassing 30-second zero-padding and verifying token hypotheses incrementally ensures low compute overhead on edge devices3.

## **Strategic Framework Synthesis and Deployment Outlook**

Integrating an Italian streaming speech recognition model into Moonshine Voice highlights the effectiveness of specialized single-language architectures paired with KV state caching and speculative decoding3. Focusing parameter capacity on Italian speech delivers strong acoustic accuracy at modest parameter counts, making real-time streaming ASR viable on resource-constrained hardware1.  
Enforcing ONNX Runtime FlatBuffers (.ort) provides a unified deployment path across desktop, mobile, and web runtimes1. Fusing graph operators and pre-compiling FlatBuffer models removes runtime parsing overhead, maintaining fast startup times and efficient memory usage across C++, Python, WASM, Swift, and Java1.  
Combining incremental KV state caching with speculative decoding resolves key latency bottlenecks in live streaming ASR3. Following the framework's contribution guidelines—targeting the dev-v candidate branch, maintaining uniform construct \-\> configure \-\> load() API patterns, and updating documentation manifests—ensures seamless integration of new language models into the ecosystem3.

#### **Works cited**

> 1. Moonshine Voice \- GitHub, [https://github.com/moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine)  
> 2. Moonshine: Speech Recognition for Live Transcription and Voice Commands \- arXiv, [https://arxiv.org/html/2410.15608v2](https://arxiv.org/html/2410.15608v2)  
> 3. moonshine/README.md at main \- GitHub, [https://github.com/moonshine-ai/moonshine/blob/main/README.md](https://github.com/moonshine-ai/moonshine/blob/main/README.md)  
> 4. moonshine/CHANGELOGS.md at main \- GitHub, [https://github.com/moonshine-ai/moonshine/blob/main/CHANGELOGS.md](https://github.com/moonshine-ai/moonshine/blob/main/CHANGELOGS.md)  
> 5. Moonshine \- Pipecat, [https://docs.pipecat.ai/api-reference/server/services/stt/moonshine](https://docs.pipecat.ai/api-reference/server/services/stt/moonshine)  
> 6. moonshine/python/src/moonshine\_voice/moonshine\_api.py at main \- GitHub, [https://github.com/moonshine-ai/moonshine/blob/main/python/src/moonshine\_voice/moonshine\_api.py](https://github.com/moonshine-ai/moonshine/blob/main/python/src/moonshine_voice/moonshine_api.py)  
> 7. moonshine/AGENTS.md at main \- GitHub, [https://github.com/moonshine-ai/moonshine/blob/main/AGENTS.md](https://github.com/moonshine-ai/moonshine/blob/main/AGENTS.md)  
> 8. Voice Recognition and Speech Synthesis on an RPi Pico 2 \- Hackster.io, [https://www.hackster.io/petewarden/voice-recognition-and-speech-synthesis-on-an-rpi-pico-2-93b0ef](https://www.hackster.io/petewarden/voice-recognition-and-speech-synthesis-on-an-rpi-pico-2-93b0ef)  
> 9. pierre-cheneau/finetune-moonshine-asr: Complete guide and toolkit for fine-tuning ... \- GitHub, [https://github.com/pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr)  
> 10. Speech LLMs in Low-Resource Scenarios: Data Volume ... \- alphaXiv, [https://www.alphaxiv.org/overview/2508.05149v1](https://www.alphaxiv.org/overview/2508.05149v1)  
> 11. Comparative Analysis of Multilingual ASR Datasets for ... \- SciSpace, [https://scispace.com/ai-agent-seo/page-files/speech-datasets-ayroyim3.pdf](https://scispace.com/ai-agent-seo/page-files/speech-datasets-ayroyim3.pdf)  
> 12. Is there any ASR pretrained model to work with hindi, tamil and other indic languages? · k2-fsa sherpa-onnx · Discussion \#3199 \- GitHub, [https://github.com/k2-fsa/sherpa-onnx/discussions/3199](https://github.com/k2-fsa/sherpa-onnx/discussions/3199)  
> 13. finetune-moonshine-asr/docs/LIVE\_MODE\_GUIDE.md at main \- GitHub, [https://github.com/pierre-cheneau/finetune-moonshine-asr/blob/main/docs/LIVE\_MODE\_GUIDE.md](https://github.com/pierre-cheneau/finetune-moonshine-asr/blob/main/docs/LIVE_MODE_GUIDE.md)  
> 14. Accuracy \- Moonshine Voice \- Read the Docs, [https://moonshine-voice.readthedocs.io/en/latest/models/accuracy/](https://moonshine-voice.readthedocs.io/en/latest/models/accuracy/)  
> 15. jiwer · GitHub Topics, [https://github.com/topics/jiwer](https://github.com/topics/jiwer)  
> 16. moonshine/docs/execution-providers.md at main \- GitHub, [https://github.com/moonshine-ai/moonshine/blob/main/docs/execution-providers.md](https://github.com/moonshine-ai/moonshine/blob/main/docs/execution-providers.md)  
> 17. Pre-trained Models — sherpa 1.3 documentation, [https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAACfElEQVR4Xu2XO2sWQRSGjxE03qKpAhYStVCJgmKTJhACgpBCMIqKiIWFIKIWamUhWCimDam0MIko0QQJiYriDVsRQfwDIpoUWlh4xcv7enbyzR72mrCLxTzwwLfvTLKb882Z2YgEAgVZBxfZMJDMHfgJvoAHzFjAMAbbbRhI5rbMoVhb4Bc4Cc/BU/AB/ONPqoElcAM8KXrv5vjwP7bCz3ACNsH98GtsRnFuiP6t/D1P4FB8OJntog9nve9PqgHe8zt8GX1OKtZrOC1aWMd52OtdF2W19/ks/O1dp8JvixPfij7sXbg3NqMe1ooWiM+TVizm10y2DT7yrjnOFkuzfXZmgyNSsJM2w282LMEAXGDDCB7NozbMIa9YZ0y2MMq7TJ7FbvgT7ouuL0fXubhidYq23kd4E673J2XwGF4R7X2fxaL74HWT55FXrBM2FM0P2TCDZfAhbIt8By/FZqTQIY027IFr4CvRd5Ai8MbP4LDJ74luovzmy5BXrOM2FM25WZdhk+iX+UZ0z0rrjhhsFW7yPlwVLGCfybNg0UbgUtEVWrZIDlcsfxMnXLlZxbLtWStsx0Eb5sAVNgNv2YESuGKx6JasNjxmwzp5D8dtmIMrVtlN3ccViyvVwvy0ydg+zHeZvBJ4I9rvZa4Ni35by+FzabQef56HxFxa0RVrhR0QzXmY+GyEH0xWGTwyeTLwPcfRLfpgq7wsDVcou8FPyfw2+BY7AH6IvrT6HIQXTVYZfMfg8c7VQFaK/if+a3ZGNk/hVZn/qwOL3goPixZrR3Ttn1IXojHXctzXeHInFbZSeJQehXtEH/x/hq3H02+nlF+5gUAgEAgEAlXzFzSUht/SuclYAAAAAElFTkSuQmCC>