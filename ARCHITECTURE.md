# ARCHITECTURE.md

## Tech Stack
- **Framework**: PyTorch 2.6.0 (deep learning), Diffusers 0.30.1 (diffusion models)
- **Model Architecture**: Video Diffusion Transformer (DiT) based on Wan2.1
- **Audio Processing**: Wav2Vec2.0, Librosa, Audio-Separator
- **Computer Vision**: OpenCV, CLIP, Custom VAE
- **Infrastructure**: CUDA 12.4, Multi-GPU with xFuser, DeepSpeed optimization
- **Build Tools**: Python pip, HuggingFace Hub, Accelerate

## Directory Structure
```
StableAvatar/
├── wan/                          # Core model implementation
│   ├── configs/                  # Model configuration files (YAML)
│   ├── dataset/                  # Dataset loading and processing
│   ├── models/                   # Core model components
│   │   ├── wan_fantasy_transformer3d_1B.py    # 1.3B transformer
│   │   ├── vocal_projector_fantasy_1B.py      # Audio projection
│   │   └── wan_vae.py            # Video VAE encoder/decoder
│   ├── pipeline/                 # Inference pipelines
│   └── utils/                    # Utility functions
├── accelerate_config/            # Multi-GPU training configs
├── deepspeed_config/             # Memory optimization configs
├── examples/                     # Sample test cases
├── checkpoints/                  # Model weights (not in git)
├── inference.py                  # Main inference script
├── train_1B_square.py           # Training script (512x512)
├── train_1B_rec_vec.py          # Training script (mixed res)
├── audio_extractor.py           # Audio preprocessing
├── vocal_seperator.py           # Vocal separation
└── requirements.txt             # Python dependencies
```

## Key Architectural Decisions

### Time-step-aware Audio Adapter
**Context**: Existing models suffer from latent distribution drift in long videos
**Decision**: Implement novel time-step-aware modulation for audio processing
**Rationale**: Prevents error accumulation across video clips in infinite generation
**Consequences**: More complex training but enables true infinite-length generation

### Mixed Resolution Training
**Context**: Need to support multiple video aspect ratios
**Decision**: Train on 512x512, 480x832, and 832x480 simultaneously  
**Rationale**: Single model handles multiple formats vs separate models
**Consequences**: Higher VRAM requirements (50GB) but unified inference

### Audio Native Guidance
**Context**: Traditional guidance doesn't leverage diffusion's audio-latent prediction
**Decision**: Use evolving joint audio-latent prediction as dynamic guidance
**Rationale**: Better audio synchronization using model's own predictions
**Consequences**: More complex inference but superior lip sync quality

## Component Architecture

### WAN Transformer Structure <!-- #wan-transformer -->
```python
// Core model components with line references
class WanTransformer3DFantasyModel { /* wan/models/wan_fantasy_transformer3d_1B.py:1-800+ */ }  // <!-- #main-transformer -->
class VocalProjectorFantasy { /* wan/models/vocal_projector_fantasy_1B.py:1-450 */ }           // <!-- #audio-projector -->
class AutoencoderKLWan { /* wan/models/wan_vae.py:1-400+ */ }                                   // <!-- #vae-encoder -->
```

### Inference Pipeline Structure <!-- #inference-pipeline -->
```python
// Main inference components
class WanI2VTalkingInferenceLongPipeline { /* wan/pipeline/wan_inference_long_pipeline.py:1-800+ */ }  // <!-- #long-pipeline -->
def main() { /* inference.py:405-570 */ }                                                              // <!-- #inference-main -->
def parse_args() { /* inference.py:237-402 */ }                                                        // <!-- #arg-parser -->
```

## System Flow Diagram
```
[Reference Image] ──┐
                    ├──> [CLIP Encoder] ──┐
[Audio Input] ──────┴──> [Wav2Vec2] ──────┼──> [WAN Transformer] ──> [VAE Decoder] ──> [Generated Video]
                                          │            │
[Text Prompt] ─────────> [T5 Encoder] ────┘            │
                                                        │
[Noise Schedule] ──────────────────────────────────────┘

Training Flow:
[Video Frames] ──> [VAE Encoder] ──> [Latents] ──> [Add Noise] ──> [WAN Transformer] ──> [Loss Calculation]
      │                                  │                               │
      └──> [Face/Lip Masks] ─────────────┼───────────────────────────────┘
                                         │
[Audio] ──> [Wav2Vec2] ──> [Audio Projection] ──┘
```

## Common Patterns

### Model Loading Pattern
**When to use**: Loading pretrained models with custom configurations
**Implementation**: Use `from_pretrained` with `additional_kwargs` from OmegaConf
**Example**: `inference.py:473-485` - WanTransformer3DFantasyModel loading

### GPU Memory Management
**When to use**: Different hardware configurations or memory constraints  
**Implementation**: Conditional CPU offloading based on `--GPU_memory_mode`
**Example**: `inference.py:497-510` - Sequential/model CPU offload modes

### Multi-GPU Distributed Setup
**When to use**: Parallel inference or training across multiple GPUs
**Implementation**: xFuser integration with ulysses_degree and ring_degree
**Example**: `inference.py:408-439` - Distributed environment initialization

## Keywords <!-- #keywords -->
- architecture
- system design  
- tech stack
- components
- patterns
- diffusion transformer
- video generation
- audio-driven
- pytorch
- wan transformer
- vae encoder
- clip encoder
- wav2vec2
- multi-gpu
- memory optimization
- inference pipeline