import os
import subprocess
import numpy as np
import torch
import imageio_ffmpeg
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2AudioForConditionalGeneration

# Resolve the imageio-ffmpeg binary once at import time.  This binary is
# bundled with moviepy/imageio-ffmpeg so it works without system ffmpeg in PATH.
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# Qwen2-VL: produce a structured scene description, not a narrative summary.
# GPT needs concrete grounding — who, what, where — not impressionistic language.
_VISUAL_PROMPT = (
    "Describe this video clip as a scene. Cover: "
    "(1) the setting and environment (indoors/outdoors, location type, time of day if visible); "
    "(2) the main subjects — who or what they are and what they are doing; "
    "(3) any key objects, text on screen, or visual details that stand out. "
    "Be specific and concrete. Do not speculate beyond what is visible."
)

# Qwen2-Audio: produce a verbatim transcript plus a sound inventory.
# A paraphrase ("someone is talking about X") loses the exact content GPT needs
# to answer questions. Verbatim dialogue is far more informative for relevance scoring.
_AUDIO_PROMPT = (
    "Transcribe this audio clip accurately. "
    "Write out any spoken dialogue or narration verbatim inside quotation marks; "
    "label each speaker as 'Speaker 1:', 'Speaker 2:', etc. if multiple voices are present. "
    "After the transcript, list any significant non-speech sounds in brackets, "
    "for example: [upbeat music], [crowd cheering], [door closing]. "
    "If there is no speech, describe only the sounds. "
    "Do not paraphrase or summarise — transcribe exactly what is said."
)


class Qwen2VLDescriber:
    """
    Wraps Qwen2-VL to generate short visual descriptions of video clip segments.
    The model is loaded once at construction and reused across all describe_clip calls.
    """

    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def describe_clip(self, clip_path: str) -> str:
        """
        Generate a 1-2 sentence visual description of the clip at clip_path.
        Returns a placeholder string if the file does not exist.
        """
        if not os.path.exists(clip_path):
            return f"[visual clip not found: {clip_path}]"

        if process_vision_info is None:
            raise ImportError(
                "qwen_vl_utils is required for Qwen2VLDescriber. "
                "Install it from the Qwen2-VL repository."
            )

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": clip_path, "max_pixels": 360 * 420, "fps": 1.0},
                        {"type": "text", "text": _VISUAL_PROMPT},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)

            # strip the input tokens from each generated sequence
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        except Exception as e:
            print(f"[warn] Qwen2-VL could not read {clip_path}: {e}", flush=True)
            return "[visual description unavailable]"


class Qwen2AudioDescriber:
    """
    Wraps Qwen2-Audio to generate short audio descriptions of video clip segments.
    The model is loaded once at construction and reused across all describe_clip calls.
    Audio is extracted from the .mp4 clip using librosa.
    """

    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def describe_clip(self, clip_path: str) -> str:
        """
        Generate a 1-2 sentence audio description of the clip at clip_path.
        librosa handles audio extraction directly from .mp4 containers.
        Returns a placeholder string if the file does not exist.
        """
        if not os.path.exists(clip_path):
            return f"[audio clip not found: {clip_path}]"

        try:
            # Pipe audio directly from ffmpeg to a numpy buffer — single process,
            # no temp files, no video decoding overhead.
            # -f f32le: output raw float32 PCM so no int→float conversion is needed.
            # -ac 1:    downmix to mono before piping.
            # -ar:      resample to the model's expected sampling rate in one pass.
            # The imageio-ffmpeg binary is used so system ffmpeg in PATH is not required.
            cmd = [
                _FFMPEG_EXE,
                "-loglevel", "error",
                "-i", clip_path,
                "-vn",                          # skip video decoding entirely
                "-ar", str(self.sampling_rate),
                "-ac", "1",
                "-f", "f32le",
                "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio = np.frombuffer(result.stdout, dtype=np.float32)

            # Instruction and audio content both go in the user turn — Qwen2-Audio
            # resolves the audio_url placeholder against the `audios` list passed
            # to the processor, so the string value is irrelevant.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": _AUDIO_PROMPT},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=text,
                audios=[audio],
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                # 256 tokens: a 30-second clip at normal speech pace is ~75 words
                # (~100 tokens), so 256 provides headroom for longer clips and
                # the bracketed sound annotations that follow the transcript.
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)

            trimmed = generated_ids[:, inputs.input_ids.size(1):]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        except Exception as e:
            print(f"[warn] Qwen2-Audio could not read {clip_path}: {e}", flush=True)
            return "[audio description unavailable]"
