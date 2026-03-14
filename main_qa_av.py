"""
AV QA for VideoMME: Answer multiple-choice questions using clip descriptions
from Qwen2-VL (visual) and Qwen2-Audio (audio).

Consumes depth_expansion_res_by_quid.json from depth_expansion_av.py.
For each question, loads the temporally sorted clip indices, describes each
clip with Qwen2-VL and Qwen2-Audio, then queries GPT with the vmme_av_qa prompt.
"""

import os
import json
from util import parse_args, makedir, load_json, save_json
from eval import eval_qa_videomme
from dataset import get_dataset
from prompts import PromptFactory
from model import get_model
from tqdm import tqdm
from pprint import pprint

from av_models import Qwen2VLDescriber, Qwen2AudioDescriber


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_clip_descriptions(clip_indices: list, visual_descs: list, audio_descs: list) -> str:
    """
    Build the $clip_descriptions block substituted into vmme_av_qa prompt.

    Args:
        clip_indices: Temporally sorted list of global clip indices.
        visual_descs: Qwen2-VL description for each clip (same order/length).
        audio_descs:  Qwen2-Audio description for each clip (same order/length).

    Returns:
        A multi-line string with one block per clip.
    """
    lines = []
    for rank, (idx, v_desc, a_desc) in enumerate(
        zip(clip_indices, visual_descs, audio_descs), start=1
    ):
        lines.append(
            f"Clip {rank} (index {idx}):\n  Visual: {v_desc}\n  Audio: {a_desc}"
        )
    return "\n\n".join(lines)


def build_clip_path(video_id: str, clip_idx: int, clip_media_path: str) -> str:
    """Build path to clip .mp4 file. Matches feature naming: clip_000.pt -> clip_000.mp4."""
    return os.path.join(clip_media_path, video_id, f"clip_{clip_idx:03d}.mp4")


def launch():
    args = parse_args()
    pprint(args)

    # Load tree node indices (depth expansion output)
    tree_node_idx_dict = {}
    if args.tree_node_idx:
        with open(args.tree_node_idx, "r") as f:
            tree_data = json.load(f)
        if isinstance(tree_data, list):
            tree_node_idx_dict = {
                item["name"]: item["sorted_values"] for item in tree_data
            }
        else:
            tree_node_idx_dict = {
                k: (v["sorted_values"] if isinstance(v, dict) else v)
                for k, v in tree_data.items()
            }

    if not args.tree_node_idx or not args.clip_media_path or not args.clip_feat_path:
        raise ValueError(
            "AV QA requires --tree_node_idx, --clip_media_path, and --clip_feat_path"
        )

    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)

    # Resume
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if "data" in processed:
            processed = processed["data"]

    quids_to_exclude = set(processed.keys())
    dataset = get_dataset(
        args, quids_to_exclude=quids_to_exclude, num_examples_to_run=args.num_examples_to_run
    )

    prompter = PromptFactory().get("vmme_av_qa")
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)

    # Load Qwen2-VL and Qwen2-Audio for clip description
    qwen_vl = Qwen2VLDescriber(args.qwen_vl_model)
    qwen_audio = Qwen2AudioDescriber(args.qwen_audio_model)

    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        quid = item["quid"]
        uid = item["uid"]

        entry = tree_node_idx_dict.get(quid)
        if entry is None:
            pred, info = -1, {"response": None}
            prompt = None
        else:
            loc = entry if isinstance(entry, list) else entry["sorted_values"]
            clip_paths = [
                build_clip_path(uid, idx, args.clip_media_path) for idx in loc
            ]
            visual_descs = [qwen_vl.describe_clip(p) for p in clip_paths]
            audio_descs = [qwen_audio.describe_clip(p) for p in clip_paths]
            clip_desc_str = format_clip_descriptions(loc, visual_descs, audio_descs)
            prompt = prompter.fill(**item, clip_descriptions=clip_desc_str)
            pred, info = model.forward(prompter.head, prompt)

        processed[quid] = item
        processed[quid]["prompt"] = prompt
        processed[quid]["prompt_template"] = prompter.get_template_str()
        processed[quid]["response"] = info["response"]
        processed[quid]["pred"] = pred
        if args.save_info:
            processed[quid]["info"] = {
                k: v for k, v in info.items() if k != "response"
            }
        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)

    # Incorporate backup prediction
    if len(args.backup_pred_path) > 0:
        backup = load_json(args.backup_pred_path)
        if "data" in backup:
            backup = backup["data"]
        for quid in processed:
            if processed[quid]["pred"] == -1 and quid in backup:
                processed[quid]["pred"] = backup[quid]["pred"]

    # Eval
    if not args.disable_eval:
        processed = eval_qa_videomme(processed)

    save_json(processed, output_path)
    print(f"Done. Results at {output_path}")


if __name__ == "__main__":
    launch()
