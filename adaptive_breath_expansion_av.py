import os
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from tqdm import tqdm
from pprint import pprint

from util import *
from eval import *
from dataset import get_dataset
from prompts import PromptFactory, parse_av_weights
from model import get_model
from av_models import Qwen2VLDescriber, Qwen2AudioDescriber


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_clip_features(video_id: str, clip_feat_path: str):
    """
    Load per-clip audio and visual embeddings from the VideoMME_clip_feature directory.

    Directory layout expected:
        {clip_feat_path}/{video_id}/visual/clip_NNN.pt   -> [1024] float32 tensor
        {clip_feat_path}/{video_id}/audio/clip_NNN.pt    -> [1024] float32 tensor

    Files are loaded in filename-sorted (i.e. temporal) order so that the
    returned 0-based clip index matches the temporal position in the video.

    Returns:
        visual_feats: [N_clips, 1024] float32 tensor
        audio_feats:  [N_clips, 1024] float32 tensor
    """
    vis_dir = os.path.join(clip_feat_path, video_id, "visual")
    aud_dir = os.path.join(clip_feat_path, video_id, "audio")

    vis_files = sorted(glob.glob(os.path.join(vis_dir, "clip_*.pt")))
    aud_files = sorted(glob.glob(os.path.join(aud_dir, "clip_*.pt")))

    assert len(vis_files) > 0, (
        f"No visual clip features found for '{video_id}' under {vis_dir}"
    )
    assert len(vis_files) == len(aud_files), (
        f"Clip count mismatch for '{video_id}': "
        f"{len(vis_files)} visual vs {len(aud_files)} audio"
    )

    visual_feats = torch.stack([torch.load(f) for f in vis_files])  # [N, 1024]
    audio_feats  = torch.stack([torch.load(f) for f in aud_files])  # [N, 1024]
    return visual_feats, audio_feats


# ---------------------------------------------------------------------------
# Feature fusion
# ---------------------------------------------------------------------------

def fuse_features(
    visual_feats: torch.Tensor,
    audio_feats: torch.Tensor,
    w_v: float,
    w_a: float,
) -> torch.Tensor:
    """
    Fuse visual and audio clip embeddings under adaptive per-modality weights.

    Each modality is L2-normalised before mixing so that magnitude differences
    between the Qwen2-VL and Qwen2-Audio embedding spaces do not distort the
    cosine geometry.

    Because both modality vectors are unit-norm, the energy contributed by
    each to the unnormalised sum scales with the *square* of the mixing
    coefficient, not the coefficient itself.  Using sqrt(w_v) and sqrt(w_a)
    as the actual coefficients therefore makes the weights directly
    interpretable as energy proportions:

        ||sqrt(w_v)·v̂ + sqrt(w_a)·â||²  =  w_v + w_a + 2·sqrt(w_v·w_a)·(v̂·â)
                                         →  w_v + w_a = 1   (when v̂ ⊥ â)

    Using linear weights instead would give w_v² + w_a², breaking the
    intuitive mapping from weight to proportion of influence.

    The result is L2-normalised to keep every fused vector on the unit sphere,
    which is required for cosine-distance K-means to behave consistently
    across loop iterations.

    Returns:
        fused: [N_clips, 1024] unit-normalised float32 tensor on the same
               device as the input tensors.
    """
    v_norm = F.normalize(visual_feats.float(), dim=1)
    a_norm = F.normalize(audio_feats.float(), dim=1)
    fused  = (w_v ** 0.5) * v_norm + (w_a ** 0.5) * a_norm
    return F.normalize(fused, dim=1)


# ---------------------------------------------------------------------------
# Cluster representative selection  (unchanged from adaptive_breath_expansion)
# ---------------------------------------------------------------------------

def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    """
    For each cluster find the single clip whose fused embedding is closest
    (L2) to the cluster centroid.

    Returns:
        dict: cluster_id -> [global_clip_index]
    """
    closest_per_cluster = {cid: [] for cid in range(len(cluster_centers))}

    for cluster_id in range(len(cluster_centers)):
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster  = x[indices_in_cluster]
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)

        if distances.numel() > 0:
            closest_idx_in_cluster = torch.argmin(distances).item()
            closest_global_idx     = indices_in_cluster[closest_idx_in_cluster].item()
            closest_per_cluster[cluster_id].append(closest_global_idx)

    return closest_per_cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_clip_path(clip_media_path: str, video_id: str, clip_idx: int) -> str:
    """
    Map a 0-based clip index to its .mp4 path.
    Filenames are 1-indexed and zero-padded to three digits, matching the
    feature file naming convention (clip_001.pt, clip_002.pt, …).
    """
    return os.path.join(clip_media_path, video_id, f"clip_{clip_idx + 1:03d}.mp4")


def format_clip_descriptions(
    tree_node: list,
    vis_descs: list,
    aud_descs: list,
) -> str:
    """
    Build the $clip_descriptions block substituted into the av_rel prompt.

    Each entry is labelled to match what the two models actually produce:
    - Scene description (Qwen2-VL): grounded who/what/where account of the visuals.
    - Transcript (Qwen2-Audio): verbatim dialogue in quotes + bracketed sound labels.
    """
    lines = []
    for rank, (idx, vis, aud) in enumerate(zip(tree_node, vis_descs, aud_descs), start=1):
        lines.append(f"Clip {rank} (index {idx}):")
        lines.append(f"  Scene description: {vis}")
        lines.append(f"  Audio transcript:  {aud}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def launch():
    args = parse_args()
    # Print args with the API key redacted so it never appears in slurm logs.
    printable = vars(args).copy()
    if printable.get("api_key"):
        printable["api_key"] = "***"
    pprint(printable)

    # output
    makedir(args.output_base_path)
    output_path           = os.path.join(args.output_base_path, args.output_filename)
    output_width_res_path = os.path.join(args.output_base_path, "width_res.json")

    # resume from a previous run
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if "data" in processed:
            processed = processed["data"]

    # dataset
    quids_to_exclude = set(processed.keys())
    dataset = get_dataset(
        args,
        quids_to_exclude=quids_to_exclude,
        num_examples_to_run=args.num_examples_to_run,
    )

    # prompt + LLM
    prompter = PromptFactory().get(args.prompt_type)
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)

    # audio-visual describers — loaded once and shared across all videos
    qwen_vl    = Qwen2VLDescriber(args.qwen_vl_model)    if args.qwen_vl_model    else None
    qwen_audio = Qwen2AudioDescriber(args.qwen_audio_model) if args.qwen_audio_model else None

    all_width_res = []

    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        ukey_name = "quid" if "quid" in item else "uid"
        ukey_1    = item[ukey_name]   # unique key per item (question-level for VideoMME/NExT)
        video_id  = item["uid"]       # raw video ID used for feature/media file lookups;
                                      # for VideoMME this is the YouTube ID matching the
                                      # clip_feat_path directory name (e.g. '-3t1rj8g6yg')

        # per-video cluster / fusion state
        tree_node       = [0]
        cluster_num     = args.init_cluster_num
        max_cluster_num = args.max_cluster_num
        iter_threshold  = args.iter_threshold
        adaptive_rate   = args.default_adpative_rate
        w_v             = args.init_visual_weight
        w_a             = args.init_audio_weight

        clip_length       = int(1 / args.fps) if args.fps < 1 else 1 / args.fps
        few_shot_examples = build_fewshot_examples(args.fewshot_example_path, args.data_path)

        # load all per-clip features for this video (CPU tensors, [N, 1024] each)
        visual_feats, audio_feats = load_clip_features(video_id, args.clip_feat_path)
        n_clips = visual_feats.size(0)

        # track descriptions for the final accepted tree_node
        vis_descs: list = []
        aud_descs: list = []
        pred        = None
        info        = {"response": None}
        prompt      = None
        clip_relevance: list = []

        # ------------------------------------------------------------------
        # Adaptive audio-visual width expansion loop
        # ------------------------------------------------------------------
        while True:
            # Cap cluster count to the number of available clips
            actual_cluster_num = min(cluster_num, n_clips)

            # 1. Fuse modalities under current weights → unit-sphere embeddings
            fused_feats = fuse_features(visual_feats, audio_feats, w_v, w_a).to("cuda")

            # 2. Cosine K-means on fused embeddings
            cluster_ids_x, cluster_centers = kmeans(
                X=fused_feats,
                num_clusters=actual_cluster_num,
                distance="cosine",
                device=torch.device("cuda:0"),
            )
            cluster_ids_x   = cluster_ids_x.to("cuda")
            cluster_centers = cluster_centers.to("cuda")

            # 3. Select the clip closest to each centroid; sort temporally
            closest_per_cluster = find_closest_points_per_cluster(
                fused_feats, cluster_ids_x, cluster_centers
            )
            if closest_per_cluster is None:
                continue
            tree_node = sorted(
                [v for sublist in closest_per_cluster.values() for v in sublist]
            )

            cluster_ids_x = cluster_ids_x.tolist()

            # 4. Generate visual + audio descriptions for representative clips only
            vis_descs = []
            aud_descs = []
            for clip_idx in tree_node:
                clip_path = build_clip_path(args.clip_media_path, video_id, clip_idx)
                vis_descs.append(
                    qwen_vl.describe_clip(clip_path)    if qwen_vl    else ""
                )
                aud_descs.append(
                    qwen_audio.describe_clip(clip_path) if qwen_audio else ""
                )

            clip_descriptions_str = format_clip_descriptions(tree_node, vis_descs, aud_descs)

            # 5. Build prompt and call the LLM for relevance scores + weight suggestions
            prompt = prompter.fill(
                **item,
                fps=args.fps,
                clip_length=clip_length,
                num_words=args.num_words_in_sum,
                examplars=few_shot_examples,
                loc_pred=tree_node,
                num_clips=len(tree_node),
                clip_descriptions=clip_descriptions_str,
            )
            pred, info = model.forward(prompter.head, prompt)

            clip_relevance = pred if isinstance(pred, list) else []
            high_relevance_count = clip_relevance.count(3)

            # 6. Stopping condition
            if high_relevance_count < iter_threshold:
                if cluster_num < max_cluster_num:
                    cluster_num = int(cluster_num * adaptive_rate)
                    w_v, w_a = parse_av_weights(info["response"])
                    continue
                else:
                    break  # cluster cap reached
            else:
                break  # sufficient high-relevance clips found

        # ------------------------------------------------------------------
        # Record results for this video
        # ------------------------------------------------------------------
        all_width_res.append({
            "name":                ukey_1,    # question-level key (quid or uid)
            "video_id":            video_id,  # raw video ID used for feature lookups
            "tree_node":           tree_node,
            "cluster_ids_x":      cluster_ids_x,
            "final_visual_weight": w_v,
            "final_audio_weight":  w_a,
            "final_cluster_num":   cluster_num,
        })

        ukey = item[ukey_name]
        processed[ukey] = item
        processed[ukey]["prompt"]            = prompt
        processed[ukey]["prompt_template"]   = prompter.get_template_str()
        processed[ukey]["response"]          = info["response"]
        processed[ukey]["pred"]              = pred
        processed[ukey]["clip_relevance"]    = clip_relevance
        processed[ukey]["final_visual_weight"] = w_v
        processed[ukey]["final_audio_weight"]  = w_a
        processed[ukey]["clip_descriptions"]   = [
            {"clip_idx": idx, "visual": vis, "audio": aud}
            for idx, vis, aud in zip(tree_node, vis_descs, aud_descs)
        ]
        processed[ukey]["tree_node"]          = tree_node
        processed[ukey]["cluster_ids_x"]      = cluster_ids_x
        processed[ukey]["final_cluster_num"]  = cluster_num
        if args.save_info:
            processed[ukey]["info"] = {
                k: v for k, v in info.items() if k != "response"
            }
        if i % args.save_every == 0:
            save_json(processed, output_path)

        pbar.update(1)

    save_json(all_width_res, output_width_res_path)

    # incorporate backup predictions for any failed cases
    if len(args.backup_pred_path) > 0:
        backup = load_json(args.backup_pred_path)
        if "data" in backup:
            backup = backup["data"]
        for uid in processed:
            if processed[uid]["pred"] == -1:
                processed[uid]["pred"] = backup[uid]["pred"]

    # evaluation
    if not args.disable_eval:
        if args.task == "qa":
            if args.dataset == "egoschema":
                processed = eval_qa_egoschema(processed)
            elif args.dataset in ["nextqa", "intentqa", "nextgqa"]:
                processed = eval_qa_nextqa(args.anno_path, processed)
        elif args.task == "gqa":
            if args.dataset == "nextgqa":
                pred_qa_path = (
                    args.nextgqa_pred_qa_path
                    if len(args.nextgqa_pred_qa_path) > 0
                    else None
                )
                processed = eval_gqa(
                    args.nextgqa_gt_ground_path, processed, pred_qa_path=pred_qa_path
                )
        elif args.task == "sum":
            processed, sum_data = eval_sum(processed)
            save_json(
                sum_data,
                f"{Path(output_path).parent / Path(output_path).stem}_data.json",
            )

    save_json(processed, output_path)


if __name__ == "__main__":
    launch()
