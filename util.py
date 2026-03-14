import os
import pickle
import json
from pathlib import Path
import argparse
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv

# Load .env from the repo root (the directory containing this file).
# Variables already set in the environment take precedence (override=False).
load_dotenv(Path(__file__).parent / ".env")


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", default='egoschema', type=str)  # 'egoschema', 'nextqa', 'nextgqa', 'intentqa'

    # subset
    parser.add_argument("--data_path", default='', type=str)
    parser.add_argument("--anno_path", default='VideoMME/videomme/test-00000-of-00001.parquet', type=str)
    parser.add_argument("--duration_path", default='', type=str) 

    # # fullset  
    # parser.add_argument("--data_path", default='/data/path/lavila_fullset.json', type=str) 
    # parser.add_argument("--anno_path", default='/data/path/fullset_anno.json', type=str)  
    # parser.add_argument("--duration_path", default='/data/path/duration.json', type=str) 
    parser.add_argument("--fps", default=1.0, type=float) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    ## backup pred
    parser.add_argument("--backup_pred_path", default="", type=str)
    ## fewshot
    parser.add_argument("--fewshot_example_path", default="", type=str) 
    ## nextgqa
    parser.add_argument("--nextgqa_gt_ground_path", default="", type=str)
    parser.add_argument("--nextgqa_pred_qa_path", default="", type=str)

    #cluster config
    parser.add_argument("--init_cluster_num", default=8, type=int)
    parser.add_argument("--max_cluster_num", default=32, type=int)
    parser.add_argument("--default_adpative_rate", default=2, type=int)
    parser.add_argument("--iter_threshold", default=4, type=int)

    #frame feature path
    parser.add_argument("--frame_feat_path", default="", type=str)

    # audio-visual expansion paths and models
    parser.add_argument("--clip_feat_path", default="data/VideoMME_clip_feature", type=str,
                        help="Root of VideoMME_clip_feature/ containing per-video audio/ and visual/ subdirs")
    parser.add_argument("--clip_media_path", default="data/VideoMME_clips", type=str,
                        help="Root directory of raw clip .mp4 segments for Qwen2-VL/Audio inference")
    parser.add_argument("--qwen_vl_model", default="Qwen/Qwen2-VL-7B-Instruct", type=str)
    parser.add_argument("--qwen_audio_model", default="Qwen/Qwen2-Audio-7B-Instruct", type=str)
    parser.add_argument("--init_visual_weight", default=0.5, type=float,
                        help="Initial fusion weight for visual embeddings (audio weight = 1 - this)")
    parser.add_argument("--init_audio_weight", default=0.5, type=float,
                        help="Initial fusion weight for audio embeddings")

    # VideoMME frame-based pipeline paths and models
    parser.add_argument("--frames_path", default="", type=str,
                        help="Root of VideoMME_frames/ containing per-video JPEG frame subdirectories")
    parser.add_argument("--llava_model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf", type=str,
                        help="HuggingFace model ID for LLaVA-OneVision frame captioner")

    # depth expansion (AV pipeline and VideoMME frame-based)
    parser.add_argument("--breadth_path", default="output/videomme_av_breath/breadth_expansion.json", type=str,
                        help="Path to breadth_expansion.json for depth expansion input")
    parser.add_argument("--width_res_path", default="", type=str,
                        help="Path to width_res.json; if empty, inferred as same dir as breadth_path")
    parser.add_argument("--num_subclusters", default=4, type=int,
                        help="Number of sub-clusters for relevance=2 clusters")
    parser.add_argument("--num_subsubclusters", default=4, type=int,
                        help="Number of sub-sub-clusters for relevance=3 clusters")

    # output
    parser.add_argument("--output_base_path", default="", type=str)  
    parser.add_argument("--output_filename", required=True, type=str)  

    # tree information
    parser.add_argument("--tree_node_idx", default="", type=str)  

    # prompting
    parser.add_argument("--model", default="gpt-4.1-mini", type=str)
    # Falls back to the OPENAI_API_KEY environment variable if not passed explicitly.
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--prompt_type", default="qa_standard", type=str)
    parser.add_argument("--task", default="qa", type=str)  # sum, qa, gqa
    ## sum
    parser.add_argument("--num_words_in_sum", default=500, type=int)  

    # other
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_info", action='store_true')
    parser.add_argument("--save_every", default=10, type=int)

    return parser.parse_args()


def build_fewshot_examples(qa_path, data_path):
    if len(qa_path) == 0 or len(data_path) == 0:
        return None
    qa = load_json(qa_path)
    data = load_json(data_path)  # uid --> str or list 
    examplars = []
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for i, (uid, examplar) in enumerate(qa.items()):
        description = data[uid]
        if isinstance(description, list):
            description = '. '.join(description)
        examplars.append(f"Examplar {i}.\n Descriptions: {description}.\n Question: {examplar['question']}\n A: {examplar['0']}\n B: {examplar['1']}\n C: {examplar['2']}\n D: {examplar['3']}\n E: {examplar['4']}\n Answer: {int_to_letter[examplar['truth']]}.")
    examplars = '\n\n'.join(examplars)
    return examplars
    
    
    