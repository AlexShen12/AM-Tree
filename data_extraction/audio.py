import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def find_visual_cuts(video_path, threshold=27.0):
    # 1. Create VideoManager and SceneManager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # 2. Add ContentDetector (Standard for visual cuts)
    # Lower threshold = more sensitive (detects smaller changes)
    # Higher threshold = less sensitive (only detects major shifts)
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    try:
        # Start the video manager and perform the analysis
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # 3. Retrieve the list of scenes
        scene_list = scene_manager.get_scene_list()
        
        print(f"Detected {len(scene_list)} scenes in {video_path}:")
        
        # Format: [(Start_Time, End_Time), ...]
        for i, scene in enumerate(scene_list):
            start, end = scene
            print(f"Scene {i+1}: Start {start.get_timecode()} | End {end.get_timecode()}")
            
        return scene_list

    finally:
        video_manager.release()

# Usage
find_visual_cuts("/users/a/l/alshen/VideoTree/VideoTree/data/VideoMME/data/_CqKv0Y1FB0.mp4")
