import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video at a specified frame rate.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        print(f"Error: Could not read video FPS for {video_path}")
        return
        
    frame_interval = int(fps / frame_rate) if frame_rate > 0 else 1
    
    count = 0
    saved_count = 0
    
    print(f"Processing {video_name} (FPS: {fps:.2f}, Saving 1 frame every {frame_interval} frames)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, '../data/test/frames')

    parser = argparse.ArgumentParser(description="Extract frames from videos to build a test dataset.")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file or directory containing videos.')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='Directory to save extracted frames.')
    parser.add_argument('--fps', type=float, default=1.0, help='Number of frames to extract per second of video.')
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    
    if os.path.isfile(input_path):
        extract_frames(input_path, output_dir, args.fps)
    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(input_path, file)
                extract_frames(video_path, output_dir, args.fps)
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")

if __name__ == '__main__':
    main()
