import os
import h5py
import cv2  # OpenCV library for reading video files
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def add_video_to_hdf5(video_path, hdf5_file_path):
    # Read video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Convert video frames to NumPy array
    frames_array = np.array(frames)
    
    # Create HDF5 file and save dataset
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('video', data=frames_array)
    
    # Return the path of successfully processed file
    return hdf5_file_path

def traverse_directory_and_collect_files(base_path, file_list, max_files=10):
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            traverse_directory_and_collect_files(entry_path, file_list, max_files)
        elif entry_path.endswith('.mp4'):
            file_list.append(entry_path)
            if len(file_list) >= max_files:
                return

        # If file list has reached max files, return immediately
        if len(file_list) >= max_files:
            return

def process_files(file_list, output_base_path, input_base_path):
    with ThreadPoolExecutor() as executor:
        futures = []
        for video_path in file_list:
            # Ensure correct relative path conversion
            rel_path = os.path.relpath(video_path, input_base_path)
            output_path = os.path.join(output_base_path, rel_path).replace('.mp4', '.h5')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            futures.append(executor.submit(add_video_to_hdf5, video_path, output_path))
        
        for future in as_completed(futures):
            output_path = future.result()  # Wait for thread completion and get processed file path
            print(f"Completed: {output_path}")

def main(mp4_directory, output_directory, max_files):
    file_list = []
    traverse_directory_and_collect_files(mp4_directory, file_list, max_files)
    process_files(file_list, output_directory, mp4_directory)

if __name__ == "__main__":
    mp4_directory = "/path/to/mp4/videos"  # Replace with your MP4 folder path
    output_directory = "/path/to/output/hdf5"  # Replace with your desired HDF5 folder path
    max_files = 100  # Maximum number of files to convert
    main(mp4_directory, output_directory, max_files)
