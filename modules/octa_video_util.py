# !pip install unidecode

from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import os
import threading
from datetime import datetime, timedelta
import cv2
import pandas as pd
import numpy as np
import shutil
from unidecode import unidecode

def formatted_blob_name(blob_name):
    words = blob_name.split(' ')
    return ' '.join(words[:-1] + [words[-1].replace('-', ':')])

def _matches_query(record, query_params):
    for field, values in query_params.items():
        if field not in record or (isinstance(values, list) and record[field] not in values) or (not isinstance(values, list) and record[field] != values):
            return False
    return True
    
def _filter_by_query(dataset, query_params):
    return dataset.apply(_matches_query, query_params=query_params, axis=1)

def filter_by_query(df, query):
    return df[_filter_by_query(df, query)]

def _assign_tag(tags_list, tags_priority_list, default_tag='normal'):
    if not tags_list:
        return default_tag
    for tag in tags_priority_list:
        if tag in tags_list:
            return tag
    return default_tag  # Default tag if none of the priority tags are found in tags_list

# from concurrent.futures import ThreadPoolExecutor
# from google.cloud import storage
# import os
# import threading

class VideoDownloader:
    def __init__(self, dataset, target_directory, credentials_path, max_threads=5):
        self.dataset = dataset
        self.target_directory = target_directory
        self.storage_client = storage.Client.from_service_account_json(credentials_path)
        self.max_threads = max_threads
        self.progress_lock = threading.Lock()
        self.total_files = None # len(self.dataset)
        self.current_file_index = 0

    def _update_progress(self):
        with self.progress_lock:
            self.current_file_index += 1
            percentage_progress = (self.current_file_index / self.total_files) * 100
            print(f"Downloaded {self.current_file_index}/{self.total_files} files ({percentage_progress:.2f}%)", end='\r')

    def download_video(self, row, overwrite=False):
        self._update_progress()
        bucket_name = row['bucket_name']
        blob_name = row['blob_name']

        destination_path = os.path.join(self.target_directory, blob_name)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Skip existing files if 'overwrite' is False
        if not overwrite and os.path.exists(destination_path):
            percentage_progress = (self.current_file_index / self.total_files) * 100
            print(f'File already exists, skipping... {self.current_file_index}/{self.total_files} files ({percentage_progress:.2f}%) - {destination_path}', end='\r')
            return

        try:
            blob = self.storage_client.bucket(bucket_name).blob(blob_name)
            blob.download_to_filename(destination_path)
        except:
            blob = self.storage_client.bucket(bucket_name).blob(formatted_blob_name(blob_name))
            blob.download_to_filename(destination_path)

    def download_videos(self, query_params=None, overwrite=False):
        filtered_dataset = self.dataset[self._filter_by_query(query_params)]
        self.total_files = len(filtered_dataset)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for _, row in filtered_dataset.iterrows():
                future = executor.submit(self.download_video, row, overwrite)
                futures.append(future)

            # Wait for all the futures (downloads) to complete
            for future in futures:
                future.result()

        print(f'DONE! {self.total_files}/{self.total_files} files downloaded.')
    
    def _matches_query(self, record, query_params):
        for field, values in query_params.items():
            if field not in record or (isinstance(values, list) and record[field] not in values) or (not isinstance(values, list) and record[field] != values):
                return False
        return True
        
    def _filter_by_query(self, query_params):
        return self.dataset.apply(self._matches_query, query_params=query_params, axis=1)

'''
# Example Usage

target_directory = 'data/videos/rotulados'
bucket_name = 'flood-video-collection'
query_params = {'code': list(cameras_with_labels)}
overwrite = False
credentials_path = '../../Flask APIs/cams-rio-api/auth/your-google-service-account-json.json'
max_threads = 100

# df_custom = df_label.copy()
# df_custom['blob_name'] = df_custom['blob_name'].str.replace('.webm', '.mp4')
# df_custom['bucket_name'] = 'flood-video-collection'

downloader = VideoDownloader(df_custom, target_directory, credentials_path, max_threads)
downloader.download_videos(query_params, overwrite)
'''

# import os
# from datetime import datetime, timedelta
# import cv2
# import shutil
# from concurrent.futures import ThreadPoolExecutor
# import threading
# from unidecode import unidecode

class VideoFrameExtractor:
    def __init__(self, dataset, base_directory, target_directory, max_threads=5):
        self.dataset = dataset
        self.base_directory = base_directory
        self.target_directory = target_directory
        self.max_threads = max_threads
        self.progress_lock = threading.Lock()
        self.total_rows = len(self.dataset)
        self.rows_processed = 0
        self.exists = 0
        self.not_found = 0
        self.total_frame_count = 0
        self.total_frames_written = 0
        self.frames_paths = []

    def _append_frame_path(self, path):
        with self.progress_lock:
            self.frames_paths.append(path)
    
    def _update_total_frame_count(self, count, written):
        with self.progress_lock:
            self.total_frame_count += count
            self.total_frames_written += written
            
    def _update_progress(self):
        with self.progress_lock:
            self.rows_processed += 1
            percentage_progress = (self.rows_processed / self.total_rows) * 100
            print(f"Processed {self.rows_processed}/{self.total_rows} rows ({percentage_progress:.2f}%)", end='\r')

    def _update_exists(self, frame_dir):
        with self.progress_lock:
            self.exists += 1
            percentage_exists = (self.exists / self.total_rows) * 100
            print(f"FRAME FOLDER ALREADY EXISTS {self.exists}/{self.total_rows} rows ({percentage_exists:.2f}%) {frame_dir}", end='\r')

    def _update_not_found(self, video_path):
        with self.progress_lock:
            self.not_found += 1
            percentage_not_found = (self.not_found / self.total_rows) * 100
            print(f"VIDEO FILE NOT FOUND {self.not_found}/{self.total_rows} rows ({percentage_not_found:.2f}%) {video_path}", end='\r')
    
    def extract_frames_for_row(self, record, overwrite=False, fps=3):
        self._update_progress()
        blob_name = record['blob_name']
        code = record['code']
        initial_timestamp_str = record['timestamp'] # Format: 'AAAA-MM-DD HH:MM:SS'
        initial_timestamp = datetime.strptime(initial_timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        # Get relative path without extension
        relative_path = unidecode(os.path.splitext(blob_name)[0])

        # Create a directory to save frames using the same internal folder structure
        frame_dir = os.path.join(self.target_directory, relative_path)

        # Skip if folder with frames already exists
        if not overwrite and os.path.exists(frame_dir):
            self._update_exists(frame_dir)
            # print(f'FILES ALREADY EXIST. SKIPPING.. {blob_name}.')
            return

        # Skip if video file is not found
        video_path = os.path.join(self.base_directory, blob_name)
        if not os.path.exists(video_path):
            self._update_not_found(video_path)
            # print(f'VIDEO FILE NOT FOUND. SKIPPING.. {video_path}.')
            return

        # Create target to store video frames if it doesn't exist
        os.makedirs(frame_dir, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video frame count
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get list of frames' timestamps
        offset_seconds = 1 / fps  # Calculate offset in seconds between frames
        offset_timedelta = timedelta(seconds=offset_seconds)

        frames_stamps = [initial_timestamp + i * offset_timedelta for i in range(video_frame_count)]

        # Initialize processed frames count
        frame_count = 0

        try:
            # Read and save frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    # The loop finished successfully
                    break

                frame_timestamp = frames_stamps[frame_count]
                formatted_timestamp = frame_timestamp.strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
                frame_name = f"CODE{int(code)} {formatted_timestamp}.jpg"
                frame_path = os.path.join(frame_dir, frame_name)
                success = cv2.imwrite(frame_path, frame)
                frame_count += 1
                
                self._append_frame_path((frame_path, success))

            cap.release()
            # print(f"Extracted {frame_count}/{video_frame_count} frames from {video_path}", end='\r')

        except KeyboardInterrupt:
            print("\n\nExtraction process interrupted. Cleaning up...")
            # Verify if all frames were written successfully
            if frame_count != video_frame_count:
                print(f"Failed to extract all frames. Removing folder... {frame_count}/{video_frame_count} {frame_dir}")
                # Remove any partially created directories
                shutil.rmtree(frame_dir)

            print("Extraction process cleaned up.")
            raise  # Re-raise the KeyboardInterrupt to exit the program

        self._update_total_frame_count(video_frame_count, frame_count)
    
    def extract_frames(self, overwrite=False):
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for _, record in self.dataset.iterrows():
                future = executor.submit(self.extract_frames_for_row, record, overwrite)
                futures.append(future)

            # Wait for all the futures (extraction tasks) to complete
            for future in futures:
                future.result()

        print(f'\n\nFINISHED.')
        print(f'\nFrames found: {self.total_frame_count}')
        print(f'Frames written to disk: {self.total_frames_written}')
        print(f'Frames folder exists: {self.exists}')
        print(f'Videos not found: {self.not_found}')
        
        self.rows_processed = 0
        self.exists = 0
        self.not_found = 0
        self.total_frame_count = 0
        self.total_frames_written = 0
            
            

'''
# Example usage

# from modules.video_collection import VideoFrameExtractor

# df_custom = df_label.copy()
# df_custom['blob_name'] = df_custom['blob_name'].str.replace('.webm', '.mp4')
# df_custom['bucket_name'] = 'flood-video-collection'

base_directory = 'data/videos/rotulados'
target_directory = 'data/imgs'
query_params = {'code': list(cameras_with_labels)}
# query_params = {'code': cameras_with_labels[0]}
df_filtered = df_custom[_filter_by_query(df_custom, query_params)]
overwrite = True
max_threads = 10

frame_extractor = VideoFrameExtractor(df_filtered, base_directory, target_directory, max_threads)
frame_extractor.extract_frames(overwrite)
'''

# import os
# from datetime import datetime, timedelta
# import pandas as pd
# import cv2
from unidecode import unidecode

def buildImageDataset(dataset, base_directory, fps=3, print_each=50):

    videos_exist = [os.path.exists(os.path.join(base_directory, blob_name)) for blob_name in dataset['blob_name']]
    dataset_filtered = dataset[videos_exist]

    frames_rows = []
    total_videos = len(dataset_filtered)
    processed_videos = 0
    for _, record in dataset_filtered.iterrows():
        blob_name = record['blob_name']
        code = record['code']
        initial_timestamp_str = record['timestamp'] # Format: 'AAAA-MM-DD HH:MM:SS'

        if not isinstance(initial_timestamp_str, str):
            processed_videos += 1
            continue    
        
        initial_timestamp = datetime.strptime(initial_timestamp_str, '%Y-%m-%d %H:%M:%S')
        seen = record['seen']
        tags = record['tags']
        id_video = record['_id']
        
        # Get relative path without extension
        relative_path = unidecode(os.path.splitext(blob_name)[0])
        
        # Get path to video file
        video_path = os.path.join(base_directory, blob_name)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video frame count
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Get list of frames' timestamps
        offset_seconds = 1 / fps  # Calculate offset in seconds between frames
        offset_timedelta = timedelta(seconds=offset_seconds)
    
        # Initialize processed frames count
        frame_count = 0
    
        # Initialize empty list for frames rows
        video_frames = []
        
        # Get and append frames info
        while frame_count < video_frame_count:
            # Get timestamp fields
            frame_timestamp = initial_timestamp + frame_count * offset_timedelta
            formatted_timestamp = frame_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]
            # file_formatted_timestamp = frame_timestamp.strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
            file_formatted_timestamp = frame_timestamp.strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
    
            # Get path fields
            frame_name = f"CODE{int(code)} {file_formatted_timestamp}.jpg"
            # frame_path = os.path.join(frame_dir, frame_name)
    
            # Create frame row
            frame_row = {
                'id_video': id_video,
                'code': code,
                'folder': relative_path,
                'file_name': frame_name,
                'file_path': os.path.join(relative_path, frame_name),
                'frame_index': frame_count,
                'timestamp': formatted_timestamp,
                'initial_timestamp': initial_timestamp_str,
                'seen': seen,
                'tags': tags,
            }
    
            # Append frame row
            video_frames.append(frame_row)
            frame_count += 1
    
        # Append video frames rows
        frames_rows.extend(video_frames)
        processed_videos += 1
        processed_percentage = round(processed_videos / total_videos * 100, 2)

        if processed_videos % print_each == 0:
            print(f'Processed videos: {processed_videos}/{total_videos} ({processed_percentage}) %', end='\r')
    
    df_imgs = pd.DataFrame(frames_rows)

    print(f'Processed videos: {processed_videos}/{total_videos} ({processed_percentage}) %', end='\r')
    print('\n') # formatted output
    return df_imgs

'''
# Example Usage

# from modules.video_collection import buildImageDataset

dataset = df_custom.dropna(subset=['timestamp']).copy()
base_directory = 'data/videos/rotulados'
target_directory = 'data/imgs'
fps = 3

df_images =  buildImageDataset(dataset, base_directory, target_directory, fps=3)

# Create unique tag column
tags_priority_list = ['alagamento', 'bolsão', 'lâmina', 'transbordo', 'poça']
df_images['tag'] = df_images['tags'].apply(lambda tags_list: _assign_tag(tags_list, tags_priority_list))

# Save images dataset
df_images.to_csv('data/datasets/images.csv', index=False)

# Print results
print('Image dataset shape:', df_images.shape)
print('Unique tags', df_images.tag.value_counts())
'''

# import os
# from datetime import datetime, timedelta
# import pandas as pd
# import cv2
# from concurrent.futures import ThreadPoolExecutor
# import numpy as np
from unidecode import unidecode

def process_video(record, base_directory, fps):
    blob_name = record['blob_name']
    code = record['code']
    initial_timestamp_str = record['timestamp']
    if not isinstance(initial_timestamp_str, str):
        return []

    initial_timestamp = datetime.strptime(initial_timestamp_str, '%Y-%m-%d %H:%M:%S')
    seen = record['seen']
    tags = record['tags']
    id_video = record['_id']
    video_frames = []

    # Get relative path without extension
    relative_path = unidecode(os.path.splitext(blob_name)[0])

    # frame_dir = os.path.join(target_directory, relative_path)
    video_path = os.path.join(base_directory, blob_name)

    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    offset_seconds = 1 / fps
    offset_timedelta = timedelta(seconds=offset_seconds)
    frame_count = 0
    while frame_count < video_frame_count:
        frame_timestamp = initial_timestamp + frame_count * offset_timedelta
        formatted_timestamp = frame_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]
        file_formatted_timestamp = frame_timestamp.strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
        frame_name = f"CODE{int(code)} {file_formatted_timestamp}.jpg"
        # frame_path = os.path.join(frame_dir, frame_name)

        frame_row = {
            'id_video': id_video,
            'code': code,
            'folder': relative_path,
            'file_name': frame_name,
            'file_path': os.path.join(relative_path, frame_name),
            'frame_index': frame_count,
            'timestamp': formatted_timestamp,
            'initial_timestamp': initial_timestamp_str,
            'seen': seen,
            'tags': tags,
        }
        video_frames.append(frame_row)
        frame_count += 1

    return video_frames

def buildImageDatasetThreads(dataset, base_directory, target_directory, fps=3, print_each=50, max_threads=1):

    videos_exist = [os.path.exists(os.path.join(base_directory, blob_name)) for blob_name in dataset['blob_name']]
    dataset_filtered = dataset[videos_exist]
    
    frames_rows = []
    total_videos = len(dataset_filtered)
    processed_videos = 0

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for _, record in dataset_filtered.iterrows():
            future = executor.submit(process_video, record, base_directory, fps)
            futures.append(future)

        for future in futures:
            video_frames = future.result()
            frames_rows.extend(video_frames)
            processed_videos += 1
            processed_percentage = round(processed_videos / total_videos * 100, 2)

            if processed_videos % print_each == 0:
                print(f'Processed videos: {processed_videos}/{total_videos} ({processed_percentage}) %', end='\r')

    df_imgs = pd.DataFrame(frames_rows)

    print(f'Processed videos: {processed_videos}/{total_videos} ({processed_percentage}) %', end='\r')
    print() # formatted output
    return df_imgs

'''
# Example Usage
 
# from modules.video_collection import buildImageDatasetThreads

dataset = df_custom.dropna(subset=['timestamp']).copy()
base_directory = 'data/videos/rotulados'
target_directory = 'data/imgs'
fps = 3
print_each = 50
max_threads = 5

df_images = buildImageDatasetThreads(dataset, base_directory, target_directory, fps, print_each, max_threads)

# Create unique tag column
tags_priority_list = ['alagamento', 'bolsão', 'lâmina', 'transbordo', 'poça']
df_images['tag'] = df_images['tags'].apply(lambda tags_list: _assign_tag(tags_list, tags_priority_list))

# Save images dataset
df_images.to_csv('data/datasets/images.csv', index=False)

# Print results
print('\nImage dataset shape:', df_images.shape)
print('Unique tags', df_images.tag.value_counts())
'''

# import os
# import shutil

def copy_images_to_folders(base_directory, target_directory, dataset, train_index=None, test_index=None, val_index=None, file_path_field='file_path', tag_field='tag'):
    
    if train_index is not None:
        train_output_dir = os.path.join(target_directory, 'train')
        os.makedirs(train_output_dir, exist_ok=True)
        total_train_files = len(train_index)
        not_found_train = 0
        
        # Copy images to train folder
        print("Copying images to train folders:")
        for i, idx in enumerate(train_index):
            file_path = dataset.loc[idx][file_path_field]
            input_path = os.path.join(base_directory, file_path)
            if not os.path.exists(input_path):
                # print('File not found error:', input_path, end='\r')
                not_found_train += 1
                continue
            tag = dataset.loc[idx][tag_field]
            output_folder = os.path.join(train_output_dir, str(tag))
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            shutil.copy(input_path, output_path)
    
            # Print progress (absolute and percentual)
            print(f"Processed {i+1}/{total_train_files} files ({(i+1)/total_train_files*100:.2f}%) - Found: {i + 1 - not_found_train}/{total_train_files}", end='\r')
        print(f"Processed {i+1}/{total_train_files} files ({(i+1)/total_train_files*100:.2f}%) - Found: {i + 1 - not_found_train}/{total_train_files}", end='\r')

    if test_index is not None:
        test_output_dir = os.path.join(target_directory, 'test')
        os.makedirs(test_output_dir, exist_ok=True)
        total_test_files = len(test_index)
        not_found_test = 0

        print("\nCopying images to test folders:")
        # Copy images to test folder
        for i, idx in enumerate(test_index):
            file_path = dataset.loc[idx][file_path_field]
            input_path = os.path.join(base_directory, file_path)
            if not os.path.exists(input_path):
                # print('File not found error:', input_path, end='\r')
                not_found_test += 1
                continue
            tag = dataset.loc[idx][tag_field]
            output_folder = os.path.join(test_output_dir, str(tag))
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            shutil.copy(input_path, output_path)
    
            # Print progress (absolute and percentual)
            print(f"Processed {i+1}/{total_test_files} files ({(i+1)/total_test_files*100:.2f}%) - Found: {i + 1 - not_found_test}/{total_test_files}", end='\r')
        print(f"Processed {i+1}/{total_test_files} files ({(i+1)/total_test_files*100:.2f}%) - Found: {i + 1 - not_found_test}/{total_test_files}", end='\r')

    if val_index is not None:
        val_output_dir = os.path.join(target_directory, 'val')
        os.makedirs(val_output_dir, exist_ok=True)
        total_val_files = len(val_index)
        not_found_val = 0

        print("\nCopying images to val folders:")
        # Copy images to test folder
        for i, idx in enumerate(val_index):
            file_path = dataset.loc[idx][file_path_field]
            input_path = os.path.join(base_directory, file_path)
            if not os.path.exists(input_path):
                # print('File not found error:', input_path, end='\r')
                not_found_val += 1
                continue
            tag = dataset.loc[idx][tag_field]
            output_folder = os.path.join(val_output_dir, str(tag))
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            shutil.copy(input_path, output_path)
    
            # Print progress (absolute and percentual)
            print(f"Processed {i+1}/{total_val_files} files ({(i+1)/total_val_files*100:.2f}%) - Found: {i + 1 - not_found_test}/{total_val_files}", end='\r')
        print(f"Processed {i+1}/{total_val_files} files ({(i+1)/total_val_files*100:.2f}%) - Found: {i + 1 - not_found_test}/{total_val_files}", end='\r')



'''
# Example usage
# Assuming base_directory and target_directory are defined, and you have train_index and test_index lists

base_directory = 'data/imgs'
target_directory = 'data/sample/1'
dataset = df_images
train_indexes = list(train_index)
test_indexes = list(test_index)

file_path_field = 'file_path'
tag_field = 'tag'
copy_images_to_folders(base_directory, target_directory, dataset, train_indexes, test_indexes, file_path_field=file_path_field, tag_field=tag_field)
'''