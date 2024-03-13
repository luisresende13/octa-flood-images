import os
import shutil

def copy_images_to_folders(base_directory, target_directory, dataset, train_index=None, test_index=None, val_index=None, file_path_field='file_path', tag_field='tag'):
    
    if train_index is not None and len(train_index):
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

    if test_index is not None and len(test_index):
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

    if val_index is not None and len(val_index):
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
            print(f"Processed {i+1}/{total_val_files} files ({(i+1)/total_val_files*100:.2f}%) - Found: {i + 1 - not_found_val}/{total_val_files}", end='\r')
        print(f"Processed {i+1}/{total_val_files} files ({(i+1)/total_val_files*100:.2f}%) - Found: {i + 1 - not_found_val}/{total_val_files}", end='\r')



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