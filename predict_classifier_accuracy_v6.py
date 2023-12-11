import pandas as pd
import re
import os
import glob
import winsound
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import argparse

# Model choices
MODEL_CHOICES = ['resnet50', 'vgg16', 'inceptionv3', 'mobilenetv2']

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process prediction files for a classifier.")
parser.add_argument("--root", default='D:/MSc_DS_Project/Experiment/', help=f"Root directory for the experiment. Default: D:/MSc_DS_Project/Experiment/. Current available models: {', '.join(MODEL_CHOICES)}")
parser.add_argument("--model", choices=MODEL_CHOICES, required=True, help="Name of the model to process.")
parser.add_argument("--num_orig_classes", type=int, default=1, help="Number of highest probability classes to consider from original images.")
parser.add_argument("--num_modified_classes", type=int, default=1, help="Number of highest probability classes to consider from modified images.")
parser.add_argument("--include-files", nargs='*', help="List of prediction files to include for processing.")

# Parse the arguments
args = parser.parse_args()

EXPERIMENT_ROOT = args.root
chosen_model = args.model
num_orig_classes = args.num_orig_classes
num_modified_classes = args.num_modified_classes

def show_popup(message):
    root = tk.Tk()
    tk.Label(root, text=message).pack()
    tk.Button(root, text="OK", command=root.destroy).pack()
    root.mainloop()

# record time
start_time = datetime.now()

if chosen_model not in MODEL_CHOICES:
    raise ValueError(f"{chosen_model} not found in available model choices!")

def extract_fields_from_filename(filename):
    # Extract Classifier
    classifier = filename.split('_')[0]

    # Extract Dataset
    dataset_part = filename.split('_t')[0]
    dataset = dataset_part.split('_')[1:]

    # Join parts for the dataset name
    dataset = '_'.join(dataset)

    # Extract Distortion
    distortion = re.search(r'_t(\d+)_', filename)
    if distortion:
        distortion = float(distortion.group(1)) / 10
    else:
        distortion = None

    # Extract Epsilon
    epsilon = re.search(r'_e(\d+)_', filename)
    if epsilon:
        epsilon = float(epsilon.group(1)) / 100
    else:
        epsilon = None

    # Extract Date - Updated regex pattern to match yymmdd format
    date = re.search(r'_images_(\d{6})_prediction', filename)
    if date:
        date = date.group(1)
    else:
        date = None

    return classifier, dataset, distortion, epsilon, date

# Function to extract the highest probability class for each .jpg file
def extract_highest_prob(row):
    if row['ImageName'].endswith('.jpg'):
        return row['Class']
    return None

def process_prediction_file(input_file_path, num_orig_classes, num_modified_classes):
    # Read the predictions from the CSV file
    df = pd.read_csv(input_file_path)

    # Create a list to store the final results
    result_data = []

    # Loop through the .jpg files
    for _, jpg_row in df[df['ImageName'].str.endswith('.jpg')].drop_duplicates(subset='ImageName').iterrows():
        image_name = jpg_row['ImageName']
        
        # get the top num_orig_classes for the original image
        orig_rows = df[df['ImageName'] == image_name]
        orig_top_classes = orig_rows.sort_values('Probability', ascending=False).iloc[:num_orig_classes]['Class'].tolist()
        
        result_dict = {
            'ImageName': image_name,
            'Top_Orig_Classes': orig_top_classes,
        }
        
        # Checks for corresponding rows and appends the results
        for suffix, column in [('_seam.png', 'Has_Same_Class_In_Seams'), 
                               ('_only_fgsm.png', 'Has_Same_Class_In_Only_FGSM'), 
                               ('_seam_fgsm.png', 'Has_Same_Class_In_Seam_FGSM')]:
            modified_name = image_name.replace('.jpg', suffix)
            modified_rows = df[df['ImageName'] == modified_name]
            modified_top_classes = modified_rows.sort_values('Probability', ascending=False).iloc[:num_modified_classes]['Class'].tolist()
            
            # Check if any of the orig_top_classes are found in the modified_top_classes
            is_class_found = 1 if any(cls in modified_top_classes for cls in orig_top_classes) else 0
            result_dict[column] = is_class_found

        result_data.append(result_dict)
    
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(result_data)

    # Generate the output file path
    output_filename = os.path.basename(input_file_path).replace('prediction', 'accuracy')
    output_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(output_dir, output_filename)

    result_df.to_csv(output_file, index=False)
    print("Results saved to CSV file:", output_file)
    
    return result_df

def filter_latest_files(summary_data):
    """
    Filters the summary data to only include the latest entry for rows 
    where the 'Classifier', 'Dataset', 'Seam_Distortion', and 'FGSM_Epsilon' are the same, 
    but the 'Date' is different.
    """
    # Sort the data by Date in descending order, treating None as the oldest date
    sorted_data = sorted(summary_data, key=lambda x: (x['Date'] is None, x['Date']), reverse=True)
    
    seen_records = set()
    filtered_data = []
    
    for item in sorted_data:
        # Create a unique identifier for each row without the date
        identifier = (item['Classifier'], item['Dataset'], item['Seam_Distortion'], item['FGSM_Epsilon'])
        
        if identifier not in seen_records:
            seen_records.add(identifier)
            filtered_data.append(item)

    return filtered_data

# 获取所有待处理文件的列表及其数量
all_prediction_files = list(glob.glob(os.path.join(EXPERIMENT_ROOT, chosen_model, '*_prediction.csv')))

include_files = args.include_files
if include_files:
    include_files = [os.path.join(EXPERIMENT_ROOT, chosen_model, f) for f in include_files]
    all_prediction_files = [f for f in all_prediction_files if f in include_files]

total_files = len(all_prediction_files)

# Loop through model choices
summary_data = []

# Inside the main loop where you process files:
for idx, prediction_file in enumerate(all_prediction_files, start=1):
    print(f"Processing file {idx} of {total_files}")

    try:
    
        result_df = process_prediction_file(prediction_file, num_orig_classes, num_modified_classes)

        # Check if the result DataFrame is empty
        if result_df.empty:
            print(f"{prediction_file} is empty. Skipping...")
            continue  # Skip the rest of the loop and move to the next file

       # 在这里得到ImageName的重复次数
        repetition_count = len(result_df) / result_df['ImageName'].nunique()

        # 对每个ImageName进行分组，并找到每列的最大值
        grouped = result_df.groupby('ImageName').max()

        # 计算这三列的均值，并乘以ImageName的重复次数
        mean_seams = grouped['Has_Same_Class_In_Seams'].mean() * repetition_count
        mean_only_fgsm = grouped['Has_Same_Class_In_Only_FGSM'].mean() * repetition_count
        mean_seam_fgsm = grouped['Has_Same_Class_In_Seam_FGSM'].mean() * repetition_count

        classifier, dataset, distortion, epsilon, date = extract_fields_from_filename(os.path.basename(prediction_file))

        summary_data.append({
            'Classifier': classifier,
            'Dataset': dataset,
            'Seam_Distortion': distortion,
            'FGSM_Epsilon': epsilon,
            'Date': date,
            'Mean_Has_Same_Class_In_Seams': mean_seams,
            'Mean_Has_Same_Class_In_Only_FGSM': mean_only_fgsm,
            'Mean_Has_Same_Class_In_Seam_FGSM': mean_seam_fgsm
        })
    except Exception as e:
        print(f"Error processing {prediction_file}: {e}")


filtered_summary_data = filter_latest_files(summary_data)

## Save the filtered summary DataFrame
current_date = datetime.now().strftime('%Y%m%d')
summary_df = pd.DataFrame(filtered_summary_data)  # 这里使用filtered_summary_data

output_path = os.path.join(EXPERIMENT_ROOT, 'acc_outcome', f'{chosen_model}_mean_accuracy_{current_date}_orig{num_orig_classes}_mod{num_modified_classes}.csv')

os.makedirs(os.path.join(EXPERIMENT_ROOT, 'acc_outcome'), exist_ok=True)
summary_df.to_csv(output_path, index=False)
print(f"Mean results saved to CSV file:", output_path)

# ring the bell
winsound.Beep(440, 1000)  # Frequency 440Hz, Duration 1 second

# time compute
end_time = datetime.now()
duration = end_time - start_time

'''
# show the popup message
message = (f"Script finished!\n"
           f"Model: {chosen_model}\n"
           f"Results saved at: {output_path}\n"
           f"Duration: {duration}\n"
           f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
show_popup(message)
#show_popup(message)
'''
