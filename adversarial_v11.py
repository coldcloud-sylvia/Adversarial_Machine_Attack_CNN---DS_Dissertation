import logging
import os
import shutil
import re
import datetime
import time
import csv
#import winsound
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess, decode_predictions as vgg_decode
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess, decode_predictions as inception_decode
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenetv2_preprocess, decode_predictions as mobilenetv2_decode


# example for running the code
# python adversarial_v3.py --root_dir "D:\MSc_DS_Project\Experiment" --data_choice "fish_test" --threshold 0.30 --model "vgg16" --epsilons 0.01 0.05 0.10
# ********************* Configurable Parameters - Start *********************
DEFAULT_PROJECT_ROOT_DIR = '/content/drive/MyDrive/ds_aml_sylvia/'
DEFAULT_DATA_CHOICE = 'SulphurButterfly'  # 'bird', 'cat', etc.
DEFAULT_THRESHOLD_VALUE = 0.10
DEFAULT_MODEL_KEY = 'vgg16'
DEFAULT_EPSILONS = [0.01, 0.05, 0.10]
# ********************* Configurable Parameters - End *********************

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process FGSM attack and predictions.')
    parser.add_argument('--root_dir', type=str, default=DEFAULT_PROJECT_ROOT_DIR, help='Root directory of the project.')
    parser.add_argument('--data_choice', type=str, default=DEFAULT_DATA_CHOICE, help='Choice of data like bird, cat etc.')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD_VALUE, help='Threshold value.')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_KEY, help='Model choice, e.g., vgg16, resnet50, inceptionv3, mobilenetv2.')
    parser.add_argument('--epsilons', type=float, nargs='*', default=DEFAULT_EPSILONS, help='List of epsilon values e.g. --epsilons 0.01 0.05 0.1')
    parser.add_argument('--top_n', type=int, default=5, help='Top N predictions to consider.')
    parser.add_argument("--delete-folders", action="store_true",
                        help="If specified, will delete the processed folders after completion.")
    parser.add_argument("--run-fgsm", action="store_true",
                    help="If specified, will run the FGSM attack part of the code.")

    return parser.parse_args()


args = parse_arguments()
PROJECT_ROOT_DIR = args.root_dir
DATA_CHOICE = args.data_choice
threshold_value = args.threshold
chosen_model_key = args.model
epsilons = args.epsilons   # Update epsilons based on parsed arguments
threshold_tag = "_t" + "{:02}".format(int(threshold_value * 10))
top_n = args.top_n  # 新增加的top_n变量

MODEL_CHOICES = {
    'resnet50': {
        'model': ResNet50(weights='imagenet'),
        'preprocess': resnet_preprocess,
        'decode': resnet_decode
    },
    'vgg16': {
        'model': VGG16(weights='imagenet'),
        'preprocess': vgg_preprocess,
        'decode': vgg_decode
    },
    'inceptionv3': {
        'model': InceptionV3(weights='imagenet'),
        'preprocess': inception_preprocess,
        'decode': inception_decode
    },
    'mobilenetv2': {
        'model': MobileNetV2(weights='imagenet'),
        'preprocess': mobilenetv2_preprocess,
        'decode': mobilenetv2_decode
    }
}

MODEL_INPUT_SIZES = {
    'resnet50': (224, 224),
    'vgg16': (224, 224),
    'inceptionv3': (299, 299),
    'mobilenetv2': (224, 224)
}


chosen_model = MODEL_CHOICES[chosen_model_key]['model']
chosen_preprocess = MODEL_CHOICES[chosen_model_key]['preprocess']
chosen_decode = MODEL_CHOICES[chosen_model_key]['decode']

def extract_epsilon_from_path(folder_path):
    match = re.search(r'_e(\d{3})_', folder_path)
    if not match:
        raise ValueError(f"Cannot find epsilon value in the path: {folder_path}")
    epsilon_str = match.group(1)
    epsilon_value = float(epsilon_str[:1] + '.' + epsilon_str[1:])
    return epsilon_value

def epsilon_to_folder_suffix(epsilon):
    """
    Converts an epsilon value to the corresponding folder suffix.
    For example, 0.03 would become "_e003_".
    """
    epsilon_int = int(epsilon * 100)
    return f"_e{epsilon_int:03}_"

def fgsm_attack(model, image_tensor, epsilon):
    image_variable = tf.Variable(image_tensor)
    with tf.GradientTape() as tape:
        tape.watch(image_variable)
        prediction = model(image_variable)
    gradient = tape.gradient(prediction, image_variable)
    perturbation = epsilon * tf.sign(gradient)
    perturbed_image = image_variable + perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 255)
    
    # Convert to NumPy array and ensure it is of uint8 type
    perturbed_image_np = perturbed_image.numpy().astype(np.uint8)
    
    return perturbed_image_np


def play_error_beep():
    winsound.Beep(500, 1500)

def play_success_beep():
    winsound.Beep(1000, 500)

def process_fgsm_for_folder(folder_path):

    target_size = MODEL_INPUT_SIZES[chosen_model_key]  # 获取模型的适当输入尺寸
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.lower().endswith('_energy.png')]
    epsilon = extract_epsilon_from_path(folder_path)
    
    for idx, image_file in enumerate(image_files, start=1):
        img_path = os.path.join(folder_path, image_file)
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = chosen_preprocess(x)  # 确保只预处理一次

        perturbed_x = fgsm_attack(chosen_model, x, epsilon)
        
        # 根据文件的扩展名来决定保存的文件名
        if image_file.lower().endswith('.jpg'):
            perturbed_img_name = image_file[:-4] + "_only_fgsm.png"
        elif image_file.lower().endswith('.png'):
            perturbed_img_name = image_file[:-4] + "_only_fgsm.png"
        perturbed_img_path = os.path.join(folder_path, perturbed_img_name)
        
        perturbed_img = image.array_to_img(perturbed_x[0])
        perturbed_img.save(perturbed_img_path)
        
        original_preds = chosen_model.predict(x)
        perturbed_preds = chosen_model.predict(perturbed_x)
        original_class = chosen_decode(original_preds, top=top_n)[0][0][1]
        perturbed_class = chosen_decode(perturbed_preds, top=top_n)[0][0][1]
        
        print(f"Function: process_fgsm_for_folder | Folder: {folder_path} | Image: {idx}/{len(image_files)} | Original Class: {original_class} | Perturbed Class: {perturbed_class}")


def model_predict_on_folder(image_folder, top_n):
    all_results = []
    target_size = MODEL_INPUT_SIZES[chosen_model_key]  # 获取模型的适当输入尺寸
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.lower().endswith('_energy.png')]
    total_images = len(image_files)
    for idx, image_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, image_file)
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = chosen_preprocess(x)  # 确保只预处理一次
        
        preds = chosen_model.predict(x)
        top_predictions = chosen_decode(preds, top=top_n)[0]  # use the parameter top_n
        
        image_predictions = []
        for pred in top_predictions:
            class_id, class_label, probability = pred
            probability = float(probability)  
            image_predictions.append({'Class': class_label, 'Probability': probability})
        all_results.append({'ImageName': image_file, 'Predictions': image_predictions})
    return all_results

def compare_probabilities(file_path, target_class):
    # 读取数据
    file_df = pd.read_csv(file_path)
    
    # 筛选目标类别
    file_df = file_df[file_df['Class'] == target_class]

    # 筛选原始图片和不同扰动类型的图片
    original_df = file_df[file_df['ImageName'].str.endswith('.jpg')]
    rand_df = file_df[file_df['ImageName'].str.endswith('_rand.png')]
    seam_df = file_df[file_df['ImageName'].str.endswith('_seam.png')]
    only_fgsm_df = file_df[file_df['ImageName'].str.endswith('_only_fgsm.png')]
    seam_only_fgsm_df = file_df[file_df['ImageName'].str.endswith('_seam_only_fgsm.png')]

    # 提取basename
    original_df['Basename'] = original_df['ImageName'].str.extract(r'(.+)\.jpg')
    rand_df['Basename'] = rand_df['ImageName'].str.extract(r'(.+)_rand')
    seam_df['Basename'] = seam_df['ImageName'].str.extract(r'(.+)_seam')
    only_fgsm_df['Basename'] = only_fgsm_df['ImageName'].str.extract(r'(.+)_only_fgsm')
    seam_only_fgsm_df['Basename'] = seam_only_fgsm_df['ImageName'].str.extract(r'(.+)_seam_only_fgsm')

    # 原始图片中取概率最大的
    max_prob_original_df = original_df.loc[original_df.groupby('Basename')['Probability'].idxmax()]

    # 左连接扰动图片的概率
    result_df = max_prob_original_df
    for disturbance_df, suffix in zip([rand_df, seam_df, only_fgsm_df, seam_only_fgsm_df],
                                      ['_rand', '_seam', '_only_fgsm', '_seam_only_fgsm']):
        result_df = pd.merge(result_df, disturbance_df[['Basename', 'Probability']], on='Basename', how='left', suffixes=('', suffix))
        
    return result_df


def process_folder(folder_path, top_n):
    model_folder_path = os.path.join(PROJECT_ROOT_DIR, chosen_model_key)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    parent_folder_name = os.path.basename(os.path.dirname(folder_path))
    today = datetime.date.today().strftime('%y%m%d')
    top_n = args.top_n
    csv_file_name = f'{chosen_model_key}_{parent_folder_name}_{os.path.basename(folder_path)}_{today}_prediction.csv'
    all_results = model_predict_on_folder(folder_path, top_n)
    
    csv_file_path = os.path.join(model_folder_path, csv_file_name)
    with open(csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ['ImageName', 'Class', 'Probability']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            image_name = result['ImageName']
            predictions = result['Predictions']
            for prediction in predictions:
                class_name = prediction['Class']
                probability = prediction['Probability']
                writer.writerow({'ImageName': image_name, 'Class': class_name, 'Probability': probability})
    print(f"Function: process_folder | Folder: {folder_path}", "Predictions saved to CSV file:", csv_file_path)



if __name__ == '__main__':
    args = parse_arguments()
    PROJECT_ROOT_DIR = args.root_dir
    DATA_CHOICE = args.data_choice
    threshold_value = args.threshold
    chosen_model_key = args.model
    epsilons = args.epsilons   # Update epsilons based on parsed arguments
    threshold_tag = "_t" + "{:02}".format(int(threshold_value * 10))

 

    completed_tasks = []
    start_time = time.time()  # start time

    try:
        original_folder = os.path.join(PROJECT_ROOT_DIR, f'data/{DATA_CHOICE}/{DATA_CHOICE}{threshold_tag}/images')
        folders_to_create = [os.path.join(PROJECT_ROOT_DIR, f'data/{DATA_CHOICE}/{DATA_CHOICE}{threshold_tag}{epsilon_to_folder_suffix(epsilon)}{chosen_model_key}/images') for epsilon in epsilons]
        
        if args.run_fgsm:
            for folder in folders_to_create:
                if not os.path.exists(folder):
                    shutil.copytree(original_folder, folder)
                process_fgsm_for_folder(folder)
                completed_tasks.append(f"Completed FGSM processing for folder: {folder}")

    except Exception as e:
        print(f"Error during FGSM processing: {e}")
        #play_error_beep()

    try:
        for folder_path in folders_to_create:
            process_folder(folder_path, top_n) 
            csv_file_name = f'{chosen_model_key}_{os.path.basename(os.path.dirname(folder_path))}_{os.path.basename(folder_path)}_{datetime.date.today().strftime("%y%m%d")}_prediction.csv'
            csv_file_path = os.path.join(chosen_model_key, csv_file_name)
            comparison_csv_file_name = f'{chosen_model_key}_{os.path.basename(os.path.dirname(folder_path))}_{os.path.basename(folder_path)}_{datetime.date.today().strftime("%y%m%d")}_probability_comparison.csv'
            comparison_csv_file_path = os.path.join(chosen_model_key, comparison_csv_file_name)
            compare_probabilities(csv_file_path, DATA_CHOICE, comparison_csv_file_path)
            completed_tasks.append(f"Generated CSV: {csv_file_path}")
            completed_tasks.append(f"Generated Comparison CSV: {comparison_csv_file_path}")
            logging.warning(f"{csv_file_path}")
            logging.warning(f"{comparison_csv_file_path}")

        
        if args.delete_folders:
            for folder in folders_to_create:
                shutil.rmtree(folder)  # 删除文件夹及其所有内容
                print(f"Deleted folder: {folder}")

    except Exception as e:
        print(f"Error during model prediction: {e}")
        #play_error_beep()

    end_time = time.time()  # end time
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    #play_success_beep()
    print(f"Execution completed in {int(hours)} hours, {int(minutes)} minutes.")
    
    # print completed tasks
    print("\nCompleted tasks:")
    for task in completed_tasks:
        print(task)

    print(f"Files saved in: {chosen_model_key} directory.")

