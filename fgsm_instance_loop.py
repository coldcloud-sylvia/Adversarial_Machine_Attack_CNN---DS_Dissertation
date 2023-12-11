import os
import numpy as np
import re
import winsound
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

# 定义模型列表，便于后续修改
MODEL_CHOICES = {
    'resnet': ResNet50(weights='imagenet'),
    'vgg16': VGG16(weights='imagenet'),
    # 添加其他模型...
}

# 选择模型
chosen_model = 'vgg16'
model = MODEL_CHOICES[chosen_model]

def extract_epsilon_from_path(folder_path):
    """
    从文件夹路径中提取epsilon值。
    """
    match = re.search(r'_e(\d{3})_', folder_path)
    if not match:
        raise ValueError(f"Cannot find epsilon value in the path: {folder_path}")

    epsilon_str = match.group(1)
    epsilon_value = float(epsilon_str[:1] + '.' + epsilon_str[1:])
    return epsilon_value

def fgsm_attack(model, image_tensor, epsilon):
    image_variable = tf.Variable(image_tensor)

    with tf.GradientTape() as tape:
        tape.watch(image_variable)
        prediction = model(image_variable)

    gradient = tape.gradient(prediction, image_variable)
    perturbation = epsilon * tf.sign(gradient)
    perturbed_image = image_variable + perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 255)

    return perturbed_image.numpy()

# Function to play a beep sound
def play_beep():
    winsound.Beep(1000, 200)  # Beep at 1000 Hz for 200 milliseconds

def process_fgsm_for_folder(folder_path, epsilon):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '_seam.png'))]

    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        perturbed_x = fgsm_attack(model, x, epsilon)

        if image_file.lower().endswith('.jpg'):
            perturbed_img_name = image_file[:-4] + "_only_fgsm.png"
        elif image_file.lower().endswith('_seam.png'):
            perturbed_img_name = image_file[:-9] + "_seam_fgsm.png"
        else:
            continue  # 如果有其他格式的图片，则跳过

        perturbed_img_path = os.path.join(folder_path, perturbed_img_name)
        perturbed_img = image.array_to_img(perturbed_x[0])
        perturbed_img.save(perturbed_img_path)

        original_preds = model.predict(x)
        perturbed_preds = model.predict(perturbed_x)

        original_class = decode_predictions(original_preds, top=1)[0][0][1]
        perturbed_class = decode_predictions(perturbed_preds, top=1)[0][0][1]

        print(f"Original Image ({image_file}) prediction: {original_class}")
        print(f"Perturbed Image prediction: {perturbed_class}\n")

    play_beep()

if __name__ == '__main__':
    folder_paths = [
        # r'D:/MSc_DS_Project/Experiment/data/goldfish/fish_test_t03_e001_vgg/images',
        # r'D:/MSc_DS_Project/Experiment/data/goldfish/fish_test_t03_e005_vgg/images',
        # r'D:/MSc_DS_Project/Experiment/data/goldfish/fish_test_t03_e010_vgg/images'
        r'D:/MSc_DS_Project/Experiment/data/goldfish/test_t01_e0001_resnet50'
    ]

    for folder_path in folder_paths:
        epsilon = extract_epsilon_from_path(folder_path)  # 提取epsilon值
        print(f"For the folder '{folder_path}', the epsilon value is: {epsilon}")
        process_fgsm_for_folder(folder_path, epsilon)
