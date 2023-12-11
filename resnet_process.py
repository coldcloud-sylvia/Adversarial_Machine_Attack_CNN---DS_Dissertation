import os
import shutil
import subprocess
import winsound

def copy_folder(src, dest):
    """
    复制文件夹。
    :param src: 源文件夹路径。
    :param dest: 目标文件夹路径。
    """
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

def play_sound(success=True):
    """
    播放声音。成功时播放长声音，失败时播放短声音。
    """
    duration = 1000 if success else 200  # 成功时的声音时长为1000毫秒，失败为200毫秒。
    winsound.Beep(1000, duration)

if __name__ == '__main__':
    try:
        # # 1. 运行 process_image_loop.py
        # result = subprocess.run(["python", "D:/MSc_DS_Project/Experiment/process/process_image_loop.py"])
        # if result.returncode != 0:
        #     play_sound(success=False)
        #     print("Error in process_image_loop.py")
        #     exit()

        # 2. 复制文件夹
        original_folder = 'D:/MSc_DS_Project/Experiment/data/goldfish/goldfish_t01/images'
        folders_to_create = [
            'D:/MSc_DS_Project/Experiment/data/goldfish/goldfish_t01_e001_resnet/images',
            'D:/MSc_DS_Project/Experiment/data/goldfish/goldfish_t01_e005_resnet/images',
            'D:/MSc_DS_Project/Experiment/data/goldfish/goldfish_t01_e010_resnet/images'
        ]

        for folder in folders_to_create:
            copy_folder(original_folder, folder)

        # 3. 运行 fgsm_instance 脚本
        result = subprocess.run(["python", "D:/MSc_DS_Project/Experiment/fgsm/fgsm_instance_loop.py"])
        if result.returncode != 0:
            play_sound(success=False)
            print("Error in fgsm_instance_loop.py")
            exit()

        # 4. 运行 resnet_loop.py
        result = subprocess.run(["python", "D:/MSc_DS_Project/Experiment/resnet/resnet_loop.py"])
        if result.returncode != 0:
            play_sound(success=False)
            print("Error in resnet_loop.py")
            exit()

        # 所有脚本都成功执行，播放成功的声音提示
        play_sound(success=True)

    except Exception as e:
        play_sound(success=False)
        print(f"An error occurred: {e}")
