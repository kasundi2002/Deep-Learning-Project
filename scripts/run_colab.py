# # 1) Mount Drive and go to your project
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# %cd /content/drive/MyDrive/SKIN_CANCER_PROJECT

# # 2) Run the splitter (80/10/10)
# !python scripts/split_ham10000.py --train 0.8 --val 0.1 --test 0.1 --seed 42 --clean

# # Example: ResNet50
# !python members/run_resnet50.py \
#   --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" \
#   --epochs 20 --warmup 3 --unfreeze 10 --batch 32 \
#   --out_dir runs --run_name m2_try1

# # pick one at a time; adjust --batch if you hit OOM
# !python members/run_effnet_b0.py      --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" --epochs 20 --unfreeze 20 --batch 32 --run_name m1_try1
# !python members/run_resnet50.py       --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" --epochs 20 --batch 32 --run_name m2_try1
# !python members/run_mobilenet_v2.py   --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" --epochs 20 --batch 32 --run_name m3_try1
# !python members/run_densenet121.py    --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" --epochs 20 --batch 32 --run_name m4_try1
# !python members/run_custom_cnn.py     --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" --epochs 20 --batch 32 --run_name custom_try1
