The author of the paper and the code is Hanzhi Li(1226860769@qq.com). I sincerely thank Professor Jia Guo from Columbia University (jg3400@columbia.edu) for his guidance on this research.

This code is associated with the paper "Vision Mamba for Accurate and Efficient Alzheimer’s Disease Classification via Brain MRI". The following content is the abstract of this article. 

Abstract—Alzheimer’s disease represents a prevalent neurodegenerative disorder, where early diagnosis is crucial for timely intervention. However, manual analysis of brain MRI scans is time-consuming and subjective, highlighting the necessity of automated methods to improve diagnostic accuracy and efficiency. The innovation of this study lies in comprehensively contrasting the novel Vision Mamba model with other established deep learning models, demonstrating its distinct advantages in classifying Alzheimer’s disease stages via brain MRI images. Specifically, the comparison includes six other deep learning architectures: four conventional convolutional neural networks (CNNs) - ResNet, VGG, InceptionNet, DenseNet, and two transformer models - Vision Transformer and Swin Transformer. The employed brain MRI dataset consists of 4189 training images, 1150 validation images, and 1267 test images, categorized into three classes: ‘NonDemented’, ‘MildDemented’, and ‘VeryMildDemented’. Every model was put through testing to see its accuracy, F1 score, parameter count, and the number of floating-point operations (FLOPs) utilized during training. Vision Mamba was found to give the best outcomes, reaching a test accuracy of 0.9013 and an F1 score of 0.8861, while using significantly fewer parameters and FLOPs, showing better performance than both the CNNs and Transformer networks. Consequently, Vision Mamba represents a significant advancement in enhancing the diagnostic accuracy and efficiency of Alzheimer's disease and demonstrates its potential as a valuable tool in cutting-edge medical diagnostics.

The dataset for this research is stored in the folder "NewThree_Dataset". The introduction of the dataset can be found in the docx file. However, due to the insufficient number of samples in the "Moderate Demented" category, only the samples from the other three categories were used for the study. 

The ipynb files of the seven models have been attached to the repository. It should be particularly noted that Vision Mamba requires the terminal for model training. The terminal input and operation records are all placed in the "记录.ipynb" file. You can also use the following sample code to train and evaluate Vision Mamba: 

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=6666 --nproc_per_node=1 main.py \
--model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 32 --drop-path 0.05 --weight-decay 0.05 --lr 5e-4 --num_workers 1 \
--data-path ./dataset --output_dir ./output --no_amp



















