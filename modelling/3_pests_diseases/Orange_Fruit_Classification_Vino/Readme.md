
# Classification of Orange Fruit Diseases

This project focuses on the classification of orange fruit diseases using deep learning techniques. Three pretrained models—ResNet-18, EfficientNet B1, and EfficientNet B7—were used and evaluated, with and without image augmentation, to assess the impact on model performance.

## Table of Contents
- [Overview of Image Classification](#overview-of-image-classification)
- [Dataset Details](#dataset-details)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Prediction](#prediction)
- [Conclusion](#conclusion)


## Overview of Image Classification

Image classification is a fundamental task in computer vision, where the goal is to assign a label to an image from a fixed set of categories. It typically involves:
1. **Preprocessing**: Images are resized, normalized, and possibly augmented (rotations, flips, etc.) to ensure consistency and variety in the data fed into the model.
2. **Model Selection**: Models such as convolutional neural networks (CNNs) are used for learning the features of the images. Pretrained models, like ResNet or EfficientNet, are often employed for transfer learning to leverage pre-learned features from large datasets (e.g., ImageNet).
3. **Training and Evaluation**: The model is trained on a portion of the data, and its performance is evaluated using validation and test datasets. Common metrics include accuracy, precision, recall, and F1 score.
4. **Post-processing**: Techniques such as confusion matrices help visualize misclassifications, and further refinements, like hyperparameter tuning or data augmentation, are applied to improve performance.

## Dataset Details

The dataset consists of 1,429 images divided into 8 classes of orange fruit diseases. The dataset was split into training (80%), validation (10%), and test (10%) sets.

**Sample images:**
![image](https://github.com/user-attachments/assets/170c6835-06e0-4261-b24a-58c3f9a60b0b)

### Number of Classes in 'Train' Split:
- **pest_psyllid**: 19 images
- **fungus_penicillium**: 34 images
- **bacteria_citrus**: 23 images
- **Canker**: 223 images
- **Black spot**: 180 images
- **Greening**: 308 images
- **healthy**: 328 images
- **Scab**: 12 images
  
![classimbalance](https://github.com/user-attachments/assets/588542b3-3b4a-44d2-a99e-4161953f76cf)

### Class Imbalance
A significant class imbalance is present in the dataset. For instance, the "healthy" class has 328 images, while the "Scab" class has only 12. This imbalance may lead to biased model performance. Data augmentation, oversampling, and class weighting were considered to address this issue.

1. **Class Imbalance**:
   - There is a noticeable class imbalance in the dataset. The "healthy" class has the highest number of images across all datasets, with 328 images in the training set. In contrast, the "Scab" class has the fewest images, with only 12 images in the training set. This disparity in class representation could lead to a biased model that may favor the classes with more samples.

2. **Impact on Model Performance**:
   - The class imbalance could affect the model’s ability to accurately classify the minority classes more effectively due to their higher representation in the dataset.

3. **Need for Data Augmentation or Resampling**:
   - To mitigate the effects of class imbalance, techniques such as data augmentation for the minority classes, oversampling, or class weighting in the loss function will be considered. 

4. **Potential for Transfer Learning**:
   - Considering the limited number of images in some classes, using a pre-trained model should help improve model performance by utilizing features learned from larger, more diverse datasets.


### Image dimensions
The image dimensions across different classes show significant variations in both height and width. Therefore all images will be standardized to a consistent size such as 256x256 and also cropped. Also, since only a small percentage of images are outliers are and all images will be resized, these will be left in the data.

<img width="214" alt="Screenshot 2024-10-03 at 1 18 31 AM" src="https://github.com/user-attachments/assets/13c3907d-888c-4c5d-961f-3b9a7cf088f1">


### RGB
The code found variations in the RGB color distribution across images, with differences in the intensity and frequency of pixel values for each color channel. We normalized the images to bring the pixel values to a consistent scale, reducing the impact of these color variations and ensuring the model processes the images uniformly during training.

![rgb1](https://github.com/user-attachments/assets/579161e9-8ffa-4c81-94ff-99f699984272)

![rgb2](https://github.com/user-attachments/assets/b55d70cc-476c-436e-b6e0-343759a89013)

### Blurriness

Number of blurry images: 438

![bluiness](https://github.com/user-attachments/assets/dd529bef-2d9a-4a22-b5cd-a8160e82d280)

Given that 30% of the images are blurry in the dataset, we will use CLAHE to enhance the contrast, RandomBrightnessContrast to adjust lighting, and GaussNoise to random noise to sharpen features.




## Modeling Approach

### Preprocessing and Augmentation

Given the variability in image dimensions, all images were standardized to 256x256 pixels. Image augmentation techniques, such as horizontal and vertical flips, random rotations, brightness/contrast adjustments, and Gaussian noise, were applied to enrich the training data and reduce overfitting.

### Augmentation Techniques Applied:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance contrast.
- **RandomBrightnessContrast** to adjust lighting.
- **GaussNoise** to add noise and improve model robustness.

### Model Configuration

The following deep learning models were used:
- **ResNet-18**
- **EfficientNet B1**
- **EfficientNet B7**

Both augmented and non-augmented versions were trained to evaluate their impact. Early stopping was implemented in some cases to prevent overfitting.

### Configurations
```python
config_ef_aug = dict(
    resize = 256,
    crop = 240,
    augment = True,
    epochs = 50,
    batch_size = 32,
    lr = 0.001,
    dropout = 0.2,
    weights_path = 'weights_augmented_ef.pth'
)


A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.1),
    A.CLAHE(clip_limit=(1, 3), tile_grid_size=(8, 8), p=0.3),
    A.CenterCrop(height=config['crop'], width=config['crop']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

```
## Results

1. **ResNet-18 (Without Augmentation)**
   - **Train Accuracy**: 82.16%
   - **Validation Accuracy**: 81.05%
   - **Testing Accuracy**: 80.89%
The ResNet-18 model, without augmentation, exhibited relatively low precision and recall, indicating that the model struggled to differentiate between classes. Both the training and validation performances are quite similar, suggesting the model did not overfit. However, its F1 score indicates suboptimal performance in disease classification due to the low balance between precision and recall.

2. **ResNet-18 (With Augmentation)**
   - **Train Accuracy**: 80.20%
   - **Validation Accuracy**: 83.66%
   - **Testing Accuracy**: 78.98%
With augmentation, the performance of ResNet-18 remained largely unchanged. While there was a slight improvement in validation accuracy and F1 score, the model's overall capability in handling disease classification was still limited. This shows that data augmentation had a marginal positive effect on performance.


3. **EfficientNet B1 (Without Augmentation)**
   - **Train Accuracy**: 99.56%
   - **Validation Accuracy**: 94.24%
   - **Testing Accuracy**: 84.66%

The EfficientNet model showed near-perfect performance during training, but there was a noticeable drop in performance during validation and testing, indicating some degree of overfitting. However, it still significantly outperformed ResNet-18, especially in terms of precision, recall, and F1 score on the test set.

4. **EfficientNet B1 (With Augmentation)**
   - **Train Accuracy**: 98.67%
   - **Validation Accuracy**: 94.96%
   - **Testing Accuracy**: 86.50%

<img width="1145" alt="Training F1" src="https://github.com/user-attachments/assets/d542347b-0196-4866-a5e1-1137d219766a">

<img width="1114" alt="Training loss" src="https://github.com/user-attachments/assets/c0ef6e04-18ba-42c5-bd83-50a88726cca9">

<img width="545" alt="confusion" src="https://github.com/user-attachments/assets/660afd8b-7a4b-4d1f-bb88-16b88f715a94">

EfficientNet B1 with augmentation demonstrated a slight improvement in generalization on the validation and testing datasets. There was a reduction in the overfitting observed in the non-augmented version, and testing accuracy increased to 86.50%. Precision, recall, and F1 score remained strong, confirming EfficientNet’s effectiveness in the classification task.


5. **EfficientNet B7 (Without Augmentation)**
   - **Train Accuracy**: 99.02%
   - **Validation Accuracy**: 93.53%
   - **Testing Accuracy**: 83.44%

WfficientNet-B7, like EfficientNet B1, showed strong performance during training but a noticeable drop during validation and testing, although it still outperformed ResNet-18. The validation and testing F1 scores suggest that while the model generalizes well, it still has room for improvement.

6. **EfficientNet B7 (With Augmentation)**
   - **Train Accuracy**: 93.88%
   - **Validation Accuracy**: 89.21%
   - **Testing Accuracy**: 82.82%

After applying augmentation, EfficientNet-B7’s validation and testing accuracy slightly decreased, but the F1 score remained consistent. This suggests that the model still has good generalization ability, but augmentation might not have had as significant an impact as with other models.

## Prediction using EfficientNet B1:
<img width="571" alt="scab" src="https://github.com/user-attachments/assets/6b272050-f31e-4830-99e8-5d3f92f4d96e">


## Conclusion

- ResNet-18 consistently underperformed relative to the EfficientNet models, even after applying augmentation, which only slightly improved its performance.
- EfficientNet B1 and EfficientNet-B7 both showed much stronger performance across all datasets. With EfficientNet B1, while "without augmentation" provides slightly better precision and F1 score during testing, "with augmentation" gives a better generalization with a higher testing accuracy (86.50% vs. 84.66%). Also, while precision (0.7396) and F1 score (0.7070) were slightly lower on the test set compared to the non-augmented model, the higher validation accuracy (94.96 vs. 94.24) and F1 score (0.8302 vs. 0.8252) suggest the augmented model is more robust overall.

**EfficientNet (with augmentation)** provided the most balanced performance overall, with solid generalization to the test data, making it the most suitable model for orange fruit disease classification. Based on the Confusion Matrix, the model generally performed well for most diseases like "Canker", "Greening", and "Healthy”. However, there was confusion in predicting "Scab", which is often misclassified “Bacteria_Citrus”. This suggests that the model struggles to distinguish between visually similar diseases, highlighting an area for further refinement.





