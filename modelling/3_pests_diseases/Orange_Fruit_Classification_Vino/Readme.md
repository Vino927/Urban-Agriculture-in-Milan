
# Classification of Orange Fruit Diseases

This project focuses on the classification of orange fruit diseases using deep learning techniques. Three pretrained models—ResNet-18, EfficientNet B1, and EfficientNet B7—were used and evaluated, with and without image augmentation, to assess the impact on model performance.

## Table of Contents
- [Overview of Image Classification](#overview-of-image-classification)
- [Dataset Details](#dataset-details)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Conclusion](#conclusion)


## Overview of Image Classification

Image classification is a fundamental task in computer vision, where the goal is to assign a label to an image from a fixed set of categories. It typically involves:
1. **Preprocessing**: Images are resized, normalized, and possibly augmented (rotations, flips, etc.) to ensure consistency and variety in the data fed into the model.
2. **Model Selection**: Models such as convolutional neural networks (CNNs) are used for learning the features of the images. Pretrained models, like ResNet or EfficientNet, are often employed for transfer learning to leverage pre-learned features from large datasets (e.g., ImageNet).
3. **Training and Evaluation**: The model is trained on a portion of the data, and its performance is evaluated using validation and test datasets. Common metrics include accuracy, precision, recall, and F1 score.
4. **Post-processing**: Techniques such as confusion matrices help visualize misclassifications, and further refinements, like hyperparameter tuning or data augmentation, are applied to improve performance.

## Dataset Details

The dataset consists of 1,429 images divided into 8 classes of orange fruit diseases. The dataset was split into training (80%), validation (10%), and test (10%) sets.

### Number of Classes in 'Train' Split:
- **pest_psyllid**: 19 images
- **fungus_penicillium**: 34 images
- **bacteria_citrus**: 23 images
- **Canker**: 223 images
- **Black spot**: 180 images
- **Greening**: 308 images
- **healthy**: 328 images
- **Scab**: 12 images

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

Summary statistics for pest_psyllid:
            Height        Width
count    19.000000    19.000000
mean    900.842105  1218.315789
std     700.213084  1097.097132
min     168.000000   225.000000
25%     364.500000   422.000000
50%     750.000000   975.000000
75%    1241.500000  1581.500000
max    2412.000000  4288.000000

Summary statistics for fungus_penicillium:
            Height        Width
count    34.000000    34.000000
mean    537.941176   652.617647
std     701.496579   610.625661
min     167.000000   194.000000
25%     185.500000   270.000000
50%     247.000000   295.500000
75%     549.750000   800.000000
max    4032.000000  3024.000000

Summary statistics for bacteria _citrus:
           Height       Width
count   23.000000   23.000000
mean   193.565217  265.391304
std     28.227688   36.864296
min    148.000000  195.000000
25%    183.000000  259.000000
50%    191.000000  264.000000
75%    194.000000  275.000000
max    259.000000  341.000000

Summary statistics for Canker:
            Height        Width
count   223.000000   223.000000
mean    675.726457   684.932735
std     254.847988   285.845151
min     256.000000   256.000000
25%     669.500000   690.000000
50%     800.000000   800.000000
75%     800.000000   800.000000
max    1417.000000  1890.000000

Summary statistics for Black spot:
            Height        Width
count   180.000000   180.000000
mean    785.922222   791.683333
std     196.598035   226.408823
min     256.000000   256.000000
25%     800.000000   800.000000
50%     800.000000   800.000000
75%     800.000000   800.000000
max    2399.000000  2699.000000

Summary statistics for Greening:
            Height        Width
count   308.000000   308.000000
mean    788.698052   789.970779
std     102.434422   105.309173
min     256.000000   256.000000
25%     800.000000   800.000000
50%     800.000000   800.000000
75%     800.000000   800.000000
max    1417.000000  1520.000000

Summary statistics for healthy:
           Height       Width
count  328.000000  328.000000
mean   316.847561  360.853659
std     79.150698  111.268705
min    138.000000  144.000000
25%    256.000000  278.500000
50%    326.000000  352.000000
75%    374.000000  426.000000
max    478.000000  750.000000

Summary statistics for Scab:
            Height        Width
count    12.000000    12.000000
mean    802.333333   879.166667
std     967.256226  1101.704946
min     110.000000   112.000000
25%     256.000000   256.000000
50%     256.000000   256.000000
75%     996.500000  1086.500000
max    2399.000000  2699.000000


### RGB
The code found variations in the RGB color distribution across images, with differences in the intensity and frequency of pixel values for each color channel. We normalized the images to bring the pixel values to a consistent scale, reducing the impact of these color variations and ensuring the model processes the images uniformly during training


### Blurriness

Number of blurry images: 438

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

## Conclusion

	•	ResNet-18 consistently underperformed relative to the EfficientNet models, even after applying augmentation, which only slightly improved its performance.
	•	EfficientNet B1 and EfficientNet-B7 both showed much stronger performance across all datasets. , With EfficientNet B1,  while without augmentation provides slightly better precision and F1 score during testing, with augmentation better generalization with a higher testing accuracy (86.50% vs. 84.66%). Also, while precision (0.7396) and F1 score (0.7070) were slightly lower on the test set compared to the non-augmented model, the higher validation accuracy (94.96 vs. 94.24) and F1 score (0.8302 vs. 0.8252) suggest the augmented model is more robust overall.


EfficientNet (with augmentation) provided the most balanced performance overall, with solid generalization to the test data, making it the most suitable model for orange fruit disease classification. Based on the Confusion Matrix, the model generally performed well for most diseases like "Canker", "Greening", and "Healthy”. However, there was confusion in predicting "Scab", which is often misclassified “Bacteria_Citrus”. This suggests that the model struggles to distinguish between visually similar diseases, highlighting an area for further refinement.





