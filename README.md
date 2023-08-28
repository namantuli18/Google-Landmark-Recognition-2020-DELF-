# Google-Landmark-Recognition-2020-DELF-

## Results
| Public Score | Private Score | Public Rank | Private Rank |
|----------|----------|----------|----------|
| 0.5147 | 0.4793 | 52  | 60

## Problem Statement  
The competitors in this challenge were asked to perform image classification on a dataset with roughly ~80K classes, with each class only having slightly adequate training samples.

## Data
The image data for the competition was divided into 2 separate folders, i.e., **train** and **test**, which consisted of the training and testing sets of images, respectively. The **train.csv** file maps each of the images in the train folder to a landmark_id, which is used as the label to classify the images.

### Sample Image:

### Resources to the dataset:  
 Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval, T. Weyand, A. Araujo, B. Cao and J. Sim, Proc. CVPR'20 [https://arxiv.org/abs/2004.01804]

## Evaluation Metric  
Submissions are evaluated using **Global Average Precision (GAP)** at 𝑘, where 𝑘=1

## Model Architecture Used  
For extracting the features from testing images on a global and local level, we have used Google's [DELF](https://github.com/tensorflow/models/blob/master/research/delf/README.md) TensorFlow Saved Models.

## Submission Methodology  
1. The pre-trained **DELF** model is used for extracting features from each image.
2. For each test image, we have ranked training images similar to that based on the embedding similarity.
3. A parameter `NUM_TO_RERANK` is used to geometrically verify and re-rank the most similar images.
4. Score aggregation is done by summing up the scores of these `NUM_TO_RERANK` images.
5. The class with the highest aggregated score is chosen as the predicted class.

## Hyperparameter Tuning  
```python
    NUM_EMBEDDING_DIMENSIONS = 2048 # Used to set the dimensions while feature extraction
    Retrieval & re-ranking parameters:
    NUM_TO_RERANK = 3
    TOP_K = 3 # Number of retrieved images used to make prediction for a test image.
    
    #RANSAC parameters:
    MAX_INLIER_SCORE = 60
    MAX_REPROJECTION_ERROR = 16.5
    MAX_RANSAC_ITERATIONS = 10000
    HOMOGRAPHY_CONFIDENCE = 0.999
```

## Challenges Encountered
1. The existence of 80K classes in the dataset makes it tough to train an image classification model, majorly because of limitations in compute and excessive training time.
2. Since for each image in the test set, the feature similarity and re-ranking process is used, it makes it tough for the execution to be completed within 12 hours total.
3. With increased time complexity, it becomess overly difficult to ensemble results from  
    - Different models
    - Same model with tuned hyperparams(RANSAC parameters/Re-ranking)

## Submission Code
The detailed python notebook for execution could be found at path
