# Image Segmentation Project

## Aim
The primary objective of this project is to perform image segmentation using a U-Net architecture. Image segmentation involves dividing an image into segments or regions to simplify the representation of an image. The goal is to accurately identify and delineate specific objects within the images.

## Approach
- **Data Loading and Splitting**
  - Load and split the dataset into unmasked and masked images.
  - Preprocess the data to prepare it for training.

- **U-Net Architecture Implementation**
  - **Model Details**
    - Develop a U-Net architecture for image segmentation.
  - **Encoder (Downsampling Block)**
    - Implement the downsampling block to extract features from the input images.
    - Exercise 1: Implement the conv_block function.
  - **Decoder (Upsampling Block)**
    - Implement the upsampling block to reconstruct the segmented image.
    - Exercise 2: Implement the upsampling_block function.
  - **Build the Model**
    - Exercise 3: Create the complete U-Net model.
  - **Set Model Dimensions**
    - Define the input and output dimensions of the model.
  - **Loss Function**
    - Define the loss function for training the U-Net.
  - **Dataset Handling**
    - Implement data handling procedures for training the U-Net.

- **Model Training**
  - **Create Predicted Masks**
    - Train the model and create predicted masks for evaluation.
  - **Plot Model Accuracy**
    - Visualize the accuracy of the model during training.
  - **Show Predictions**
    - Display predictions made by the trained model.
# Implementation

For the implementation of the image segmentation project, the following libraries and tools were utilized:

- **Python:** The core programming language for implementing the image segmentation solution.

- **TensorFlow and Keras:** Leveraged the powerful combination of TensorFlow as the deep learning framework and Keras as the high-level neural networks API. These libraries facilitated the seamless implementation of the U-Net architecture for image segmentation.

- **Pandas and NumPy:** Utilized Pandas for data manipulation and NumPy for numerical operations, providing efficient handling and processing of dataset-related tasks.

- **Scikit-learn:** Employed Scikit-learn for various machine learning utilities, including metrics for model evaluation and preprocessing tools.

- **Matplotlib:** Used Matplotlib for creating visualizations, allowing for the exploration and analysis of the dataset, model training progress, and results.

- **CARLA Self-Driving Car Dataset:** The dataset used for training and evaluating the image segmentation model was sourced from the CARLA simulator. This dataset provides a diverse set of images for autonomous driving scenarios, contributing to the robustness and real-world applicability of the trained model.

## Results
The trained U-Net model demonstrates effective image segmentation, accurately delineating objects within the images. The project provides a comprehensive understanding of U-Net architecture and its application in image segmentation tasks.  

## License
This project is licensed under the [MIT License](LICENSE).

## Disclaimer

I recognize the time people spend on building intuition, understanding new concepts and debugging assignments. The solutions uploaded here are **only for reference**. They are meant to unblock you if you get stuck somewhere. Please do not copy any part of the code as-is (the programming assignments are fairly easy if you read the instructions carefully). Similarly, try out the quizzes yourself before you refer to the quiz solutions. This course is the most straight-forward deep learning course I have ever taken, with fabulous course content and structure. It's a treasure by the deeplearning.ai team.
