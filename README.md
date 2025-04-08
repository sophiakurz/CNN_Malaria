This project is an end-to-end implementation of an automated malaria detection system that classifies single-cell blood images as parasitized or uninfected. It harnesses the power of transfer learning by employing a pre-trained MobileNetV2 as its backbone and builds upon it with custom dense and dropout layers to tailor the model specifically for binary classification in the medical imaging context.

Key steps in the project include:

Data Preparation:
The dataset comprises high-resolution images, each containing a single blood cell, along with a CSV file that provides image names and binary labels. The code verifies image existence, splits the dataset into training and validation subsets using stratification, and applies data augmentation (rotations, shifts, shear, zoom, and horizontal flips) to increase training data variability and robustness.

Model Architecture:
The core of the model is MobileNetV2, pre-trained on ImageNet, with its layers frozen to leverage pre-learned visual features. A global average pooling layer condenses the spatial information, followed by a fully connected layer with ReLU activation and dropout for regularization. The final sigmoid-activated layer outputs the probability of a cell being parasitized.

Training and Evaluation:
The model is trained on the augmented training dataset using the Adam optimizer and binary crossentropy loss. Performance is validated using both accuracy and the F1 score, ensuring a balanced evaluation of the model's precision and recall. A separate test set is processed similarly to generate predictions, which are then formatted for submission to Kaggle.

Overall, the project demonstrates how deep learning and transfer learning techniques can be effectively applied to medical diagnostics, offering a scalable and efficient alternative to traditional malaria screening methods.
