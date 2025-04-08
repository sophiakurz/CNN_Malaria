This project is an end-to-end implementation of an automated malaria detection system that classifies single-cell blood images as either parasitized or uninfected. Leveraging transfer learning, the project uses a pre-trained MobileNetV2 as its backbone, augmented with custom dense and dropout layers for binary classification in a medical imaging context.

Key steps in the project include:

Data Preparation:
The dataset consists of high-resolution images, each displaying a single blood cell, along with a CSV file listing image names and binary labels. The script checks for image existence, splits the data into training and validation sets using stratification, and employs data augmentation techniques (including rotations, shifts, shear, zoom, and horizontal flips) to boost the robustness and variability of the training data.

Model Architecture:
The core of the model is MobileNetV2, pre-trained on ImageNet, with its layers frozen to retain established visual features. A global average pooling layer is used to condense spatial information, followed by a dense layer with ReLU activation and dropout for added regularization. A final sigmoid-activated layer outputs the probability of a cell being parasitized.

Training and Evaluation:
The model is trained using the Adam optimizer with binary crossentropy loss. Validation performance is evaluated using accuracy and F1 score, with the model achieving an impressive validation accuracy of 0.94217. These results underscore the model’s effectiveness in distinguishing between parasitized and uninfected cells.

Test Predictions:
After training and evaluation, the model generates predictions on a separate test set, and the results are formatted for submission on Kaggle.

Overall, this project demonstrates how deep learning and transfer learning techniques can be effectively applied to medical diagnostics, offering a scalable and efficient alternative to traditional malaria screening methods.
