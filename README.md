# 🧬 Breast Histopathology Image Classification

This project is focused on building a Convolutional Neural Network (CNN) to classify breast cancer histopathology images—specifically identifying **Invasive Ductal Carcinoma (IDC)** from non-cancerous tissue. It uses high-resolution image data from the [Kaggle IDC Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

---

##  Dataset

- **Source**: Kaggle - [paultimothymooney/breast-histopathology-images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- **Classes**: 
  - `0`: Non-IDC (non-cancerous)
  - `1`: IDC (cancerous)

Images are 50x50 pixel patches extracted from whole-slide images.

---

##  Project Workflow

### 1.  Dataset Download
- Automated using `kagglehub`, avoiding manual downloads.
- Download path handled dynamically using:
  
  ```python
  kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")

  ````
 ### 2.  Data Preprocessing
 
 - Dataset paths shuffled and split:

    - Train: 80%
  
    - Validation: 10%
  
    - Test: 10%

  - Directory structure created using os, shutil, and imutils.paths.

  - Files moved into respective training, validation, and testing folders.

###  3.  Model Building

- CNN built using Keras (TensorFlow backend).

- Architecture includes:

    - Convolutional Layers
    
    - MaxPooling
    
    - Dropout
    
    - Dense Layers with Softmax output
    
    - Binary classification with final sigmoid or softmax (depending on architecture).

###  4.  Model Training
  - Training done using model.fit() with:

    - Early stopping

    - Validation monitoring

  Data generators for image batches

###  5.  Evaluation
  - Accuracy and loss plotted using matplotlib.

  - Model tested on the test set with classification metrics:

    - Accuracy
  
    - Confusion Matrix

    - Classification Report
   
----

  ###   Dependencies
  - Install all required libraries with:

  ```bash

    pip install numpy opencv-python pillow tensorflow keras imutils scikit-learn matplotlib kagglehub

  ```

  ###   Getting Started
  
  1. Clone this repository.
  
  2. Run the Jupyter Notebook.
  
  3. The dataset will be downloaded and processed.
  
  4. Train and evaluate the model.

  ###  Results
  -   After training.
 

  Training Accuracy: 0.968
  
  Validation Accuracy: 0.85
  
 
  Precision: 0.92
  
  Recall: 0.88


  

  💡 Future Improvements
  
  1. Use data augmentation to boost generalization.
  
  2. Experiment with transfer learning using models like MobileNet, ResNet.
  
  3. Try patch-level aggregation for WSI (Whole Slide Image) prediction.


  
  ###  References
  
  - Original Kaggle Dataset
  
  - Histopathologic Cancer Detection - Kaggle
  
  - Keras Image Classification









