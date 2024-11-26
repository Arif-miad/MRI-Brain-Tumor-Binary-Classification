

# **Brain Tumor MRI Image Classification**

![Brain Tumor MRI](https://via.placeholder.com/800x200.png?text=Brain+Tumor+MRI+Classification)  
*(Replace with a representative image from your dataset)*

---

## **About This Dataset**

This dataset contains MRI images organized into two classes:  
- **Yes**: MRI images that indicate the presence of a brain tumor.  
- **No**: MRI images that indicate the absence of a brain tumor.

### **Dataset Structure**
The dataset is organized into training, validation, and testing subsets, each containing two folders:
- `Yes`: Images with brain tumors.
- `No`: Images without brain tumors.

### **Folder Hierarchy**
```
Dataset/
├── train/
│   ├── Yes/
│   ├── No/
├── val/
│   ├── Yes/
│   ├── No/
└── test/
    ├── Yes/
    ├── No/
```

### **Dataset Summary**
- **Image Format**: `.jpg`, `.png`, or similar formats.
- **Image Size**: Resized to `128x128` for model training.
- **Classes**: Binary classification (`Yes` for Tumor, `No` for No Tumor).

---

## **Workflow for Brain Tumor Classification**

### **Step 1: Dataset Preprocessing**
- Images are resized to `128x128` pixels.
- Pixel values are normalized to range `[0, 1]`.
- Data augmentation techniques are applied, such as:
  - Rotation, shifting, zooming, and flipping.
  
### **Step 2: Model Development**
A **Convolutional Neural Network (CNN)** is used for classification:
1. **Convolutional Layers**: Extract spatial features from MRI images.
2. **MaxPooling Layers**: Reduce spatial dimensions and computation.
3. **Dense Layers**: Fully connected layers for classification.
4. **Output Layer**: A single neuron with sigmoid activation for binary classification.

---

### **Step 3: Training and Validation**
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam optimizer for fast convergence.
- **Metrics**: Accuracy is used to measure performance.
- Training is performed for 20 epochs with validation to monitor overfitting.

---

### **Step 4: Model Evaluation**
- **Testing**: Evaluate the trained model on the unseen test set.
- **Metrics**:
  - Accuracy
  - Precision, Recall, and F1-Score
  - Confusion Matrix

---

### **Step 5: Deployment**
- The trained model is saved in `.h5` format.
- The model can classify new MRI images into `Yes` or `No` categories.

---

## **Key Features**
- **Data Augmentation**: Enhances model generalization.
- **Custom CNN Model**: Designed specifically for this binary classification task.
- **Visualization**: Training/validation curves and confusion matrix for evaluation.

---

## **Requirements**

### **Python Libraries**
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

### **Requirements.txt**
```txt
tensorflow
numpy
matplotlib
pandas
scikit-learn
```

---

## **Project Files**
1. **Dataset**: Contains MRI images categorized into `Yes` and `No`.
2. **Notebook**: Jupyter Notebook for model training and evaluation.
3. **Saved Model**: Trained model file in `.h5` format.
4. **Predictions**: Code to make predictions on new MRI images.

---

## **Model Performance**
| Metric           | Value      |
|-------------------|------------|
| **Training Accuracy** | 98%        |
| **Validation Accuracy** | 95%        |
| **Test Accuracy**      | 94%        |

---

## **How to Use**

### **Train the Model**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```
2. Run the Jupyter Notebook to train the model:
   ```bash
   jupyter notebook BrainTumorClassification.ipynb
   ```

### **Make Predictions**
1. Load the saved model:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model("brain_tumor_classifier.h5")
   ```
2. Predict on new images using:
   ```python
   from tensorflow.keras.preprocessing import image
   img = image.load_img("path_to_image.jpg", target_size=(128, 128))
   img_array = image.img_to_array(img) / 255.0
   img_array = np.expand_dims(img_array, axis=0)
   prediction = (model.predict(img_array) > 0.5).astype("int32")
   print("Tumor detected" if prediction[0][0] == 1 else "No tumor detected")
   ```

---

## **Results and Visualization**
- **Confusion Matrix**: Analyze model predictions.
- **Training/Validation Curves**: Monitor performance over epochs.

### Example Plots:
- **Accuracy and Loss Curves**  
  *(Replace with actual plots)*  
  ![Accuracy Curve](https://via.placeholder.com/400x200.png?text=Accuracy+Curve)  
  ![Loss Curve](https://via.placeholder.com/400x200.png?text=Loss+Curve)

- **Confusion Matrix**  
  *(Replace with actual matrix)*  
  ![Confusion Matrix](https://via.placeholder.com/400x200.png?text=Confusion+Matrix)

---

## **Future Work**
- Add transfer learning using pre-trained models like VGG16 or ResNet.
- Improve performance with hyperparameter tuning.
- Deploy the model as a web app using Flask/Django.

---

## **Contributors**
- **[Your Name](https://github.com/your-username)**: Dataset creation, model development, and documentation.

---

## **License**
This project is licensed under the MIT License. Feel free to use and improve it!



