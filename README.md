
# 🧠 Brain Tumor Segmentation using Machine Learning

This project focuses on automated brain tumor segmentation in MRI scans using deep learning techniques, specifically a ResNet101-based Convolutional Neural Network (CNN). It aims to enhance the accuracy, speed, and consistency of brain tumor detection, aiding early diagnosis and personalized treatment planning.

---

## ✅ Objective

To build a machine learning model capable of accurately segmenting brain tumors (gliomas, meningiomas, and pituitary tumors) from MRI images using deep learning, thereby reducing the time and error associated with manual diagnosis.

---

## 🧪 Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **Google Colab**
- **Streamlit** (for UI/Visualization)

---

## 🧠 Model Overview

- **Architecture**: ResNet101-based CNN
- **Layers**: 104 convolutional layers (33 blocks, 29 reused from earlier blocks)
- **Pre-trained**: On ImageNet (224x224 resolution)
- **Approach**: Identity shortcut connections to support deeper training without degradation

---

## 📂 Dataset

- Annotated MRI scans containing images labeled for glioma, meningioma, and pituitary tumors.
- Preprocessing included resizing, normalization, and augmentation.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run app.py
   ```

> You can also run training in **Google Colab** using the provided `.ipynb` notebook.

---

## 📊 UML Diagrams Included

- Activity Diagram
- Sequence Diagram
- Use Case Diagram
- Component Diagram
- Class Diagram

---

## 📈 Features

- Automated tumor detection and segmentation
- User-friendly interface with Streamlit
- Supports multiple tumor types
- Efficient diagnosis support for radiologists

---

## ✅ Advantages

- Improves diagnostic **accuracy and speed**
- Reduces **manual effort**
- Enables **early detection** and **personalized treatment plans**

---

## ⚠️ Limitations

- Requires large annotated datasets
- High computational resource needs
- AI predictions lack full transparency

---

## 🔮 Future Scope

- Real-time tumor segmentation in clinical settings
- Integration with surgical systems for intra-operative guidance
- Personalized therapy recommendations using AI-based tumor profiling


