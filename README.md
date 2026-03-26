# ♻️ Waste Classification using CNN (ResNet18)

## 📌 Overview

This project is a deep learning-based waste classification system that categorizes waste into six classes: cardboard, glass, metal, paper, plastic, and trash. It helps automate waste segregation and improve recycling efficiency.

---

## 🎯 Objective

To build an AI-powered system that can classify waste images into recyclable categories using Convolutional Neural Networks (CNNs).

---

## 🧠 Model

* Transfer Learning using ResNet18
* Pretrained on ImageNet
* Fine-tuned on TrashNet dataset
* Achieved **92.6% validation accuracy**

---

## 📂 Dataset

* TrashNet dataset
* Categories:

  * Cardboard
  * Glass
  * Metal
  * Paper
  * Plastic
  * Trash

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Torchvision
* Streamlit
* Scikit-learn

---

## 📊 Results

* Accuracy: **92.6%**
* F1 Score: ~0.91
* Evaluation metrics:

  * Confusion Matrix
  * Precision / Recall

---

## 🚀 Features

* Image classification using CNN
* Web app using Streamlit
* Real-time prediction with confidence score
* Auto-download of trained model

---

## 🌐 Live Demo

👉 https://waste-classifier-acffnssuz4yqrzdgmrz2hb.streamlit.app/

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/DSJamwal2004/waste-classifier.git
cd waste-classifier

pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📌 Project Structure

```
waste-classifier/
├── app/
├── src/
├── data/
├── models/ (ignored)
├── requirements.txt
├── README.md
```

---

## 🔧 Challenges Faced

* Class imbalance (trash category)
* Similar features (glass vs plastic)
* Deployment with large model files

---

## 🔮 Future Improvements

* Real-time webcam detection
* Mobile app integration
* Improve accuracy using deeper models

---

## 👨‍💻 Author

**Devansh Jamwal**
