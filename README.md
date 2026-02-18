# 🎫 AI-Powered Support Ticket Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)

## � Live Demo
Check out the live application here: [Support Ticket Classification App](https://support-ticket-classification-and-prioritization.streamlit.app/)


## �📌 Project Overview
The **Support Ticket Classification System** is an intelligent automation tool designed to streamline IT support operations. By leveraging **Natural Language Processing (NLP)** and **Machine Learning**, the system automatically categorizes incoming support tickets and assigns priority levels, significantly reducing manual triage time and response latencies.

The project features a robust **DistilBERT** deep learning model for classification and a sleek **Streamlit Dashboard** for real-time interaction and analytics.

---

## 🚀 Key Features

### 🧠 Intelligent Classification
- **Automated Categorization**: Uses a fine-tuned **DistilBERT** model to classify tickets into specific departments (e.g., Hardware, Access, HR, Billing).
- **Smart Priority Assignment**: Implements a hybrid approach combined with keyword analysis to detect urgency (High, Medium, Low) based on critical terms like "outage", "urgent", or "error".

### 📊 Interactive Dashboard
A professional web interface built with **Streamlit** allows users to:
- **Classify Live Tickets**: Type a ticket description and get instant feedback.
- **View Real-time Analytics**: Visualizations of ticket distribution, top categories, and keyword clouds.
- **Explore Data**: Search and filter through historical ticket data.

### 📈 Analytics & Insights
- **Confusion Matrices**: Visual performance evaluation to understand model accuracy.
- **Word Clouds**: Discover trending issues through keyword visualization.

---

## 🛠️ Technology Stack
- **Language**: Python 3.8+
- **Deep Learning**: PyTorch, Hugging Face Transformers (DistilBERT)
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **Classical ML**: Scikit-learn (Random Forest baseline)

---

## 📂 Project Structure

```bash
├── 📁 saved_model/          # Stores the trained DistilBERT model & tokenizer
├── 📁 venv/                 # Virtual Environment (excluded from git)
├── .gitignore               # Git ignore file
├── all_tickets_processed_improved_v3.csv  # Dataset used for training
├── dashboard.py             # 🚀 Main Streamlit Dashboard application
├── dataset.py               # 🛠️ Data loading and preprocessing class
├── main.py                  # 📜 Legacy/Alternative Scikit-learn pipeline
├── predict.py               # 🔮 Inference script for CLI predictions
├── README.md                # 📘 Project documentation
├── requirements.txt         # 📦 List of dependencies
├── train.py                 # 🏋️‍♂️ Training script for the DistilBERT model
└── utils.py                 # 🧩 Helper functions for the dashboard
```

---

> **Important:** This project requires Python 3.8 to 3.12. Python 3.13 is currently incompatible with Streamlit due to the removal of `imghdr`. We recommend using Python 3.10.

## ⚙️ Installation & Setup


### 1. Clone the Repository
```bash
git clone https://github.com/pranay9981/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit plotly wordcloud  # Install additional dashboard requirements
```
> **Note:** Ensure you have PyTorch installed. If not, visit [pytorch.org](https://pytorch.org/get-started/locally/) for the command specific to your system.

---

## 🖥️ Usage Guide

### 1️⃣ Run the Dashboard (Recommended)
Launch the web interface to interact with the model:
```bash
streamlit run dashboard.py
```
*The dashboard will open automatically in your browser.*

### 2️⃣ Train the Model
If you want to retrain the model on new data:
```bash
python train.py
```
*This will train the DistilBERT model for 3 epochs and save artifacts to the `saved_model/` directory.*

### 3️⃣ Run CLI Predictions
Test the model quickly from the command line:
```bash
python predict.py
```

---

## 📊 Model Performance
The system is evaluated using Accuracy, Precision, Recall, and F1-Score.
- **Accuracy**: ~90%+ on test set.
- **Visuals**: Confusion matrices are generated during training to visualize misclassifications.

---

## 🔮 Future Improvements
- [ ] **Dockerization**: Containerize the application for easy deployment.
- [ ] **API Integration**: Expose a REST API using FastAPI.
- [ ] **Feedback Loop**: Allow users to correct predictions in the dashboard to retrain the model (Active Learning).

---

### 👨‍💻 Author
**Built by Pranay Bagaria**

- [GitHub](https://github.com/pranay9981)
- [LinkedIn](https://www.linkedin.com/in/pranay-bagaria/)
