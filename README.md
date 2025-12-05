# APAN5560GEN-AI-Project: 🧠 Empathetic Therapist Chatbot  
*A Generative AI emotional-support system combining LSTM emotion classification with an OpenAI-powered therapeutic dialogue model.*

---

## 🌟 Overview  
This project builds an **empathetic AI chatbot** capable of detecting user emotions and generating supportive, therapist-style responses.  
It is designed for the **APAN5560 – GenAI Group Project** and integrates:

- A **custom PyTorch LSTM emotion classifier** trained on the EmpatheticDialogues dataset  
- An **OpenAI GPT-based therapist persona**, supporting multi-turn conversation  
- A **Gradio web user interface** for interactive demo and live testing  

The result is a conversational agent that can:

✔️ Detect emotional tone (anger, sadness, joy, neutral)  
✔️ Respond in a warm, reflective, and validating style  
✔️ Offer actionable suggestions when the user explicitly asks for help  
✔️ Maintain context across multiple turns  

---

## 🚀 Key Features

### 🔹 1. Emotion Classification (LSTM Model)
- Custom PyTorch LSTM architecture  
- Tokenized text with vocabulary generation  
- Trained using EmpatheticDialogues dataset  
- Outputs emotion label + confidence score  
- Inference pipeline used inside the chatbot  

### 🔹 2. Therapist-Style Dialogue (OpenAI API)
- Multi-turn conversation with memory  
- System prompt defines therapeutic persona  
- Responses balance:
  - emotional validation  
  - reflective questioning  
  - gentle, practical suggestions when requested  

### 🔹 3. Web Demo (Gradio)
A clean and interactive Chat UI that allows users to:
- Type messages  
- View detected emotions  
- Read therapist-style responses  
- Engage in multi-turn dialogue  

Launch with:

```bash
python -m app.web_ui
```

---
## 📂 Project Structure
```
APAN5560GEN-AI-Project/
│
├── app/
│   ├── chatbot.py               # OpenAI integration + multi-turn logic + emotion pipeline
│   ├── emotion_model.py         # LSTM emotion classifier + inference pipeline
│   ├── web_ui.py                # Gradio web interface
│
├── training/
│   ├── download_dataset.py      # Downloads EmpatheticDialogues dataset to CSV
│   ├── train_emotion_model.py   # Trains LSTM classifier and saves weights + vocab
│
├── data/                        # (Optional) local dataset storage
├── models/                      # Trained LSTM model weights (emotion_lstm.pt, vocab.pkl)
│
├── requirements.txt
├── README.md
```

## 🛠️ Installation
```
git clone https://github.com/anastasiazhang555/GenAI_Group_Project_Chatbot.git  #Clone the repository
cd GenAI_Group_Project_Chatbot
```
```
python3 -m venv .venv                                                           #Create a virtual environment
source .venv/bin/activate
```
```
pip install -r requirements.txt                                                 #Install dependencies
```
```
export OPENAI_API_KEY=“copy and paste APIkey”                        #Set your OpenAI API key
```

## 🎯 Usage
```
python -m app.chatbot #▶️ Run the terminal chatbot
python -m app.web_ui  #🌐 Launch the Gradio web interface
```
```
http://127.0.0.1:7860 #Open in the browser
```

## 🧪 Model Training Workflow
```
python -m training.download_dataset      #1. Download dataset
```
```
python -m training.train_emotion_model   #2. Train LSTM emotion classifier
```

## 🧬 System Architecture
```
User Input
    ↓
LSTM Emotion Classifier  ───→  emotion label + confidence
    ↓
Therapist Prompt + Conversation Memory
    ↓
OpenAI GPT Response Generator
    ↓
Chatbot Output (reflective, warm, supportive)
```
⸻

## 👥 Team Members
	•	Jiayin Zhang
	•	Sitong Liu
	•	Lanqi Zhang
    •	Xiaoyu Zhu
	•	Chloe

## 📜 License

This repository is for academic use as part of the APAN5560 course project.

## 🙏 Acknowledgements
	•	EmpatheticDialogues dataset (Facebook Research)
	•	OpenAI API
	•	Gradio Interface
	•	Columbia University – Applied Analytics Program