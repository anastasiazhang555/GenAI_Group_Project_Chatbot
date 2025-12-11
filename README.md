# APAN5560GEN-AI-Project: 🧠 Empathetic Therapist Chatbot  
*A Generative AI emotional-support system combining LSTM emotion classification with an OpenAI-powered therapeutic dialogue model.*

---

## 🌟 Overview  
This project builds an **empathetic AI chatbot** capable of detecting user emotions and generating supportive, therapist-style responses.  
It is designed for the **APAN5560 – GenAI Group Project** and integrates:

- A **custom PyTorch LSTM emotion classifier** trained on the EmpatheticDialogues dataset  
- An **OpenAI GPT-based therapist persona**, supporting multi-turn conversation  
- A **Gradio web user interface** for interactive demo and live testing
- Supporting **FastAPI backend deployment**

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
├── app/
│   ├── main.py                 # FastAPI entrypoint (health + chat)
│   ├── chatbot.py              # Multi-turn OpenAI therapist persona
│   ├── emotion_model.py        # LSTM classifier + preprocessing
│   ├── web_ui.py               # Optional Gradio UI
│   ├── config.py               # Environment variable loading
│   └── schemas.py              # Request/response models
│
├── models/
│   ├── emotion_lstm.pt         # Trained PyTorch model weights
│   └── vocab.pkl               # Vocabulary for tokenizer
│
├── training/
│   ├── download_dataset.py     # Gets EmpatheticDialogues dataset
│   └── train_emotion_model.py  # Reproduces the LSTM classifier
│
├── data/                       # (Optional) downloaded dataset
│
├── Dockerfile                  # FastAPI deployment container
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
└── .env.example                # Example env file
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
### ▶️ Option 1 — Run the FastAPI backend
Start the server:
```
uvicorn app.main:app --reload
```
Now open:
```
http://127.0.0.1:8000/health
```
Expected output:
```
{"status": "ok"}
```
Chat endpoint: http://127.0.0.1:8000/chat
Example:
```
{
  "message": "I'm feeling anxious about my future."
}
```
Response Example:
```
{
  "emotion": "sadness",
  "confidence": 0.87,
  "response": "It sounds like you’re carrying a lot of worry right now..."
}
```


### ▶️ Option 2 — Launch the Gradio demo UI
```
python -m app.chatbot #▶️ Run the terminal chatbot
python -m app.web_ui  #🌐 Launch the Gradio web interface
```
This opens a friendly chat interface in the browser:
```
http://127.0.0.1:7860
```

## 🧪 Model Training Workflow
Download dataset:
```
python -m training.download_dataset      
```
Train LSTM emotion classifier:
```
python -m training.train_emotion_model   
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

## 👥 Team Members & Responsibility
	•	Model Building, Backend & Frontend Setup: Jiayin Zhang, Sitong Liu
	•	Report and Slides: Lanqi Zhang, Xiaoyu Zhu, Chloe

## 📜 License

This repository is for academic use as part of the APAN5560 course project.

## 🙏 Acknowledgements
	•	EmpatheticDialogues dataset (Facebook Research)
	•	OpenAI API
	•	Gradio Interface
	•	Columbia University – Applied Analytics Program