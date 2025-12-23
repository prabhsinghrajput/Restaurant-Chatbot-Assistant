🤖 Restaurant Chatbot Assistant

📌 Project Overview

Restaurant Chatbot Assistant is an intelligent, interactive web-based chatbot built using Python, Flask, and Machine Learning (NLP).
It helps restaurant customers quickly get information about menus, offers, delivery, timings, location, reservations, and more.

The chatbot uses TF-IDF + Naive Bayes for intent classification and provides a modern chat UI with quick-access buttons.

✨ Key Features

💬 Interactive chatbot UI (HTML, CSS, JavaScript)
🧠 NLP-based intent classification
📋 Menu, offers, delivery & reservation queries
🔄 Context-aware responses
🎨 Modern animated UI with floating food emojis
⚡ Fast response using trained ML model
📱 Fully responsive design

🛠️ Tech Stack
--Backend

Python
Flask
Scikit-learn
NLTK
Naive Bayes Classifier
TF-IDF Vectorizer

--Frontend

HTML5
CSS3 (Custom UI)
JavaScript (Fetch API)

📂 Project Structure
ChatbotWebApp/
│
├── app.py              # Main Flask application
├── intents.json        # Training data (intents & responses)
├── requirements.txt    # Python dependencies
├── venv/               # Virtual environment (not pushed to GitHub)
└── README.md

⚙️ How It Works

1.User sends a message via chat UI
2.Message is tokenized & stemmed
3.TF-IDF converts text into vectors
4.Naive Bayes predicts the intent
5.Bot selects a suitable response
6.Context is maintained using Flask sessions

🚀 Installation & Run
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/restaurant-chatbot-assistant.git
cd restaurant-chatbot-assistant

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000

📸 Screenshots

Interactive chatbot UI with quick menu buttons and animated background

<img width="1920" height="1200" alt="Screenshot (467)" src="https://github.com/user-attachments/assets/ad359cb7-0b96-4cd4-a87a-fc928d5be1c5" />

📈 Future Improvements

🔐 User authentication
🧾 Order placement support
🗄️ Database integration
🌍 Multi-language support
☁️ Cloud deployment (Render / Railway / AWS)

📜 License

This project is developed for educational and demonstration purposes.
Feel free to modify and enhance it.

👨‍💻 Developer

Prabhjot Singh
Python | Flask | Machine Learning | NLP

⭐ If you like this project, don’t forget to star the repository!
