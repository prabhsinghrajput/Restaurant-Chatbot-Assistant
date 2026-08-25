# 🍽️ Spice & Savor Bistro - Next-Gen AI Dining Concierge

An intelligent, interactive, full-featured AI Restaurant Assistant powered by **Python, Flask, Google Gemini LLM with Function Calling / Action Tools**, and a **modern luxury Glassmorphic Web UI**.

---

## ✨ Key Features

- 🧠 **Next-Gen AI & Natural Language**: Powered by Google Gemini API with smart fallback. Understands complex dietary requests, dish pairings, and questions.
- 🥘 **Dynamic Menu Cards**: Rich interactive cards with dietary tags (🌱 Vegan, 🧀 Veg, 🍗 Non-Veg), calories, spice levels, allergens, and **[+ Add to Cart]** buttons.
- 🛒 **Live Order Cart & Checkout**: Slide-out cart drawer with item quantity counters, coupon code discounts (`SAVE20`, `WELCOME10`), tax & delivery calculation, and checkout.
- 📅 **Interactive Table Reservation**: Book tables with party size, date, time, and custom requests, generating stylized confirmation tickets (`#RES-XXXX`).
- 📦 **Real-Time Live Order Tracker**: Visual stepper timeline showing: *Order Confirmed ➔ Cooking in Kitchen ➔ Out for Delivery ➔ Delivered*.
- 🎙️ **Voice Recognition (Speech-to-Text)**: Speak to the bot naturally using the integrated microphone button.
- 🔊 **Text-to-Speech (Audio Voice)**: Toggle real-time audio narration of bot responses.
- 🎨 **Luxury Glassmorphic UI**: Built with modern typography, dark theme with warm amber accents, ambient glow effects, and responsive mobile-first design.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask, python-dotenv
- **AI & NLP**: Google Gemini API (`google-generativeai`) with Function Calling & Tool Calling, Intelligent Local Agent Router
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism + Animations), Modern JavaScript (Fetch API, Web Speech API, SpeechSynthesis)
- **Data**: Structured JSON / In-memory data store (`menu_data.json`)

---

## 📂 Project Structure

```
Restaurant-Chatbot-Assistant/
│
├── app.py                # Main Flask application & REST API endpoints
├── restaurant_agent.py   # AI Agent logic, tool execution & Gemini LLM router
├── menu_data.json        # Structured menu items, categories, allergens, promos
├── requirements.txt      # Clean Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/restaurant-chatbot-assistant.git
cd restaurant-chatbot-assistant
```

### 2️⃣ Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
# or: venv\Scripts\activate.bat
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ (Optional) Configure Gemini API Key
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Add your Gemini API key in `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
> **Note:** If you do not provide an API key, the chatbot will automatically run with its built-in **Intelligent Local Agent Engine** so all booking, cart, filtering, and tracking tools work immediately out of the box!

### 5️⃣ Run Application
```bash
python app.py
```

### 6️⃣ Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🧪 Interactive Chat Examples to Try

- *"Show me today's Chef Specials"*
- *"What vegan dishes do you have?"*
- *"Show vegetarian dishes under £12"*
- *"I want to book a table for 4 tonight at 8 PM under John"*
- *"What active discount coupons do you have?"*
- *"Track order #1042"*
- Click on any dish card's **+ Add** button, open the Cart drawer, apply code `SAVE20`, and complete checkout!

---

## 📜 License
This project is open-source and free for educational and demonstration purposes.
