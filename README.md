<<<<<<< HEAD
## SMART AI CHATBOT PRO (Jarvis Style)

SMART AI CHATBOT PRO is a **resume-ready Python desktop assistant** with:

- **Modern dark Tkinter UI** with chat bubbles + microphone input
- **Offline Smart Mode (default)**: typo-tolerant matching, keyword + mood detection, lightweight context follow-ups
- **Online AI Mode (optional)**: OpenAI or Gemini via environment variables (falls back to offline if unavailable)
- **Memory**: username + favorite personality + chat history (JSON-first)
- **SQLite database**: users + chat history + message_count + favorite_mode + last_login

## Project structure

- `main.py`: CLI + GUI launcher + command routing
- `gui.py`: premium dark Tkinter app with bubbles, timestamps, typing state
- `chatbot.py`: intent-first offline brain + personality tone layer
- `voice.py`: text-to-speech + microphone speech-to-text with mute support
- `memory.py`: JSON memory store + in-memory buffer + SQLite integration
- `database.py`: SQLite persistence (`users`, `chat_history`, `settings`, `analytics`)
- `utils.py`: shared utilities (timestamps + slash command parser)
- `config.py`: app configuration and startup directory creation
- `ai_mode.py`: optional OpenAI/Gemini online mode (stdlib HTTP)
- `assets/`: UI/static assets directory
- `data/`: runtime JSON/SQLite storage

## Slash commands

- `/help`
- `/time`
- `/date`
- `/joke`
- `/clear`
- `/history`
- `/mode funny`
- `/mode motivator`
- `/mode angry`
- `/mode professional`

## Requirements

- Python **3.10+** recommended (works with modern 3.x)
- Windows: voice features rely on SAPI via `pyttsx3` dependencies

Install:

```bash
python -m pip install -r requirements.txt
```

## Run

### GUI (recommended)

```bash
python main.py --gui
```

### CLI

```bash
python main.py
```

Backward-compatible:

```bash
python chatbot.py
```

## Modes

### 1) Offline Smart Mode (default)

Works with **no API keys**. Includes:

- Typo tolerance (`helo`, `thnaks`, etc.)
- Better keyword detection with aliases (e.g. `cv` → `resume`)
- Mood detection (anxious/angry/sad/tired/confused)
- Context follow-ups (e.g. “tell me more”, “what do you mean?”)

### 2) Online AI Mode (optional)

Enable in the GUI by switching **Mode → Online AI**.

#### OpenAI

Set:

- `OPENAI_API_KEY`

Default endpoint: OpenAI Responses API (`/v1/responses`).

#### Gemini

Set:

- `GEMINI_API_KEY`

Uses native `generateContent` REST endpoint.

If keys are missing or a request fails, the app **automatically falls back** to Offline Smart Mode.

## Data files

- `smartai_memory.json`: username + favorite personality + last conversation history
- `smartai.db` (default): SQLite database for users and chat history

## Notes for GitHub

- Do **not** commit `.venv/` or local DB/history files.
- Add your screenshots and a short demo GIF for a strong portfolio README.

=======
# Smart-AI-Chatbot
AI chatbot with multiple personalities (Funny, Motivator, Angry, Professional) and GUI
>>>>>>> fe76314d6d542f3c7fa86a2f346ebbfd4ded42eb
