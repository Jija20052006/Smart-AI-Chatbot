from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import font, simpledialog, ttk

from ai_mode import AiMode, AiProvider, OnlineAiClient, OnlineAiConfig, OnlineAiError
from chatbot import ChatbotEngine, PERSONALITY_MAP
from config import AppConfig, ensure_project_dirs
from database import ChatDatabase
from memory import ChatMemory
from memory import JsonMemoryStore
from voice import VoiceIO


@dataclass(frozen=True)
class Theme:
    bg: str = "#0b1220"
    panel: str = "#0f172a"
    text: str = "#e5e7eb"
    muted: str = "#9ca3af"
    user_bubble: str = "#2563eb"
    bot_bubble: str = "#111827"
    bot_bubble_border: str = "#1f2937"
    input_bg: str = "#0b1324"
    input_border: str = "#223053"


class ScrollableMessages(ttk.Frame):
    def __init__(self, master: tk.Widget, *, theme: Theme) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=theme.bg, highlightthickness=0, bd=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.inner = tk.Frame(self.canvas, bg=theme.bg)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_inner_width)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _sync_scroll_region(self, _e: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_inner_width(self, _e: tk.Event) -> None:
        self.canvas.itemconfigure(self.inner_id, width=self.canvas.winfo_width())

    def _on_mousewheel(self, e: tk.Event) -> None:
        delta = getattr(e, "delta", 0)
        if delta:
            self.canvas.yview_scroll(int(-delta / 120), "units")

    def scroll_to_bottom(self) -> None:
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)


class SmartAIGui(tk.Tk):
    def __init__(self, engine: ChatbotEngine, memory: ChatMemory, voice: VoiceIO) -> None:
        super().__init__()
        self.engine = engine
        self.memory = memory
        self.voice = voice
        self.theme = Theme()
        self._online_busy = False
        self._listening = False

        self.title("SmartAI Chatbot")
        self.geometry("980x660")
        self.minsize(820, 560)

        self._init_fonts()
        self._apply_theme()
        self._build()
        self._ensure_username()
        self._render_history()
        self._welcome_back()

    def _init_fonts(self) -> None:
        self.font_ui = font.Font(family="Segoe UI", size=10)
        self.font_msg = font.Font(family="Segoe UI", size=11)
        self.font_name = font.Font(family="Segoe UI Semibold", size=11)

    def _apply_theme(self) -> None:
        self.configure(bg=self.theme.bg)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.theme.bg)
        style.configure("Top.TFrame", background=self.theme.panel)
        style.configure("TLabel", background=self.theme.panel, foreground=self.theme.text, font=self.font_ui)
        style.configure("Status.TLabel", background=self.theme.panel, foreground=self.theme.muted, font=self.font_ui)
        style.configure("TCheckbutton", background=self.theme.panel, foreground=self.theme.text, font=self.font_ui)
        style.configure(
            "TCombobox",
            fieldbackground=self.theme.input_bg,
            background=self.theme.input_bg,
            foreground=self.theme.text,
            arrowcolor=self.theme.text,
        )

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=12, style="Top.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(10, weight=1)

        ttk.Label(top, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=AiMode.OFFLINE)
        ttk.Radiobutton(top, text="Offline Smart", value=AiMode.OFFLINE, variable=self.mode_var).grid(row=0, column=1, padx=(6, 8), sticky="w")
        ttk.Radiobutton(top, text="Online AI", value=AiMode.ONLINE, variable=self.mode_var).grid(row=0, column=2, padx=(0, 10), sticky="w")

        ttk.Label(top, text="Provider").grid(row=0, column=3, sticky="w")
        self.provider_var = tk.StringVar(value=AiProvider.OPENAI)
        self.provider_combo = ttk.Combobox(
            top,
            textvariable=self.provider_var,
            values=[AiProvider.OPENAI, AiProvider.GEMINI, AiProvider.GROQ],
            state="readonly",
            width=10,
        )
        self.provider_combo.grid(row=0, column=4, padx=(6, 10), sticky="w")

        ttk.Label(top, text="Model").grid(row=0, column=5, sticky="w")
        self.model_var = tk.StringVar(value="gpt-4.1-mini")
        self.model_entry = ttk.Entry(top, textvariable=self.model_var, width=18)
        self.model_entry.grid(row=0, column=6, padx=(6, 10), sticky="w")

        ttk.Label(top, text="Personality").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.personality_var = tk.StringVar(value=self.engine.active_personality)
        self.personality_combo = ttk.Combobox(top, textvariable=self.personality_var, values=list(PERSONALITY_MAP.keys()), state="readonly", width=16)
        self.personality_combo.grid(row=1, column=1, padx=(6, 10), sticky="w", pady=(10, 0))
        self.personality_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_personality_change())

        self.tts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Voice (TTS)", variable=self.tts_var, command=self._on_tts_toggle).grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Button(top, text="Clear chat", command=self._clear_chat).grid(row=1, column=3, padx=(0, 8), pady=(10, 0))
        ttk.Button(top, text="New chat", command=self._new_chat).grid(row=1, column=4, padx=(0, 8), pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").grid(row=1, column=10, sticky="e", pady=(10, 0))

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.messages = ScrollableMessages(body, theme=self.theme)
        self.messages.grid(row=0, column=0, sticky="nsew")

        bottom = tk.Frame(self, bg=self.theme.bg)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            bottom,
            textvariable=self.input_var,
            font=self.font_msg,
            bg=self.theme.input_bg,
            fg=self.theme.text,
            insertbackground=self.theme.text,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.theme.input_border,
            highlightcolor=self.theme.user_bubble,
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(12, 8), pady=(0, 12), ipady=8)
        self.input_entry.bind("<Return>", lambda _e: self._send())

        self.mic_btn = ttk.Button(bottom, text="🎤", width=4, command=self._mic)
        self.mic_btn.grid(row=0, column=1, padx=(0, 8), pady=(0, 12))
        self.send_btn = ttk.Button(bottom, text="Send", command=self._send)
        self.send_btn.grid(row=0, column=2, padx=(0, 12), pady=(0, 12))
        self.input_entry.focus_set()

    def _ensure_username(self) -> None:
        if self.memory.profile.username:
            return
        name = simpledialog.askstring("SmartAI", "What's your name? (for memory)", parent=self)
        if name and name.strip():
            self.memory.set_username(name.strip())

    def _welcome_back(self) -> None:
        if self.memory.profile.username:
            self._add_bubble("assistant", f"Welcome back {self.memory.profile.username}")

    def _add_bubble(self, role: str, text: str) -> None:
        outer = tk.Frame(self.messages.inner, bg=self.theme.bg)
        outer.pack(fill="x", pady=6, padx=12)
        is_user = role == "user"
        bubble_bg = self.theme.user_bubble if is_user else self.theme.bot_bubble
        bubble_border = self.theme.user_bubble if is_user else self.theme.bot_bubble_border
        bubble = tk.Frame(outer, bg=bubble_bg, highlightthickness=1, highlightbackground=bubble_border, bd=0)
        bubble.pack(anchor="e" if is_user else "w")
        tk.Label(bubble, text=("You" if is_user else "SmartAI"), bg=bubble_bg, fg="#ffffff" if is_user else self.theme.muted, font=self.font_name).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(bubble, text=text, bg=bubble_bg, fg="#ffffff" if is_user else self.theme.text, font=self.font_msg, justify="left", wraplength=max(360, int(self.winfo_width() * 0.62))).pack(anchor="w", padx=12, pady=(0, 10))
        self.messages.scroll_to_bottom()

    def _render_history(self) -> None:
        for msg in self.memory.messages[-200:]:
            self._add_bubble(msg.role, msg.content)

    def _on_tts_toggle(self) -> None:
        self.voice.enabled = bool(self.tts_var.get())

    def _on_personality_change(self) -> None:
        confirm = self.engine.set_personality(self.personality_var.get())
        self.memory.set_favorite_personality(self.engine.active_personality)
        self.status_var.set(confirm)

    def _clear_chat(self) -> None:
        self.memory.start_new_conversation()
        for child in list(self.messages.inner.winfo_children()):
            child.destroy()
        self.status_var.set("Chat cleared")

    def _new_chat(self) -> None:
        self._clear_chat()

    def _set_online_busy_ui(self, busy: bool) -> None:
        self._online_busy = busy
        state = "disabled" if busy else "normal"
        self.send_btn.configure(state=state)
        self.mic_btn.configure(state=state)
        self.input_entry.configure(state=state)
        self.status_var.set("Online AI: thinking…" if busy else "Ready")

    def _set_listening_ui(self, listening: bool) -> None:
        self._listening = listening
        state = "disabled" if listening else "normal"
        self.send_btn.configure(state=state)
        self.mic_btn.configure(state=state)
        self.input_entry.configure(state=state)
        self.status_var.set("Listening…" if listening else "Ready")

    def _mic(self) -> None:
        if self._listening or self._online_busy:
            return
        self._set_listening_ui(True)

        def worker() -> None:
            try:
                text = self.voice.listen()
            except Exception as e:
                self.after(0, lambda: self.status_var.set(f"Mic error: {e}"))
                self.after(0, lambda: self._set_listening_ui(False))
                return

            def done() -> None:
                self._set_listening_ui(False)
                if text.strip():
                    self.input_var.set(text.strip())
                    self._send()
                else:
                    self.status_var.set("Didn't catch that. Try again.")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _generate_online_reply(self, user_text: str) -> str:
        cfg = OnlineAiConfig(provider=self.provider_var.get(), model=self.model_var.get() or "gpt-4.1-mini")
        client = OnlineAiClient(cfg)
        history = self.memory.messages[-12:]
        history_text = "\n".join([f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in history])
        prompt = (
            "You are SmartAI Chatbot. Be helpful, concise, and natural.\n"
            f"Personality: {self.engine.active_personality}\n\n"
            "Conversation so far:\n"
            f"{history_text}\n\n"
            f"User: {user_text}\nAssistant:"
        )
        return client.generate(prompt)

    def _send(self) -> None:
        text = self.input_var.get().strip()
        if not text or self._online_busy:
            return

        self.input_var.set("")
        self._add_bubble("user", text)
        self.memory.add("user", text, personality=self.engine.active_personality)

        if text.lower().startswith("switch personality"):
            parts = text.lower().split()
            new_p = parts[-1] if len(parts) > 2 else ""
            confirm = self.engine.set_personality(new_p)
            self.memory.set_favorite_personality(self.engine.active_personality)
            self._add_bubble("assistant", confirm)
            self.memory.add("assistant", confirm, personality=self.engine.active_personality)
            return

        if self.mode_var.get() == AiMode.ONLINE:
            self._set_online_busy_ui(True)

            def worker() -> None:
                try:
                    reply = self._generate_online_reply(text)
                except OnlineAiError as e:
                    self.after(0, lambda: self.status_var.set(f"Online AI error: {e}"))
                    reply = self.engine.reply(text)
                except Exception as e:
                    self.after(0, lambda: self.status_var.set(f"Online AI error: {e}"))
                    reply = self.engine.reply(text)

                def done() -> None:
                    self._set_online_busy_ui(False)
                    self._add_bubble("assistant", reply)
                    self.memory.add("assistant", reply, personality=self.engine.active_personality)
                    try:
                        self.voice.speak(reply)
                    except Exception as e:
                        self.status_var.set(f"Voice error: {e}")

                self.after(0, done)

            threading.Thread(target=worker, daemon=True).start()
            return

        reply = self.engine.reply(text)
        self._add_bubble("assistant", reply)
        self.memory.add("assistant", reply, personality=self.engine.active_personality)
        try:
            self.voice.speak(reply)
        except Exception as e:
            self.status_var.set(f"Voice error: {e}")


def run_gui(engine: ChatbotEngine, memory: ChatMemory, voice: VoiceIO) -> None:
    app = SmartAIGui(engine=engine, memory=memory, voice=voice)
    app.mainloop()


if __name__ == "__main__":
    # Direct GUI entrypoint (must not trigger CLI menu)
    ensure_project_dirs()
    cfg = AppConfig()
    engine = ChatbotEngine()
    memory = ChatMemory(store=JsonMemoryStore(cfg.memory_file), db=ChatDatabase(cfg.database_file))
    if memory.profile.favorite_personality:
        engine.set_personality(memory.profile.favorite_personality)
    if not memory.messages:
        memory.start_new_conversation()
    voice = VoiceIO(enabled=True)
    run_gui(engine=engine, memory=memory, voice=voice)

