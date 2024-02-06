import tkinter as tk
from tkinter import scrolledtext
import threading
import speech_recognition as sr
import pyttsx3

class SpeechToTextApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Speech to Text")

        self.text_output = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10)
        self.text_output.pack(padx=10, pady=10)

        self.start_button = tk.Button(master, text="Start Listening", command=self.start_listening)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Stop Listening", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # Initialize speech recognition and text-to-speech engines
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def start_listening(self):
        
        self.text_output.delete(1.0, tk.END)  # Clear previous text
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Start a new thread for listening to avoid freezing the GUI
        self.thread = threading.Thread(target=self.listen_and_recognize)
        self.thread.start()

    def stop_listening(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.recognizer.stop_listening()

    def listen_and_recognize(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                print("You said:", text)
                self.text_output.insert(tk.END, "You said: " + text + "\n")
                self.engine.say(text)
                self.engine.runAndWait()

            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio.")
                self.text_output.insert(tk.END, "Speech Recognition could not understand audio.\n")

            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                self.text_output.insert(tk.END, f"Error: {e}\n")

            finally:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()
