from gtts import gTTS
from tkinter import Tk, Label, Text, Entry, Button, filedialog
import os
import pygame
from tkinter import PhotoImage

def text_to_speech(text, language='en', output_file='output.mp3'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(output_file)
    return output_file

def on_button_click():
    text = text_box.get("1.0", "end-1c")  # Get text from the Text widget
    if text.strip():  # Check if there is non-empty text
        audio_output_file = audio_output_entry.get()

        if not audio_output_file:
            audio_output_file = 'output.mp3'

        audio_file = text_to_speech(text, language='en', output_file=audio_output_file)

        # Open the audio file automatically
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Update play and pause buttons
        play_button.config(state='disabled')
        pause_button.config(state='normal')

def on_pause_click():
    pygame.mixer.music.pause()
    play_button.config(state='normal')
    pause_button.config(state='disabled')

def on_play_click():
    pygame.mixer.music.unpause()
    play_button.config(state='disabled')
    pause_button.config(state='normal')

def on_close_click():
    # Stop the audio before closing
    pygame.mixer.music.stop()
    root.destroy()  # Close the Tkinter window

# Create a simple GUI
root = Tk()
root.title("Text-to-Speech Player")

label = Label(root, text="Enter Text:")
label.pack()

text_box = Text(root, width=50, height=10)
text_box.pack()

audio_output_label = Label(root, text="Audio Output File:")
audio_output_label.pack()

audio_output_entry = Entry(root, width=50)
audio_output_entry.pack()

button = Button(root, text="Text to Voice", command=on_button_click)
button.pack()

play_button_img = PhotoImage(data="R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")  # Default play button image
play_button = Button(root, image=play_button_img, command=on_play_click, state='disabled')
play_button.pack()

pause_button_img = PhotoImage(data="R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")  # Default pause button image
pause_button = Button(root, image=pause_button_img, command=on_pause_click, state='disabled')
pause_button.pack()

close_button = Button(root, text="Close", command=on_close_click)
close_button.pack()

# Continuously run the Tkinter event loop
root.mainloop()
