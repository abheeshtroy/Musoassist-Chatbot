from tkinter import *
import json
from tkinter.filedialog import asksaveasfile
import sys
import os
import gtts
window = Tk()
window.geometry('640x300')
window.title('text2video generation')
# function to add to JSON 

def t2mp3():
    q2 = Qs2.get()
    tts = gtts.gTTS(q2)
    tts.save("1.mp3")
    

def clearBox(self):
    self.txt1.delete("1.0", "end")

def run():
    t = Tag.get()
    q1 = Qs1.get()
    st='python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face '+q1+' --audio 1.mp3 --outfile results/'+t+'.mp4'
    os.system(st)

btn = Button(window, text="Video", bg="red", fg="white",command=run)
btn.grid(column=12, row=1)

T = Text(window, height=2, width=30)
tag = Label(window, text=" Rendered Video Name:")
Tag = Entry(window)
qs1 = Label(window, text="Source Video or image name:")
Qs1 = Entry(window)
qs2 = Label(window, text="Text to be sync:")
Qs2 = Entry(window)

submit = Button(window,text='T2MP3',bg="red", fg="white",command = t2mp3).grid(row=6, column=1)
#placement of things in GUI
tag.grid(row=0, column=0)
qs1.grid(row=1,column=0)
qs2.grid(row=2,column=0)

#rs1.grid(row=5,column=0)
Tag.grid(row=0, column=1)
Qs1.grid(row=1, column=1)
Qs2.grid(row=2, column=1)
#T.grid(row=9, column=1)
 
 
mainloop()
