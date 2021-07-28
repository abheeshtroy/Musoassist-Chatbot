from tkinter import *
import json
from tkinter.filedialog import asksaveasfile
import sys
import os
 
window = Tk()
window.geometry('640x300')
window.title('trainin set generation')
# function to add to JSON 
def write_json(data, filename='intents.json'): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 

def check():
    with open('intents.json') as json_file: 
        data = json.load(json_file) 
        temp = data['intents'] 
        t = Tag.get()
        q1 = Qs1.get()
        q2 = Qs2.get()
        q3 = Qs3.get()
        q4 = Qs4.get()
        r1 = Rs1.get()
        data1 = {}
        data1['tag'] = t
        data1['patterns'] = [q1,q2,q3,q4]
        data1['responses'] = [r1]
        temp.append(data1)
    write_json(data)
    #clear textbox
    Tag.delete("0", "end")
    Qs1.delete("0", "end")
    Qs2.delete("0", "end")
    Qs3.delete("0", "end")
    Qs4.delete("0", "end")
    Rs1.delete("0", "end")
    #show success message
    T.insert(END, "Tag :"+t+" Saved Successfully..")
    

def clearBox(self):
    self.txt1.delete("1.0", "end")

#def run():
    #os.system('python train_bot.py')

#btn = Button(window, text="Train bot", bg="red", fg="white",command=run)
#btn.grid(column=12, row=1)

T = Text(window, height=2, width=30)
tag = Label(window, text="Tag:")
Tag = Entry(window)
qs1 = Label(window, text="Question-1:")
Qs1 = Entry(window)
qs2 = Label(window, text="Question-2:")
Qs2 = Entry(window)
qs3 = Label(window, text="Question-3:")
Qs3 = Entry(window)
qs4 = Label(window, text="Question-4:")
Qs4 = Entry(window)
rs1 = Label(window, text="Responce:")
Rs1 = Entry(window)
submit = Button(window,text='Submit',command = check).grid(row=6, column=1)
#placement of things in GUI
tag.grid(row=0, column=0)
qs1.grid(row=1,column=0)
qs2.grid(row=2,column=0)
qs3.grid(row=3,column=0)
qs4.grid(row=4,column=0)
rs1.grid(row=5,column=0)
Tag.grid(row=0, column=1)
Qs1.grid(row=1, column=1)
Qs2.grid(row=2, column=1)
Qs3.grid(row=3, column=1)
Qs4.grid(row=4, column=1)
Rs1.grid(row=5, column=1)
T.grid(row=9, column=1)
 
 
mainloop()
