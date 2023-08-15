import cv2
import os
import numpy as np
import mss
import time
from PIL import ImageTk
import PIL.Image
from tkinter import Tk, Button, Label, Frame, IntVar, Radiobutton
import matplotlib.pyplot as plt

# directory with arousal.png and valence.png
selfeval_imgs_path = "D:/Downloads 2/Andre_Thesis/DataRecorder/selfeval_imgs"

# directory with pngs for image comparisson
game_events_path = "D:/Downloads 2/Andre_Thesis/DataRecorder/event_imgs"
tolerance = 16
event_index = 4
images = os.listdir(game_events_path)
selfeval = []

## EVENT TRIGGER
def capture_screen():
    with mss.mss() as sct:
        screen = sct.shot(output="screen.png")
    screen = np.array(PIL.Image.open(screen))
    screen = screen[-1000:,...] #crop image so that the top bar is not considered
    #screen = screen - np.mean(screen)
    return screen

def set_template():
    file = images[event_index]
    image_path = os.path.join(game_events_path, file)
    template = np.array(PIL.Image.open(image_path).convert('RGB'))
    template = template[-1000:,...]
    #template = template - np.mean(template)
    return template

def image_comparison(screen_image, template, tolerance=12):
    # Subtract one image from the other
    #diff = PIL.ImageChops.difference(screen_image, template)
    diff = screen_image - template
    # Calculate mean square error
    #mse = np.sum(np.array(diff) ** 2) / (1920 * 1000)
    mse = np.mean(np.square(diff))
    
    # If mse < threshold, return True
    threshold = np.mean(template) + tolerance
    if mse < threshold:
        print(f'\nimage {event_index} detected, mse: {mse}, thr: {threshold}')
        return True
    #print(f'image {event_index} not detected, mse: {mse}, thr: {threshold}')
    return False

def event():
    print("Match found! Your function is called.")
    a = input()

def image(root,image):
    img = PIL.Image.open(image)
    img = img.resize((840, 211))
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image = img)
    panel.image = img
    return panel

def window():
    root = Tk()
    t_0 = time.time()
    root.eval('tk::PlaceWindow . center')
    root.title("Self evaluation")
    root.geometry('%dx%d+%d+%d' % (840, 900, 500, 50))
    root.focus_set()
    root.grab_set()
    Label(root, text="Assess your levels of arousal, valence, and dominance at the current moment"
          ,font=("Arial", 10, "bold")).grid(row=0,column=0)
    
    Label(root, text="Arousal, how alert or activated do you feel?").grid(row=1, column=0)
    Label(root, text="Valence, how positive or pleasant do you feel?").grid(row=4, column=0)
    Label(root, text="Dominance, how much in control do you feel?").grid(row=7, column=0)
    
    panel=image(root, selfeval_imgs_path + '/arousal.png')
    panel.grid(row=2, column=0)
    panel2=image(root,selfeval_imgs_path + '/valence.png')
    panel2.grid(row=5, column=0)
    panel3=image(root,selfeval_imgs_path + '/dominance.png')
    panel3.grid(row=8, column=0)

    frame = Frame(root)
    frame.grid(row=3, column=0, columnspan=9)
    frame2 = Frame(root)
    frame2.grid(row=6, column=0, columnspan=9)
    frame3 = Frame(root)
    frame3.grid(row=9, column=0, columnspan=9)

    var1 = IntVar()
    for i in range(1,10):
        Radiobutton(frame, text = str(i),variable=var1, value=str(i)).grid(row=3, column=i)
    var2 = IntVar()
    for i in range(1,10):
        Radiobutton(frame2, text = str(i),variable=var2, value=str(i)).grid(row=6, column=i)
    var3 = IntVar()
    for i in range(1,10):
        Radiobutton(frame3, text = str(i),variable=var3, value=str(i)).grid(row=9, column=i)


    def myClick():
        if not var1.get()==0 and not var2.get()==0 and not var3.get()==0:
            t_1= time.time()
            selfeval.append([t_0, t_1, var1.get(), var2.get(), var3.get()])
            root.grab_release()
            root.destroy()
            print('ANSWER: ', selfeval[-1][-3:])
    Button(root, text="Submit and Continue", command=myClick,bg='green'
           ,fg='white', activeforeground='red', padx=10, pady=5
           ,font=("Arial", 12), ).grid(row=10,column=0)
    root.bind('<Return>')
    root.mainloop()




template = set_template()

# Example usage:
while True:  # Main loop to continuously capture the screen and compare
    screen_image = capture_screen()
    if image_comparison(screen_image, template, tolerance):
        window()
        event_index += 1
        template = set_template()
    #time.sleep(1)  # Adjust the sleep time based on the desired checking frequency
    
    
    
    
### V2
