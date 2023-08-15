import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mss
from scipy.interpolate import interp1d
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from pynput.mouse import Button, Listener
import keyboard
from PIL import ImageTk
import PIL.Image
from tkinter import Tk, Button, Label, Frame, IntVar, Radiobutton

## CONFIGS
# directory for saving data
data_path = 'D:/Downloads 2/Andre_Thesis/Data/WOLF_dataset'
file_name = 'Subject_3'

# directory with arousal.png and valence.png
selfeval_imgs_path = "D:/Downloads 2/Andre_Thesis/DataRecorder/selfeval_imgs"

# directory with pngs for image comparisson
game_events_path = "D:/Downloads 2/Andre_Thesis/DataRecorder/event_imgs"
tolerance = 16
# initializing image comparison
event_index = 0
images = os.listdir(game_events_path)

#
aux = []

## SELF EVALUATION
selfeval = []

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

## MOUSE
#event = ''
def on_click(x, y, button, pressed):
    #global event
    t = time.time()
    
    if button == button.left and pressed:
        ultracortex.insert_marker(float(t))
        emotibit.insert_marker(float(t), preset=BrainFlowPresets.AUXILIARY_PRESET)
        aux.append((t, x, y))
        
        #if event == 'event1':
         #   print(f'event1, t={t}, first click after recording')
          #  event = ''
           # window()
        #elif event == 'event2':
         #   print(f'event2, t={t}, first click after 10 sec')
          #  event = ''
           # window()
            
def start_click(x, y, button, pressed):
    global click_count
    if button == button.left and pressed:
        click_count += 1
        print(f"Left mouse click {click_count}")
        if click_count >= 2:
            """
            # hotkey to stop OBS recording
            with keyboard.pressed(Key.ctrl_l, Key.shift_l):
                keyboard.press('1')
                keyboard.release('1')
            time.sleep(2)
            print('start obs')
            """
            # stop the listener
            listener.stop()

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


# Saving
def create_saving_folder(path, folder_name):
    full_path = os.path.join(path, folder_name)
    counter = 1

    while os.path.exists(full_path):
        folder_name_ = f"{folder_name}_{counter}"
        full_path = os.path.join(path, folder_name_)
        counter += 1

    os.makedirs(full_path)
    return full_path




## RECORDING
BoardShim.enable_dev_board_logger()

# setting emotibit board
emotibit_params = BrainFlowInputParams()
#emotibit_presets = BoardShim.get_board_presets(BoardIds.EMOTIBIT_BOARD.value)
#print(emotibit_presets)
emotibit = BoardShim(BoardIds.EMOTIBIT_BOARD, emotibit_params)

# setting ultracortex board
ultracortex_params = BrainFlowInputParams()
ultracortex_params.serial_port = "COM3"
ultracortex = BoardShim(BoardIds.CYTON_DAISY_BOARD, ultracortex_params)

# prepare recording
print('preparing session')
template = set_template() #set first template for image comparison

emotibit.prepare_session()
ultracortex.prepare_session()

# start recording
click_count = 0
while (True):
    print("\nstart recording on OBS\nthen type \"start\" and press \"Enter\" to start recording")
    input1 = input()
    if input1 == "start":
        print("\nstart the game with the next two left mouse clicks")
        with Listener(on_click=start_click) as listener:
            listener.join()
        break
    else:
        print("wrong command!")

# Create mouse listener instance
listener = Listener(on_click=on_click)

print('\nrecording...')
print('press \"esc\" to stop recording\n')
emotibit.start_stream()
ultracortex.start_stream()

t0= time.time() # get start unix time
ultracortex.insert_marker(1)

# Start mouse listener
listener.start()

# Capture screen and compare with game events in order
while True:  
    screen_image = capture_screen()
    if image_comparison(screen_image, template, tolerance):
        if event_index in [1,4,5,6,7,9]: #if ISC epochs
            t = float(time.time())
            ultracortex.insert_marker(t)
            emotibit.insert_marker(t, preset=BrainFlowPresets.AUXILIARY_PRESET)
            aux.append((t, event_index, -1))
            #window() #test delete this ltr
        elif event_index in [0,2,3,8]: #if SelfEval
            window()
        elif event_index == 10:
            window()
            break
        event_index += 1
        template = set_template()
        
    #time.sleep(1)

    if keyboard.is_pressed("esc"):
        break

"""
event = 'event1'

time.sleep(5)
print(f'5s past, t={time.time()}') #append as event

window()

time.sleep(10)
event = 'event2'


while (True):
        print("\ntype \"stop\" and press \"Enter\" to stop recording\nthen stop recording on OBS")
        input1 = input()
        if input1 == "stop":
            break
        else:
            print("wrong command!")

"""
t1= time.time() # end unix time
listener.stop()

# get data
emotibit_data = emotibit.get_board_data(preset=BrainFlowPresets.DEFAULT_PRESET).T
emotibit_data_aux = emotibit.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET).T
emotibit_data_anc = emotibit.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET).T

ultracortex_data = ultracortex.get_board_data().T


# end recording
emotibit.stop_stream()
ultracortex.stop_stream()

emotibit.release_session()
ultracortex.release_session()

print('\nrecording is over')
print('saving data')

"""
print('stop obs')
with keyboard.pressed(Key.ctrl_l, Key.shift_l):
    keyboard.press('2')
    keyboard.release('2')
"""

# EMOTIBIT
# Upsample to match max length
ppg_len = len(emotibit_data_aux)

# select PPG, and marker channel
data = emotibit_data_aux[:,1:4]

# interpolate IMU
for i in range(1,10):
    interpolator = interp1d(np.arange(len(emotibit_data)), emotibit_data[:,i])
    interpolated = interpolator(np.linspace(0, len(emotibit_data) - 1, ppg_len)).reshape(ppg_len,1)
    data = np.concatenate((data, interpolated), axis=1)

# interpolate EDA, temp   
for i in range(1,3):
    interpolator = interp1d(np.arange(len(emotibit_data_anc)), emotibit_data_anc[:,i])
    interpolated = interpolator(np.linspace(0, len(emotibit_data_anc) - 1, ppg_len)).reshape(ppg_len,1)
    data = np.concatenate((data, interpolated), axis=1)

# reorder [IMUx9, PPGx3, EDA,Temp]
data = data[:, [3,4,5,6,7,8,9,10,11,0,1,2,12,13]]

# save data to DataFrame
df = pd.DataFrame(data, columns = ['accel_X', 'accel_Y', 'accel_Z', 
                                     'gyro_X', 'gyro_y', 'gyro_Z', 
                                     'mag_X', 'mag_y', 'mag_Z',
                                     'ppg_1', 'ppg_2', 'ppg_3',
                                     'eda1', 'temp'])

# ULTRACORTEX
# save eeg to DataFrame
df_eeg = pd.DataFrame(columns = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value))
for i, channel in enumerate(df_eeg.columns):
    df_eeg[channel] = ultracortex_data[:,i+1]

# AUX
aux = np.array(aux)
df_aux = pd.DataFrame(columns = ['unix_time', 'sample_125Hz', 'sample_25Hz', 'x_mouse', 'y_mouse'])
df_aux['unix_time'] = aux[:,0]
df_aux['sample_125Hz'] = [i for i, value in enumerate(ultracortex_data[:,-1]) if value in aux[:,0]]
df_aux['sample_25Hz'] = [i for i, value in enumerate(emotibit_data_aux[:,-1]) if value in aux[:,0]]
df_aux['x_mouse'] = aux[:,1]
df_aux['y_mouse'] = aux[:,2]

#adding start and end times
df_aux.loc[-1] = [t0, 0, 0, 'nan', 'nan']
df_aux = df_aux.sort_index().reset_index(drop=True)
df_aux.loc[len(df_aux)] = [t1, len(df_eeg), len(df), 'nan', 'nan']

# SELF EVALUATION
df_self = pd.DataFrame(selfeval, columns = ['popup_time','submit_time','arousal', 'valence', 'dominance'])

# SAVING
folder_path = create_saving_folder(data_path, file_name)

df.to_csv(folder_path + '/' + file_name + "_emotibit.csv")
df_eeg.to_csv(folder_path + '/' + file_name + "_ultracortex.csv")
df_aux.to_csv(folder_path + '/' + file_name + "_aux.csv")
df_self.to_csv(folder_path + '/' + file_name + "_selfeval.csv")

print(f'\nt0 = {t0} \nt1 = {t1}')
print(f'total duration: {aux[-1,0]-aux[0,0]}')
print('\n###\nEND\n###\n')









"""
def image_comparison(screen_image, folder_path, threshold=50):
    # Initialize the index of the last compared image
    if not hasattr(image_comparison, "event_index"):
        image_comparison.event_index = 0

    # Get the list of images in the folder
    images = os.listdir(folder_path)

    # Check if there are any more images to compare
    if image_comparison.event_index >= len(images):
        print('image_comparison.event_index >= len(images)')
        return False

    # Compare the screen image with the next image in the folder
    file = images[image_comparison.event_index]
    image_path = os.path.join(folder_path, file)
    template = PIL.Image.open(image_path)
    
    # Convert both images to the same mode
    if template.mode != screen_image.mode:
        template = template.convert(screen_image.mode)
    
    # Subtract one image from the other
    diff = PIL.ImageChops.difference(screen_image, template)

    # Calculate mean square error
    mse = np.sum(np.array(diff) ** 2) / (1920 * 1080)

    # If mse < threshold, return True
    if mse < threshold:
        t = time.time()
        print(f'\n{file} detected\nmse: {mse} \nunix_time: {t}')

        # Update the last compared image index for the next call
        image_comparison.event_index += 1
        return True

    print(f'{file} not detected, mse: {mse}')
    return False
"""
