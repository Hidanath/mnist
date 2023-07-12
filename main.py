import numpy as np
import tkinter as tk
import win32gui
from PIL import ImageGrab
from mnist import Model

class RecognizeNumbers(tk.Canvas):

    def __init__(self, window):
        super().__init__(window, width=280, height=280) #Set canvas geometry
        window.resizable(False, False)
        
        self.bind("<ButtonPress-1>", self.draw)
        self.bind("<B1-Motion>", self.draw)
        recognize_button = tk.Button(text="Распознать", width=10, height=1, command=self.recognize)
        recognize_button.pack(side="bottom")  

        self.brush_size = 10

        self.model = Model()
        self.model.loadModel()

    def draw(self, event):
        self.create_oval(event.x - self.brush_size,
                            event.y - self.brush_size,
                            event.x + self.brush_size,
                            event.y + self.brush_size,
                            fill="black", outline="black")
        self.update()

    def recognize(self):
        image = self.grabCanvas()
        image = (255-np.array(image)).reshape((28*28)) #Pixel color inversion
        prediction = self.model.predict(image)

        print(np.argmax(prediction))
        self.delete("all")
        self.update()

    def grabCanvas(self):
        HWND = self.winfo_id() 
        rect = win32gui.GetWindowRect(HWND) #Get canvas coords
        image = ImageGrab.grab(rect) #Get canvas image
        image = image.resize((28, 28)).convert("L") #Resize and convert image to grayscale mode
        image = np.array(image)

        return image

if __name__ == "__main__":
    window = tk.Tk()
    window.geometry("280x310")
    recognizeNumbers = RecognizeNumbers(window)
    recognizeNumbers.pack()
    window.mainloop()