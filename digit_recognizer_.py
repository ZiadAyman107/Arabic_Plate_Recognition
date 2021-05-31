import cv2
import numpy as np 
from keras.models import model_from_json

class Number_Recognizer:
    def __init__(self):
        self.arabic_digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        json_file = open("Characters Model/digits model json.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("Characters Model/digits weights.h5")

    def get_sides(self, length):
        if length % 2 == 0:
            return length//2,length//2
        else:
            return (length-1)//2,1+(length-1)//2

    def preprocess(self, character):
        (wt, ht) = (28, 28)
        (h, w) = character.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
        character = cv2.resize(character, newSize)
        if character.shape[0] < 28:
            add_zeros_up = np.zeros((self.get_sides(28-character.shape[0])[0], character.shape[1]))
            add_zeros_down = np.zeros((self.get_sides(28-character.shape[0])[1], character.shape[1]))
            character = np.concatenate((add_zeros_up,character))
            character = np.concatenate((character, add_zeros_down))
        if character.shape[1] < 28:
            add_zeros_left = np.zeros((28, self.get_sides(28-character.shape[1])[0]))
            add_zeros_right = np.zeros((28, self.get_sides(28-character.shape[1])[1]))

            character = np.concatenate((add_zeros_left,character), axis=1)
            character = np.concatenate((character, add_zeros_right), axis=1)
        character= character.T/255.0
        character = np.expand_dims(character , axis = 2)
        return character

    def ocr(self, img):
        img  = self.preprocess(img)
        img = img.reshape(-1,28, 28, 1)
        pred = self.loaded_model.predict(img)
        return self.arabic_digit[np.argmax(pred)]

