import cv2
import numpy as np
from keras.models import model_from_json

class Character_Recognizer:

    def __init__(self):
        self.arabic_characters = ['alf', 'beh', 'teh', 'theh', 'gem', 'hah', 'khah', 'dal', 'zal',
                                  'reh', 'zen', 'sen', 'shen', 'sad', 'daad', 'tah', 'zah', 'een',
                                  'gheen', 'feh', 'qaaf', 'kaf', 'lam', 'mem', 'noon', 'heeh', 'waw', 'yeh']
        json_file = open("Characters Model/character model json.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("Characters Model/character weights.h5")

    def get_sides(self, length):
        if length%2==0:
            return length//2,length//2
        else:
            return (length-1)//2,1+(length-1)//2


    def preprocess(self, character):

        (wt, ht) = (32,32)
        (h, w) = character.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
        character = cv2.resize(character, newSize)

        if character.shape[0] < 32:
            add_zeros_up = np.zeros((self.get_sides(32-character.shape[0])[0], character.shape[1]))
            add_zeros_down = np.zeros((self.get_sides(32-character.shape[0])[1], character.shape[1]))
            character = np.concatenate((add_zeros_up,character))
            character = np.concatenate((character, add_zeros_down))

        if character.shape[1] < 32:
            add_zeros_left = np.zeros((32, self.get_sides(32-character.shape[1])[0]))
            add_zeros_right = np.zeros((32, self.get_sides(32-character.shape[1])[1]))

            character = np.concatenate((add_zeros_left,character), axis=1)
            character = np.concatenate((character, add_zeros_right), axis=1)


        character= character.T/255.0
        character = np.expand_dims(character , axis = 2)
        return character

    def ocr(self, img):
        # img  = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img  = self.preprocess(img)
        img = img.reshape(-1,32, 32, 1)
        pred = self.loaded_model.predict([[img]])
        return self.arabic_characters[np.argmax(pred)]