import cv2
import numpy as np

class Extract_Characters:
    def extractCharacters(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret2,binary_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary_img = ~binary_img
        kernel = np.ones((3,3),np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=4)
        resized_character = []
        for i in range(len(centroids)):
          x = stats[i, cv2.CC_STAT_LEFT]
          y = stats[i, cv2.CC_STAT_TOP]
          w = stats[i, cv2.CC_STAT_WIDTH]
          h = stats[i, cv2.CC_STAT_HEIGHT]
          if h <= 80 and h >= 20 and w >= 15 and w < 30:
            source = binary_img[y-10:y+h+10,x:x+w]
            source = cv2.resize(source, (32,32))
            source = cv2.resize(source, (16,16))
            source = cv2.copyMakeBorder(source,8,8,8,8,0)
            resized_character.append((source, x))
        resized_character= sorted(resized_character,key=lambda x: x[1])
        return resized_character

    def extract(self, original_img):
        resized_img = cv2.resize(original_img, (200, 150))
        resized_num_character = self.extractCharacters(resized_img[:, 0:100])
        resized_char_character = self.extractCharacters(resized_img[:, 100:])
        return [x[0] for x in resized_num_character], [x[0] for x in resized_char_character]