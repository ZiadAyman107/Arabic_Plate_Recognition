from Extract_Character import *
from Character_Recognizer import *
from digit_recognizer_ import *
from Car_Plate_Detection import *
cr = Character_Recognizer()
nr = Number_Recognizer()
Ec = Extract_Characters()
cp = Car_Plate_Detection()

# Change Path to Picture
image = cv2.imread('Test/2.jpg')

PlateImg = cp.Detect_Plate(image)
if PlateImg.all() == None:
    print("No plate found in image")
else:
    numbers, characters = Ec.extract(PlateImg)
    word = []
    for i in range(len(numbers)):
        word.append(nr.ocr(numbers[i]))

    for i in range(len(characters)):
        word.append(cr.ocr(characters[i]))
    image = cv2.resize(image, (480, 480))
    cv2.putText(image, str(word), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 0), 2, cv2.LINE_AA)
    cv2.imwrite("Output/2.png", image)
    cv2.imshow("Photo", image)
    cv2.waitKey(0)


