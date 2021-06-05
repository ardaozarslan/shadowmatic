import cv2
import numpy as np
import os
from PIL import Image as pil

def on_trackbar(ySlideInt):
    global photo, shadow, mask, maskedImage, resizedPhoto

    croppedPhoto = resizedPhoto[0 + ySlideInt:croppedTheSize + ySlideInt, 0:croppedTheSize]
    croppedPhoto = cv2.cvtColor(croppedPhoto, cv2.COLOR_BGR2BGRA)
    blank_image = np.zeros((theSize, theSize, 4), np.uint8)
    blank_image[theGap:theDifference, theGap:theDifference] = croppedPhoto
    maskedImage = cv2.bitwise_and(blank_image, blank_image, mask=lastMask)
    maskedImage = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2BGRA)
    shadow = cv2.bitwise_and(shadow, shadow, mask=notLastMask)
    maskedImage = cv2.bitwise_or(maskedImage, shadow)

    cv2.imshow(windowName, maskedImage)

theSize = 500
theGap = 38
theDifference = theSize - theGap
croppedTheSize = theSize - 2 * theGap
windowName = "son"
dirName = "newcomers"
outputDirName = "outputs"

filesList = os.listdir(dirName)

for filePath in filesList:
    print(filePath)
    f = open(f"{dirName}/{filePath}", "rb")
    bytes = f.read()
    chunk_arr = np.frombuffer(bytes, dtype=np.uint8)
    photo = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)


    # photo = cv2.imread(f"photos/{filePath}")
    shadow = cv2.imread("shadow.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.imread("is_green.png", cv2.IMREAD_UNCHANGED)

    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    _, lastMask = cv2.threshold(grayMask, 180, 255, cv2.THRESH_BINARY_INV)
    _, notLastMask = cv2.threshold(grayMask, 180, 255, cv2.THRESH_BINARY)

    finalWidth = croppedTheSize

    width = photo.shape[1]
    scalePercent = finalWidth / width

    width = round(scalePercent * width)
    height = round(scalePercent * photo.shape[0])
    dim = (width, height)

    resizedPhoto = cv2.resize(photo, dim, interpolation=cv2.INTER_AREA)
    print(resizedPhoto.shape)
    ySliderMax = resizedPhoto.shape[0] - croppedTheSize

    cv2.namedWindow(windowName)
    trackbarName = "Height"
    cv2.createTrackbar(trackbarName, windowName, 0, ySliderMax, on_trackbar)

    on_trackbar(0)
    if cv2.waitKey() == ord("s"):
        continue
    elif cv2.waitKey():
        fileName = filePath.split(".")[0]
        maskedImage = cv2.cvtColor(maskedImage, cv2.COLOR_BGRA2RGBA)
        pil.fromarray(maskedImage).save(f"{outputDirName}/{fileName}.png")
        # cv2.imwrite(f"outputs/{fileName}.png", maskedImage)