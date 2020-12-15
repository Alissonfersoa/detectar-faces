import cv2 as cv
import numpy as np

print('Carregando código.....')

#Realiza o loading do modelo treinado
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Seleciona a imagem
imagem = cv.imread('img/imagem4.jpg')

imagemCinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

detecta_face = face_cascade.detectMultiScale(imagemCinza)

#Percorre faces/rostos
for (x, y, l, a) in detecta_face:
    cv.rectangle(imagem,(x,y),(x+l,y+a),(255,0,0),2)

print('Face detectada!!.....')
cv.imshow("FACE", imagem)
cv.waitKey()
cv.destroyAllWindows()