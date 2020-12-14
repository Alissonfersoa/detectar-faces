import cv2

print('Carregando código.....')

#Realiza o loading do modelo treinado
loadingCls = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#Seleciona a imagem
imagem = cv2.imread('img/imagem1.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detecta = loadingCls.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=8, minSize=(25, 25))


cv2.imshow("Faces", detecta)
cv2.waitKey()