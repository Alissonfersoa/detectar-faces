import cv2 as cv   #Importa OpenCV

print('Carregando código.....')

#Realiza o loading do modelo treinado
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Seleciona a imagem
imagem = cv.imread('img/imagem1.jpg')

#Converte para preto e branco/escala de cinza
gray = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))

#Percorre faces/rostos
for (x, y, l, a) in faces:
    #desenha o rec nas faces
    cv.rectangle(imagem,(x,y),(x+l,y+a),(255,255,0),2)

    #Contador de faces
    #contador = str(faces.shape[0])
    #cv.putText(imagem, contador, (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    #cv.putText(imagem, "Quantidade de Faces: " + contador, (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

print('Face detectada!!.....')
cv.imshow("FACE", imagem)
#cv.imshow("GRAY", gray)
cv.waitKey()
cv.destroyAllWindows()