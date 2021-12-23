import cv2
from keras.models import load_model
import numpy as np


model = load_model('trained_modelXception.h5')

tipos = [
    'Mascarilla correctamente puesta',
    'Mascarilla por debajo de la nariz',
    'Mascarilla en la barbilla',
    'Sin mascarilla'
]


def predict(image):
    X_test = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    X_test.append(image)
    X_test = np.array(X_test)
    prediction = model.predict(X_test)
    return prediction


def predic_on_camera(index):
    cap = cv2.VideoCapture(index)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            print('La clasifico como tipo: {}'.format(tipos[np.argmax(predict(frame))]))

        if cv2.waitKey(2) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()


# Pasar como index el numero de la camara, en mi caso el 2 porque tengo una externa, la webcam integrada es el 0
def main():
    predic_on_camera(2)


if __name__ == '__main__':
    main()
