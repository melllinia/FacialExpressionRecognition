import cv2

class ImageProcessor:
    def __init__(self, img):
        self.img = cv2.imread(img)
        self.faces = None

    def find_faces(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        self.faces = {face_id: [x, y, w, h] for face_id, (x, y, w, h) in enumerate(faces)}

    def show_faces(self):
        cv2.imshow('Faces detected', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_cropped_faces(self):
        if self.faces is None:
            return None

        return {face_id: self.img[y:y + h, x:x + w] for face_id, (x, y, w, h) in self.faces.items()}

    def add_labels(self, labels):
        if self.faces is None:
            return None

        for face_id in self.faces:
            (x, y, w, h) = self.faces[face_id]
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(self.img, labels[face_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return self.img

