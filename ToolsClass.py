from ultralytics import YOLO
import cv2 as cv
import matplotlib.pylab as plt
import numpy as np


class Tools():

    def __init__(self):
        self.__black_img = np.zeros((512, 512, 3), np.uint8)
        self.__image = None
        self.__drawing = False
        self.__startPoint = (0, 0)
        self.__color = (255, 255, 255)
        self.__approxs_list = []


    def increase_light(self, image, gamma=0.4, show=False):
        LookUpTable = np.empty((1, 256), np.uint8)
        low_light_image = image

        for i in range(0, 256):
            LookUpTable[0][i] = np.clip(pow(i/255, gamma) * 255, a_min=0,a_max=255)

        high_light_image = cv.LUT(low_light_image, LookUpTable)

        if show:
            plt.figure(figsize=[8, 4])
            plt.subplot(121);plt.imshow(low_light_image[::, ::, ::-1]);plt.title('low light image')
            plt.subplot(122);plt.imshow(high_light_image[::, ::, ::-1]);plt.title('high light image')
            plt.show()

        return high_light_image


    def resize_with_aspect_ratio(self, image, width=None, height=None):
            h, w = image.shape[:2]
            
            if width is None and height is None:
                return image
            
            if width is not None:
                scale = width / w
                new_width, new_height = width, int(h * scale)
            else:
                scale = height / h
                new_width, new_height = int(w * scale), height

            return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)


    def paint(self):

        def draw_line(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                self.__drawing = True
                self.__startPoint = (x, y)

            if event == cv.EVENT_MOUSEMOVE:
                if self.__drawing:
                    cv.line(self.__black_img, pt1=self.__startPoint, pt2=(x, y), color=self.__color, thickness=3)
                    self.__startPoint = (x, y)

            if event == cv.EVENT_LBUTTONUP:
                self.__drawing = False


        cv.namedWindow('image')
        cv.setMouseCallback('image', draw_line)

        while True:
            cv.imshow('image', self.__black_img)
            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('b'):
                self.__color = (255, 0, 0)
            elif key == ord('g'):
                self.__color = (0, 255, 0)
            elif key == ord('r'):
                self.__color = (0, 0, 255)
            elif key == ord('w'):
                self.__color = (255, 255, 255)
            elif key == ord('c'):
                self.__black_img = np.zeros((512, 512, 3), np.uint8)
            elif key == ord('s'):
                cv.imwrite(filename="./paint.jpg", img=self.__black_img)
                

        cv.destroyAllWindows()
        
    
    def geometric_shapes_detection(self, image, show=False):
        def resize_with_aspect_ratio(image, width=None, height=None):
            h, w = image.shape[:2]
            
            if width is None and height is None:
                return image
            
            if width is not None:
                scale = width / w
                new_width, new_height = width, int(h * scale)
            else:
                scale = height / h
                new_width, new_height = int(w * scale), height

            return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)

        img = image

        max_width = 500 
        resized_original = resize_with_aspect_ratio(img, width=max_width)
        resized_edges = resize_with_aspect_ratio(img, width=max_width)

        max_height = max(resized_original.shape[0], resized_edges.shape[0])

        padded_original = cv.copyMakeBorder(resized_original, 0, max_height - resized_original.shape[0], 0, 0, cv.BORDER_CONSTANT)


        image_path = './detect1.jpg'

        image = padded_original
        if image is None:
            print("❌ Error: Image not found! Make sure the file path is correct.")


        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        edges = cv.Canny(blurred, 50, 150)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            self.__approxs_list.append(approx)

            M = cv.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0


            shape_name = "Unknown"
            if len(approx) == 3:
                shape_name = "triangle"
            elif len(approx) == 4:

                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = w / float(h)
                shape_name = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
            elif len(approx) > 6:
                shape_name = "circle"

            cv.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv.putText(image, shape_name, (cX - 20, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if show:
            cv.imshow("Detected Shapes", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        
        return self.__approxs_list


    def object_detection(self, image, show=False, save=True):

        model = YOLO("./yolov8n.pt")

        self.__image = image

        self.__image = cv.cvtColor(self.__image, cv.COLOR_BGR2RGB)

        results = model(self.__image)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                if conf > 0.4:
                    cv.rectangle(self.__image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv.putText(self.__image, f"Class: {cls}, Conf: {conf:.2f}", (x1, y1 - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if show:
            plt.imshow(self.__image)
            plt.show()

            if save:
                cv.imwrite("./detected_image.jpg", self.__image[::, ::, ::-1])
