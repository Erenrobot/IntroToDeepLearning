import cv2
import numpy as np

def convolution_uygulama(image_path, kernel):
    image= cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print("fotografin ici bos")
        return

    convolved_image = cv2.filter2D(image, -1, kernel)

    return convolved_image

if __name__ == "__main__":
    kernel = np.array([[1 / 13, 1 / 14, 1 / 15],[1 / 12, 1 / 13, 1 / 15],[1 / 11, 1 / 17, 1 / 15]],dtype=float)
    input_image_path="C://Users//Eren//Downloads//a.jpg"
    output_image_path="C://Users//Eren//Downloads//b.jpg"

    new_image= convolution_uygulama(input_image_path, kernel)

    cv2.imwrite(output_image_path, new_image)

    cv2.imshow("orijinal foto", cv2.imread(input_image_path,cv2.IMREAD_COLOR))
    cv2.imshow("convolution uygulanmis image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()