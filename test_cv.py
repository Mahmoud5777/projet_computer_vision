import cv2
import numpy as np

def detect_contours_and_coins(image_path, min_area=2000, canny_thresh=(25, 75)):
    img = cv2.imread(image_path)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, *canny_thresh, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            large_contours.append(approx)

    cv2.drawContours(output, large_contours, -1, (255, 0, 0), 2)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,         # 🔧 augmenter pour éviter les cercles proches
        param1=120,         # seuil Canny
        param2=60,          # 🔧 plus haut = plus strict
        minRadius=40,       # 🔧 ignorer les petits cercles parasites
        maxRadius=70
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"{len(circles[0])} pièce(s) détectée(s).")
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output, center, radius, (255, 0, 0), 2)    
            cv2.circle(output, center, 3, (0, 0, 255), 3)             
    else:
        print("Aucune pièce détectée.")
    
    cv2.namedWindow('Contours + Pièces', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Contours + Pièces', 1200, 800)
    cv2.imshow("Contours + Pièces", output)
    cv2.imwrite("fusion_result_cleaned.jpg", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_contours_and_coins("image2.jpg")
