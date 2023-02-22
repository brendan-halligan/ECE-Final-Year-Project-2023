import cv2
import matplotlib.pyplot as plt
import numpy             as np


def determine_optimal_lowe_threshold(save_path=""):
    """
    This function is used to find the optimal Lowe threshold value for the provided images.
    The Lowe Test is outlined as follows:
        - Each keypoint of the first image is matched with a number of key points from the second image.
        - The best two matches for each keypoint is kept. 'Best matches' are matches with the smallest distance
          measurement.
        - Lowe's test checks that the two distances are sufficiently different.
        - If they are not, then the keypoint is eliminated and will not be used for future calculations.

    This function iterates through several threshold values to find the value which produces the most good matches for
    the application at hand. Accuracy is defined as being the ratio of good matches to false matches.

    :return: None
    """
    # Get the first frame from the passed input file.
    cap       = cv2.VideoCapture(save_path)
    ret, img1 = cap.read()

    """
    Read the satellite image to compare to the first frame of the input file.
    NB: In a more sophisticated application, Google Earth's API could be used to retrieve these images.
        This implementation is outside the scope of this project.
    """
    img2          = cv2.imread("satellite_image.jpg")

    """
    Convert the satellite image to the same resolution as the first frame of the input video.
    i.e. scale the satellite image to a 3840x2160 resolution.
    """
    height, width = img2.shape[:2]
    aspect_ratio  = width / height
    target_height = 2160
    target_width  = int(target_height * aspect_ratio)
    img2          = cv2.resize(img2, (target_width, target_height), fx=aspect_ratio, fy=aspect_ratio)

    # Convert both images to gray-scale to improve SIFT accuracy and performance.
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to both images to improve SIFT performance.
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    """
    Instantiate and apply a Contrast Limited AHE (CLAHE).
    CLAHE is a variant of adaptive histogram equalization that limits contrast amplification to reduce noise 
    amplification.
    Read more about CLAHE here: https://www.analyticsvidhya.com/blog/2022/08/image-contrast-enhancement-using-clahe
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1  = clahe.apply(img1)
    img2  = clahe.apply(img2)

    # Apply Canny filter to enhance edges.
    img1 = cv2.Canny(img1, 50, 150)
    img2 = cv2.Canny(img2, 50, 150)

    # Initiate SIFT detector.
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect and compute the descriptors of the two images using the SIFT detector.
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    thresholds    = np.arange(0.1, 1, 0.1)
    good_matches  = []
    false_matches = []

    """
    Iterate through each Lowe threshold value.
    Instantiate a Brute Force matcher and determine the matches between the two images for each iteration.
    Count the number of good and false matches. 
        - A good match is a match that is within a specified distance of a keypoint.
    """
    for t in thresholds:
        matcher           = cv2.BFMatcher(crossCheck=False)
        matches           = matcher.knnMatch(des1, des2, k=2)
        num_good_matches  = 0
        num_false_matches = 0

        # Iterate through each match, classifying good and false matches.
        for m, n in matches:
            if m.distance < t * n.distance and m.distance < 10:
                num_good_matches += 1
            else:
                num_false_matches += 1

        good_matches.append(num_good_matches)
        false_matches.append(num_false_matches)

    """
    Calculate the ratio of good matches to false matches.
    Scale the ratios to a 0 to 1 scale.
    """
    ratios  = np.array(good_matches) / np.array(false_matches)
    min_val = np.min(ratios)
    max_val = np.max(ratios)
    ratios  = (ratios - min_val) / (max_val - min_val)

    # Generate a curve of best fit for the scaled ratios.
    p       = np.polyfit(thresholds, ratios, 5)
    x_curve = np.linspace(thresholds.min(), thresholds.max(), 100)
    y_curve = np.polyval(p, x_curve)

    # Ploy the curve of best fit.
    plt.plot(x_curve, y_curve)
    plt.xlabel("Lowe Thresholds")
    plt.ylabel("Scaled Accuracy")
    plt.title("Optimal Lowe Ratio")

    # Ensure that the curve tends to the minimum and maximum values of the dataset
    y_range = ratios.ptp()
    y_min   = ratios.min() - 0.6 * y_range
    y_max   = ratios.max() + 0.6 * y_range
    plt.ylim([y_min, y_max])
    plt.show()

    # Calculate the optimal Lowe threshold as the maximum ratio value.
    optimal_threshold = thresholds[np.argmax(ratios)]
    print("Optimal Lowe Threshold: ", optimal_threshold)


if __name__ == "__main__":
    determine_optimal_lowe_threshold()
