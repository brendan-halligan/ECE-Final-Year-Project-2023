import cv2


def perform_sift_location_check(path_to_video_file):
    """
    Function to compare the first frame of the passed video file to a specified satellite image.
    Comparison is performed using a Brute Force Scale Invariant Feature Transform (SIFT) in conjunction to a Lowe test.
    Two images are said to be of the same location if there are more than 30 good matches between the two.
    The resulting SIFT comparison is stored in the ``sift_images`` folder.

    :param path_to_video_file: Path to video file which object tracking will be performed.
    :return:                   True or False depending on whether the video's location matches the specified satellite
                               location.
    """
    # Get the first frame from the passed input file.
    cap = cv2.VideoCapture(path_to_video_file)
    ret, img1 = cap.read()

    """
    Read the satellite image to compare to the first frame of the input file.
    # NB: In a more sophisticated application, Google Earth's API could be used to retrieve these images.
    #     This implementation is outside the scope of this project.
    """
    img2 = cv2.imread("./sift_images/satellite_image.jpg")
    # img2 = cv2.imread("./sift_images/incorrect_location.jpg")

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

    # Initiate SIFT detector.
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect and compute the descriptors of the two images using the SIFT detector.
    kp_1, desc_1 = sift.detectAndCompute(img1, None)
    kp_2, desc_2 = sift.detectAndCompute(img2, None)

    # Instantiate a Brute Force matcher and determine the matches between the two images.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_1, desc_2, k=2)

    """
    Utilise Lowe's  test to separate good matches from all matches.
    Each keypoint of the first image is matched with a number of key points from the second image. The two best matches
    for each keypoint (i.e. the matches with the smallest distance) are kept.
    Lowe's test checks that the two distances are sufficiently different. If they are not, then the keypoint is 
    eliminated and will not be used for future matches.
    """
    good_points = []
    ratio       = 0.7
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_points.append([m])

    # Draw all the matches and the good matches on two separate images.
    img3 = cv2.drawMatchesKnn(img1, kp_1, img2, kp_2,
                              matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img4 = cv2.drawMatchesKnn(img1, kp_1, img2, kp_2,
                              good_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Write both images to a file.
    cv2.imwrite("./sift_images/all_matches_image.jpg", img3)
    cv2.imwrite("./sift_images/good_matches_image.jpg", img4)

    print("Number of Total Matches: ", len(matches))
    print("Number of Good Matches: ",  len(good_points))

    # If the number of matches is above a certain threshold, then both images are said to be of the same location.
    if len(good_points) > 100:
        print("The images are of the same location")
        return True
    else:
        print("The images are not of the same location")
        return False
