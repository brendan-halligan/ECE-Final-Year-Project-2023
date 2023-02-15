import cv2


# Read the video file which frames are to be extracted from
video_path = "../Training Footage.mp4"
video = cv2.VideoCapture(video_path)
success = True
count = 0

while success:
    success, frame = video.read()

    # Extract all frames as pngs,saving each image to a folder
    # Name each frame using a standardised number format
    if count < 10:
        name = "./extracted_frames/frame_00000" + str(count) + ".png"
    elif count < 100:
        name = "./extracted_frames/frame_0000" + str(count) + ".png"
    elif count < 1000:
        name = "./extracted_frames/frame_000" + str(count) + ".png"
    else:
        name = "./extracted_frames/frame_00" + str(count) + ".png"

    if success is True:
        cv2.imwrite(name, frame)
        print("Frame {} Extracted Successfully".format(count))
        count = count + 1
    else:
        break
