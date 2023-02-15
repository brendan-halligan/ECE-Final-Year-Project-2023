import copy
import cv2
import numpy
import os
import re
import time


def generate_heat_map(save_path):
    """
    Function to generate a heatmap for each object detected in the video whose path is passed as a paramater.
    A heatmap is generated using Intel's Motion Heatmap technology, which utilises background subtraction to identify
    moving objects in a frame.

    :param save_path: Path to the source video footage.
    :return:          None.
    """
    print("Beginning HeatMapping Process...")

    # Get the time at which the heat mapping process begins.
    t0 = time.time()

    # Setup variables to read video footage from provided path.
    capture                   = cv2.VideoCapture(save_path)
    number_of_frames          = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    first_iteration_indicator = 1

    # Instantiate an MOG background subtractor object.
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    """
    Iterate through each frame in the provided video footage. 
    Apply the background subtractor to each frame, and subsequently impose the heatmap to said frame.
    Store each frame in the ``heatmap/frames`` directory.
    """
    for i in range(0, int(number_of_frames)):
        ret, frame = capture.read()

        # If the current frame is the first frame in the video, initialise the accumulated heatmap image to this frame.
        if first_iteration_indicator == 1:
            initial_frame             = copy.deepcopy(frame)
            height, width             = frame.shape[:2]
            accumulated_image         = numpy.zeros((height, width), numpy.uint8)
            first_iteration_indicator = 0
        else:
            """
            Filter the current frame using the background subtractor.
            The strength of the background subtracted image is governed by the intensity of the kernel.
            """
            filtered_image = background_subtractor.apply(frame)
            kernel         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)

            # Set the threshold intensity of the heatmap.
            threshold = 2
            maxValue  = 2
            ret, th1  = cv2.threshold(filtered_image, threshold, maxValue, cv2.THRESH_BINARY)

            """
            Superimpose the accumulated image onto the original frame.
            Note that the accumulated image must be converted from gray-scale to colour prior to merging. This 
            conversion is completed using OpenCVs HOT colour map.
            Store the merged frame to the ``heatmap/frames`` folder. Each frame is chronologically numbered.
            """
            accumulated_image   = cv2.add(accumulated_image, th1)
            colour_mapped_image = cv2.applyColorMap(accumulated_image, cv2.COLORMAP_HOT)
            video_frame         = cv2.addWeighted(frame, 0.9, colour_mapped_image, 0.3, 0)
            name                = "./heatmap/frames/frame%d.jpg" % i
            cv2.imwrite(name, video_frame)
            print(f"video 1/1 ({i}/{int(number_of_frames)}) HeatMap Added")

    # Create a final overlay of the entire heatmap. Save this to the ``heatmap`` folder.
    color_image    = cv2.applyColorMap(accumulated_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(initial_frame, 0.9, color_image, 0.3, 0)
    cv2.imwrite("./heatmap/diff-overlay.jpg", result_overlay)

    """
    NB: All code in the remainder of this function is used to merge all of the stored frames into a single mp4 video 
    file.
    """

    # Sort each image in natural order based on their filenames.
    images = [img for img in os.listdir("./heatmap/frames/")]
    images.sort(key=natural_keys)

    # Set the size of the output video to the size of the initial frame.
    frame = cv2.imread(os.path.join("./heatmap/frames/", images[0]))
    height, width, layers = frame.shape if frame is not None else (0, 0, 0)

    # Define the output video codec to mp4. Instantiate a VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video  = cv2.VideoWriter("./heatmap/output.mp4", fourcc, 30.0, (width, height))

    # Add each frame in the ``heatmap/frames`` directory to the output video file.
    for image in images:
        video.write(cv2.imread(os.path.join("./heatmap/frames/", image)))

    # Remove all individual frames once the final video is completed to avoid needless storage.
    for file in os.listdir("./heatmap/frames/"):
        os.remove("./heatmap/frames/" + file)

    # Display the time taken to complete the heatmap process.
    print(f"...Completed HeatMapping Process. ({time.time() - t0:.3f}s)")

    capture.release()
    cv2.destroyAllWindows()


def return_digits_from_input(input_string):
    """
    Helper function which returns the digits from within the passed input string.

    :param input_string: String from which digits are to be extracted.
    :return:             Integer retrieved from the input string.
    """
    return int(input_string) if input_string.isdigit() else input_string


def natural_keys(input_string):
    """
    Helper function which generates keys which will be used to sort frames in natural order.

    :param input_string: String to be used to generate keys.
    :return:             Keys to be used to sort frames in their natural order.
    """
    return [return_digits_from_input(c) for c in re.split(r'(\d+)', input_string)]
