import cv2
import time


def merge_tracking_and_heatmap_footage(tracking_footage_path, heatmap_footage_path):
    """
    Function which merges the tracking video footage with the heatmap video footage

    :param tracking_footage_path: Path to the tracking video footage
    :param heatmap_footage_path:  Path to the heatmap video footage
    """
    print("Beginning to Merge Footage...")

    # Get the time at which the merging process begins.
    t0 = time.time()

    """
    Instantiate two VideoCapture objects:
        1. The first object is used to read frames from the heatmap video.
        2. The second object is used to read frames from the boundary box footage.
    Define ```final_render.mp4`` as the file to store the resulting merged footage.
    """
    cap1 = cv2.VideoCapture(heatmap_footage_path)
    cap2 = cv2.VideoCapture(tracking_footage_path)
    out  = cv2.VideoWriter('final_render.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840, 2160))

    while True:
        # Read frames from the first video, so long as there are frames available.
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        # Read frames from the second video, so long as there are frames available.
        ret2, frame2 = cap2.read()
        if not ret2:
            break

        """
        Use OpenCVs addWeighted function to merge the frames together.
        A ratio of 0.4 to 0.8 is used to minimise the vibrancy of the heatmap.
        """
        blended_frame = cv2.addWeighted(frame1, 0.4, frame2, 0.8, 0)
        out.write(blended_frame)

    # Display the time taken to complete the merging process.
    print(f"...Completed Merging Process. ({time.time() - t0:.3f}s)")

    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
