import cv2
import math
import pathlib
import torch

from coordinate_system          import *
from heatmap_generator          import *
from merge_tracking_and_heatmap import *
from models.experimental        import attempt_load
from shapely.geometry           import LineString
from sift_functionality         import *
from sort                       import *
from utils.datasets             import LoadImages
from utils.general              import apply_classifier, check_img_size, increment_path, non_max_suppression, \
                                       scale_coords, set_logging
from utils.torch_utils          import load_classifier, time_synchronized, TracedModel, select_device


object_speed_dictionary  = {}  # Dictionary to store the speed of each tracked object.
object_trajectory_list   = []  # List to store the trajectory of each tracked object.
object_boundary_box_list = []  # List to store the boundary boxes of each tracked object.


# Populate the object speed dictionary. It is assumed that there will not be more than 10000 tracked objects.
for i in range(0, 10000):
    object_speed_dictionary[i] = 0


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    """
    Function to draw encapsulating boundary boxes over a tracked object.

    :param img:        Image upon which the boundary boxes will be imposed.
    :param bbox:       Array of boundary box objects.
    :param identities: Array containing the ID of every tracked object.
    :param categories: Array containing the name of each possible object type, i.e. car, cyclist & pedestrian.
    :param names:      Array which contains the name corresponding to each ID.
    :param offset:     Set to (0, 0), i.e there is no offset.
    :return:           Return the passed image with imposed boundary boxes.
    """
    # Iterate through each boundary box.
    for index, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(index) for index in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Determine the ID & category of the current object.
        object_id = int(identities[index]) if identities is not None else 0
        cat = int(categories[index]) if categories is not None else 0

        # Set the label of the current object.
        label = str(object_id) + ":" + names[cat]

        # Depending on the category of the current object, set the colour of the boundary box and label accordingly.
        if names[cat] == "car":
            main_colour = (255, 0, 0)
            label_colour = (255, 0, 20)
        elif names[cat] == "pedestrian":
            main_colour = (0, 0, 255)
            label_colour = (20, 0, 255)
        else:
            main_colour = (0, 255, 0)
            label_colour = (20, 255, 20)

        # Determine width and height of label.
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw boundary and text boxes.
        cv2.rectangle(img, (x1, y1), (x2, y2), label_colour, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), main_colour, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

        # Add current the boundary box to the list of all boundary boxes.
        object_boundary_box_list.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    return img


def calculate_speed(x2, y2, x1, y1):
    """
    Function to calculate the speed of a particular tracked object.

    :param x2: X co-ordinate of the object's current centroid.
    :param y2: Y co-ordinate of the object's current centroid.
    :param x1: X co-ordinate of the object's previous centroid.
    :param y1: Y co-ordinate of the object's previous centroid.

    :return: Speed of a tracked object in kilometres per hour.
    """
    """
    Pixels based off of the following:
        1. Width of a car park space is 2.8m.
        2. Width of the frame is approximately 120 metres.
        3. Width of the frame is 3840 pixels.
        4. Pixels per metre is 32.
    """
    pixels_per_metre = 32

    pixel_distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    metre_distance = pixel_distance / pixels_per_metre

    # Multiply by 3.6 to convert metres per second to kilometers per hour.
    speed = metre_distance * 3.6

    return speed


def draw_trajectory_line(img, speed, current_x, current_y, previous_x, previous_y):
    """
    Function to draw extending trajectory line off of a tracked object

    :param img:         Current frame
    :param speed:       Speed at which the  current object is moving at
    :param current_x:   X location of the object's current centroid
    :param current_y:   Y location of the object's current centroid
    :param previous_x:  X location of the object's previous centroid
    :param previous_y:  Y location of the object's previous centroid
    """
    if speed > 0 and (current_x - previous_x) != 0 and (current_y - previous_y) != 0:
        """
        The length of an objects trajectory is equal to its speed in kilometers per hour divided by 3.6.
        From this, the distance in pixels may be derived.
        """
        length_of_trajectory = speed / 3.6

        # The angle which an object is travelling may be found by calculating the arc tangent of its slope.
        dy                = current_y - previous_y
        dx                = current_x - previous_x
        angle_of_movement = math.atan2(dy, dx)

        """
        The start point of an object's trajectory is the coordinate of it's current centroid.
        The end point of an object's trajectory line is the calculated based on the direction and speed which the object
        is travelling at. This is based off of the formula shown on this webpage:
        https://www.omnicalculator.com/math/right-triangle-side-angle
        """
        start_point_x = int(current_x)
        start_point_y = int(current_y)
        end_point_x   = int(current_x + (length_of_trajectory * math.degrees(math.cos(angle_of_movement))))
        end_point_y   = int(current_y + (length_of_trajectory * math.degrees(math.sin(angle_of_movement))))

        # Draw the object trajectory line. Append this line to the object trajectory list.
        cv2.line(img, (start_point_x, start_point_y), (end_point_x, end_point_y), (242, 0, 137), 2)
        object_trajectory_list.append([[start_point_x, start_point_y], [end_point_x, end_point_y]])


def detect():
    """
    Function to iterate through each frame in the provided video footage.
    Object tracking, speed calculation, trajectory derivation and collision prediction functions are invoked upon each
    iteration.
    This function is largely based upon that found within YOLOv7's ``detect.py`` file.

    :return: None
    """
    # Counter which increments upon each iteration.
    frame_count = 0

    # Assign each value parsed from input.
    source   = opt.source
    weights  = opt.weights
    save_txt = opt.save_txt
    imgsz    = opt.img_size
    trace    = not opt.no_trace
    save_img = not opt.nosave and not source.endswith(".txt")

    # Initialise SORT tracker object.
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

    # Define the directory which results will be saved.
    save_dir = pathlib.Path(increment_path(pathlib.Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialization of device to perform detection and tracking.
    set_logging()
    device = select_device(opt.device)
    half   = device.type != "cpu"

    # Load FP32 stride model.
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())

    # Check the specified image size.
    imgsz = check_img_size(imgsz, s=stride)

    # If the trace flag is asserted, instantiate a TracedModel.
    if trace:
        model = TracedModel(model, device, opt.img_size)

    # If the half flag is asserted, use an FP16 model rather than an FP32 model.
    if half:
        model.half()

    # Instantiate a second-stage classifier if the classify flag is asserted.
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

    # Set the dataloader to be used.
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names of model modules.
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference.
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Get the time at which the detection and tracking process begins.
    t0 = time.time()

    # Iterate through each element in the loaded dataset.
    for path, img, im0s, vid_cap in dataset:
        # Convert unit8 to FP16/32, with a range of 0 to 255.
        img  = torch.from_numpy(img).to(device)
        img  = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup.
        if device.type != 'cpu':
            if old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]:
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]

        # Inference time begins.
        t1 = time_synchronized()

        # If gradients were calculated, a GPU leak would occur.
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]

        t2 = time_synchronized()

        # Apply NMS.
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply classifier.
        if classify:
            pred = apply_classifier(pred, None, img, im0s)

        # Iterate through each detection within the current image.
        for index, det in enumerate(pred):
            # Reset the list of trajectories and boundary boxes for each frame.
            object_trajectory_list.clear()
            object_boundary_box_list.clear()

            global save_path
            p, s, im0, current_frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = pathlib.Path(p)
            save_path = str(save_dir / p.name)

            if len(det):
                # Rescale boxes from img_size to im0 size.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print the number of detections recorded for each class.
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Pass an empty array to sort.
                dets_to_sort = numpy.empty((0, 6))

                # Add each detection to be sorted to a stack.
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = numpy.vstack((dets_to_sort, numpy.array([x1, y1, x2, y2, conf, detclass])))

                # Run SORT algorithm.
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                # If there is at least one tracked object, invoke the draw boxes function.
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names)

                """
                Iterate through each track. 
                The speed of each tracked object is calculated, and the function to calculate the appropriate trajectory
                is invoked.
                The loop subsequently determines whether or not this trajectory collides with any other boundary boxes 
                or trajectories. 
                Finally, the function to determine the coordinates of each object is also called.
                """
                for track in tracks:
                    # Every second (vertex.e. every 30 frames), re-perform calculations.
                    if frame_count % 30 == 0:
                        # If an object has been tracked more than once we may calculate it's current speed.
                        if len(track.centroidarr) > 30:
                            current_x  = int(track.centroidarr[-1][0])   # Get the X location of the current centroid.
                            current_y  = int(track.centroidarr[-1][1])   # Get the Y location of the current centroid.
                            previous_x = int(track.centroidarr[-30][0])  # Get the X location of the previous centroid.
                            previous_y = int(track.centroidarr[-30][1])  # Get the Y location of the previous centroid.

                            """
                            Calculate the speed of the current object by invoking the calculate_speed function.
                            Append the calculated speed to the object's entry in the object_speed_dictionary.
                            """
                            object_speed_dictionary[track.id] = int(calculate_speed(current_x, current_y,
                                                                                    previous_x, previous_y))

                    # Overlay the current speed of an object upon the current frame.
                    cv2.putText(im0, str(object_speed_dictionary[track.id]) + "km/h",
                                (int(track.centroidarr[-1][0] - 30), int(track.centroidarr[-1][1] - 30)),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.6, [255, 255, 255])

                    """
                    If the location of the video footage has been found (and its coordinates have been identified), 
                    then valid_location would have been asserted. This assertion is completed by the SIFT function 
                    defined in ``sift_functionality.py``.
                    If valid location is true, calculate the coordinates of each tracked object.
                    """
                    if valid_location is True:
                        # Determine the Decimal Degree location of an object using its pixel coordinates.
                        object_dd = bilinear_interpolation(int(track.centroidarr[-1][0]),
                                                           int(track.centroidarr[-1][1]),
                                                           corners)

                        # Convert the object's Decimal Degree location to Degrees Minutes Seconds format.
                        object_dms = dd_to_dms_formatter(object_dd[0], object_dd[1])

                        # Superimpose the coordinates of each object onto the current frame.
                        cv2.putText(im0, object_dms[0],
                                    (int(track.centroidarr[-1][0] - 80), int(track.centroidarr[-1][1])),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, [255, 255, 255])
                        cv2.putText(im0, object_dms[1],
                                    (int(track.centroidarr[-1][0] - 80), int(track.centroidarr[-1][1]) + 30),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, [255, 255, 255])

                    """
                    Provided an object has been tracked for at least one frame, draw it's trajectory onto the current
                    frame
                    """
                    if len(track.centroidarr) > 30:
                        draw_trajectory_line(im0,
                                             object_speed_dictionary[track.id],
                                             int(track.centroidarr[-1][0]),
                                             int(track.centroidarr[-1][1]),
                                             int(track.centroidarr[-30][0]),
                                             int(track.centroidarr[-30][1]))

                """
                Check if two or more trajectories intersect. 
                If an intersection occurs, then highlight the point of intersection using a superimposed turquoise 
                circle.
                """
                for cl in object_trajectory_list:
                    for line in object_trajectory_list:
                        if cl != line:
                            line1 = LineString(cl)
                            line2 = LineString(line)
                            if line1.intersects(line2):
                                intersection = line1.intersection(line2)
                                cv2.circle(im0, (int(intersection.x), int(intersection.y)), 10, (255, 255, 0),
                                           thickness=-1)

                """
                Check if a trajectory intersects with a boundary box.
                If an intersection occurs, in this case, highlight the point of intersection using a superimposed
                yellow circle.
                """
                for cl in object_trajectory_list:
                    for box in object_boundary_box_list:
                        # Need to check each vertex of the current boundary box for a potential intersection.
                        for vertex in range(4):
                            line1 = LineString(cl)
                            # Check both sides of the boundary box vertex.e. a line could hypothetically cross a box.
                            if vertex == 3:
                                line2 = LineString((box[vertex], box[0]))
                                if line1.intersects(line2):
                                    intersection = line1.intersection(line2)
                                    boundary_box_corners = numpy.array(box)
                                    outcome = cv2.pointPolygonTest(boundary_box_corners,
                                                                   (line1.coords[0][0], line1.coords[0][1]),
                                                                   measureDist=False)
                                    """
                                    If the outcome of the pointPolygonTest is equal to -1, then it means that the 
                                    trajectory line does not originate from within the current boundary box.
                                    This is an important check, as otherwise each trajectory would immediately
                                    intersect with its originating object's boundary box.
                                    """
                                    if outcome == -1:
                                        cv2.circle(im0, (int(intersection.x), int(intersection.y)), 10, (0, 255, 255),
                                                   thickness=-1)
                            else:
                                line2 = LineString((box[vertex], box[vertex + 1]))
                                if line1.intersects(line2):
                                    intersection = line1.intersection(line2)
                                    boundary_box_corners = numpy.array(box)
                                    outcome = cv2.pointPolygonTest(boundary_box_corners,
                                                                   (line1.coords[0][0], line1.coords[0][1]),
                                                                   measureDist=False)
                                    if outcome == -1:
                                        cv2.circle(im0, (int(intersection.x), int(intersection.y)), 10, (0, 255, 255),
                                                   thickness=-1)

                # Increment the current frame counter.
                frame_count = frame_count + 1

            # Print time (inference + NMS).
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            """
            If save_img is asserted, save the resulting images with detections.
            Otherwise, save the resulting images to a video.
            """
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer.write(im0)

    # Display the time taken to complete the object detection and collision prediction process.
    print(f"...Completed Object Detection & Collision Prediction. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      nargs="+",                 type=str, default="yolov7.pt",
                        help="model.pt path(s)")
    parser.add_argument("--no-download",  dest="download",           action="store_false")
    parser.add_argument("--source",       type=str,                  default="inference/images",
                        help="source")
    parser.add_argument("--img-size",     type=int,                  default=640,
                        help="inference size (pixels)")
    parser.add_argument("--conf-thres",   type=float,                default=0.25,
                        help="object confidence threshold")
    parser.add_argument("--iou-thres",    type=float,                default=0.45,
                        help="IOU threshold for NMS")
    parser.add_argument("--device",       default="",                help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img",     action="store_true",       help="display results")
    parser.add_argument("--save-txt",     action="store_true",       help="save results to *.txt")
    parser.add_argument("--save-conf",    action="store_true",       help="save confidences in --save-txt labels")
    parser.add_argument("--nosave",       action="store_true",       help="do not save images/videos")
    parser.add_argument("--classes",      nargs="+", type=int,
                        help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true",       help="class-agnostic NMS")
    parser.add_argument("--augment",      action="store_true",       help="augmented inference")
    parser.add_argument("--update",       action="store_true",       help="update all models")
    parser.add_argument("--project",      default="runs/detect",     help="save results to project/name")
    parser.add_argument("--name",         default="object_tracking", help="save results to project/name")
    parser.add_argument("--exist-ok",     action="store_true",       help="existing project/name ok, do not increment")
    parser.add_argument("--no-trace",     action="store_true",       help="don`t trace model")
    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        # Check that location is valid.
        global result
        valid_location = perform_sift_location_check(opt.source)

        if valid_location is True:
            """
            Define the latitude and longitude boundary coordinates of the specified video footage in Degrees Minutes 
            Seconds format.
            Coordinates are hardcoded from Google Maps for this specific application.
            """
            LATITUDE_TOP_LEFT     = dms_to_dd(53, 17, 22, 'N')
            LATITUDE_BOTTOM_LEFT  = dms_to_dd(53, 17, 22, 'N')
            LATITUDE_TOP_RIGHT    = dms_to_dd(53, 17, 26, 'N')
            LATITUDE_BOTTOM_RIGHT = dms_to_dd(53, 17, 26, 'N')

            LONGITUDE_TOP_LEFT     = dms_to_dd(9, 4, 19, 'W')
            LONGITUDE_BOTTOM_LEFT  = dms_to_dd(9, 4, 16, 'W')
            LONGITUDE_TOP_RIGHT    = dms_to_dd(9, 4, 17, 'W')
            LONGITUDE_BOTTOM_RIGHT = dms_to_dd(9, 4, 14, 'W')

            corners = [(0, 0, LATITUDE_TOP_LEFT, LONGITUDE_TOP_LEFT),
                       (0, 2159, LATITUDE_TOP_RIGHT, LONGITUDE_TOP_RIGHT),
                       (3839, 3839, LATITUDE_BOTTOM_RIGHT, LONGITUDE_BOTTOM_RIGHT),
                       (3839, 0, LATITUDE_BOTTOM_LEFT, LONGITUDE_BOTTOM_LEFT)]

        # Invoke the ``detect`` function to perform object tracking, speed calculations and collision prediction.
        detect()

        # Invoke the ``generate_heat_map`` function to superimpose a heatmap onto the provided footage.
        generate_heat_map(opt.source)

        # Merge both the heatmap and tracking video footage.
        save_path = "./" + save_path.replace("\\", "/")
        merge_tracking_and_heatmap_footage(save_path, "./heatmap/output.mp4")
