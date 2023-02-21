# ECE Final Year Project 2023: Vehicle Detection and Collision Prediction

<p align="center">
  <img width="600" alt="gif" src="https://user-images.githubusercontent.com/125507627/219214471-77abf121-0086-44e4-935b-875e9383a8c3.gif">
</p>

This final year project involves the research and development of an application that can detect vehicles, cyclists and pedestrians in real-time using YOLOv7, an advanced object detection algorithm, and OpenCV, a computer vision library. In addition to detecting and tracking vehicles, the system can predict potential collisions between the detected objects based on their alculated speeds, positions and movements. 

The application incorporates a motion heatmap via Intel's Motion Heatmap technology, and Degree Minute Second (DMS) coordinate mapping using a Scale-Invariant Feature Transform (SIFT) to further track objects.

The project is implemented in Python and can operate in real-time, making it potentially suitable for several applications, including traffic monitoring and surveillance. These applications are particularly relevant due to the advancement of 5G networks within the autonomous vehicle sphere. 

<br>

<details>
  <summary>Training a YOLOv7 Model using a Custom Dataset</summary>
  <h2>Training a YOLOv7 Model using a Custom Dataset</h2>
  For this project, a custom dataset was developed to train a You Only Look Once (YOLO) model. The dataset comprised 500 annotated training images, as well as a further 100 validation images. Upon training the model, a mAP@0.5 score of 0.956 was recorded, indicating a high level of accurac.
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219227360-294292f5-ffdf-4bfe-8307-4dc71af5c6f9.png">
  </p>
  
  The YOLOv7 model operates as follows:
  <ul>
    <li>Firstly, the algorithm pre-processes input images by resizing them and normalizing pixel values.</li>
    <li>Pre-processed images are subsequently passed through several convolutional and max pooling layers to extract features.</li>
    <li>Then, the algorithm predicts bounding boxes and class probabilities for each cell in the grid using a fully connected layer.</li>
    <li>Finally, the algorithm applies non-maximum suppression to remove overlapping bounding boxes and returns the remaining boxes as the final object detections.         </li>
  </ul>
  
  <br>
</details>


<details>
  <summary>Object Tracking and Collision Prediction</summary>
  <h2>Object Tracking and Collision Prediction</h2>
  A Simple Online and Realtime Tracking (SORT) algorithm is used in this application to track objects and calculate their speed and trajectory. A Kalman filter is used to improve the estimate the location of each detected object in the current frame, based on its previous locations and motion. 
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219225819-c8ea5921-aefa-4934-9cb0-0baa99545b33.png">
  </p>
  
  The number of pixels per metre is subsequently determined to calculate the speed of each object. Once tracked, each object's speed is kept in a dictionary with its ID as the key. 
  
  The Collision Detection Algorithm utilises the predicted trajectory of tracked objects to anticipate collisions. The trajectory is determined by the
angle of an object's movement and the speed of the object. The endpoint of the trajectory can be found using trigonometric laws. 
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219227635-063b1e19-5b68-42a7-b665-1b1bf2a03fcf.png">
  </p>
  
  <br>
</details>

<details>
  <summary>Heatmap Generation</summary>
  <h2>Heatmap Generation</h2>
  Intel's Motion Heatmap technology describes a specific implementation of motion detection and heatmap generation in OpenCV. It involves the utilisation of OpenCV's   background substraction function to separate the foreground and background of a video. Background subtraction works by comparing the current frame of a video with a   background model to detect the pixels that have changed due to motion.
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219228392-f20c2557-a47a-4d39-9823-4fe09441ea82.png">
  </p>
  
  Once motion has been detected, NumPy is utilised to create a 2D histogram of the motion pixels, with MatPlotLib being used to display the resulting heatmap.
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219226636-46a3a8f4-e693-43bc-ad5a-7620556b6544.jpg">
  </p>
  
  <br>
</details>

<details>
  <summary>Coordinate Mapping</summary>
  <h2>Coordinate Mapping</h2>
  SIFT is a feature detection algorithm used to identify and describe local features in images, which can be used for tasks like image recognition, object detection, and matching. Implementing SIFT in OpenCV involves detecting keypoints, extracting descriptors, matching keypoints, filtering matches, and visualizing the matches. 
  <br>
  <br>
  In this application, the first frame of the passed video footage is to be compared to several satellite images using a SIFT algorithm. If a match is detected (i.e. if an image is of the same location as the location where the video footage was filmed), then the geographical meta data of the comparative image will be used to map the coordinates of each tracked object in provided video footage.
  <br>
  <br>
  Below is an example of a successful match, which would result in the geographical meta data of the comparative image (on the right) being utilised to map the           coordinates of each tracked object in the provided video footage.
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219226555-b1aecf80-e677-4e11-9eb3-978749617f92.jpg">
  </p>
  
  The following image shows an unsuccessful comparison. The images are not of the same location, therefore no cooordinate mapping will occur.
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219226700-1023c629-3931-4c43-a477-e0594e9be20c.jpg">
  </p>

  Coordinate mapping appears as follows.
  
  <p align="center">
    <br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/125507627/219228009-5be494f4-7c09-47ac-8790-f8a104f3fd79.png">
  </p>
  
  <br>
</details>

<br>

<h2>Referenced Repositories</h2>
<ul>
  <li>YOLOv7: https://github.com/WongKinYiu/yolov7</li>
  <li>YOLOv7 Object Tracking Using PyTorch, OpenCV and Sort Tracking: https://github.com/RizwanMunawar/yolov7-object-tracking</li>
  <li>Motion Heatmap OpenCV: https://github.com/robertosannazzaro/motion-heatmap-opencv</li>
</ul>
