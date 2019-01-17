#!/usr/bin/env python 

# This program was written by: Abdulrhman Saad & Mohanned Ahmed
# Refernce: https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

# USAGE

# Start ROS:
# roscore
 
# Open camera:
# rosrun usb_cam usb_cam_node _pixel_format:=yuyv

# Launch turtlebot2:
# roslaunch turtlebot_bringup minimal.launch

# Run soundplay node:
# rosrun sound_play soundplay_node.py

# Run surveillance mode:
# python recognize_faces_video_ros_v3c.py --encodings encodings.pickle --display 1 --detection-method hog --mode FRS

# OR

# Run following mode:
# python recognize_faces_video_ros_v3c.py --encodings encodings.pickle --display 1 --detection-method hog --mode FRF


# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import datetime 
import math
import csv
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

# To use ROS image.
bridge = CvBridge()
cv_image = None
# To store encodings.
data = []

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-m", "--mode", type=str, default="FRS",
	help="face detection mode to use: either `FRS` or `FRF`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")

def image_callback(ros_image):
  global bridge, cv_image
  #convert ros_image into an opencv-compatible image
  try:
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
  except CvBridgeError as e:
      print(e)

rospy.init_node('face_image', anonymous=True)
#for turtlebot3 waffle
#image_topic="/camera/rgb/image_raw"
#for usb cam
image_topic="/usb_cam/image_raw"
image_sub = rospy.Subscriber(image_topic,Image, image_callback)
soundhandle = SoundClient()
time.sleep(2)

last_name = "" # Last name detected by camera.
first_name = "" # First name detected by camera.
is_box_not_in_center = True
# Set CSV file. 
writeFile = open('data.csv', 'a')
writer = csv.writer(writeFile)
# To store data
camera_data = []
order_counter = 0
# Set Velocity Publisher.
cmd_vel_topic='/cmd_vel_mux/input/teleop'
velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = cv_image
	
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	names_accuracy = [] 

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		name_accuracy = 0.0
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name_max = max(counts, key=counts.get)
			#print(counts)
			counts_sum = sum(counts.itervalues())
			name_accuracy = float(counts[name_max])/counts_sum * 100.0
			if name_accuracy > 40.0: # Train More!!!!!
				name = name_max
			else: 
				name = "Unknown"
		
		# update the list of names
		names.append(name)
		names_accuracy.append(name_accuracy)	

	# loop over the recognized faces
	for ((top, right, bottom, left), name, name_accuracy) in zip(boxes, names, names_accuracy):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		color = (0, 255, 0) # Green
		if name == "Unknown":
			color = (0, 0, 255) # Red 
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			color, 2)
		y = top - 15 if top - 15 > 15 else top + 15
		
		cv2.putText(frame, (name + ": " + str("%.2f" % name_accuracy) + "%"), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, color, 2)
	# Store and print the data 
	if len(names) >= 1:
		if last_name != names[len(names)-1]:
			date_time = datetime.datetime.now()	
			cv2.imwrite("images/" + str(date_time) + ".png",frame)
			row = [(order_counter+1),"-".join(names), date_time, "images/" + str(date_time) + ".png"]
			camera_data.append(row)
			print("\nStoring \nImage: " + str(order_counter+1) + "\nName(s): " + ",".join(names) +"\n")
			print("<--------------->\n")	
			last_name = names[len(names)-1]
			order_counter += 1


	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break	
	# If following mode is used.
	if args["mode"] == 'FRF' and len(boxes) > 0:   
		first_box = boxes[0] 
		top, right, bottom, left = first_box
		box_center = (int(right * r) + int(left * r))/2 
		frame_center = frame.shape[1]/2
		# Apply sound.
		if first_name != names[0] and not is_box_not_in_center:
			#soundhandle.say("Following " + names[0])
			#time.sleep(2.0) 
			first_name = names[0]

		velocity_message = Twist()
		velocity_message.linear.x=0
		velocity_message.linear.y=0
		velocity_message.linear.z=0
		velocity_message.angular.x=0
		velocity_message.angular.y=0
		velocity_message.angular.z=0

		frame_width = frame.shape[1] + 0.1
		speed_angular = -(((box_center-frame_center)/frame_width)*7.0)
		t0 = rospy.Time.now().to_sec()
		total_time = 0
		while total_time < 5.5 and not is_box_not_in_center:
			#print(">>>>> Linear")
			velocity_message.linear.x = 0.2
			velocity_publisher.publish(velocity_message)
			t1 = rospy.Time.now().to_sec()
			total_time = t1-t0
			if not (frame_center - 50 < box_center < frame_center + 50):
				is_box_not_in_center = True
		while (not (frame_center - 5 < box_center < frame_center + 5)) and total_time < 0.5 and is_box_not_in_center:
			#print(">>>>> Angular")
			velocity_message.angular.z = speed_angular
			velocity_publisher.publish(velocity_message)
			t1 = rospy.Time.now().to_sec()
			total_time = t1-t0
			if (frame_center - 50 < box_center < frame_center + 50):
				is_box_not_in_center = False
		velocity_message.angular.z = 0
		velocity_publisher.publish(velocity_message)

 				
# do a bit of cleanup
cv2.destroyAllWindows()
# Save camera data to CSV file.  
writer.writerows(camera_data)
writeFile.close()
