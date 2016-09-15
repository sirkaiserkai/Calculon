import numpy as np
import cv2

# source: http://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than 
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 0

	# construct a list of bounding boxes and sort them form top to
	# bottom
	boudingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boudingBoxes) = zip(*sorted(zip(cnts, boudingBoxes), key=lambda b:b[1][i], reverse=reverse))

	# returns a list of sorted contours and bounding boxes
	return (cnts, boudingBoxes)


class CompVision:
	

	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.thresh = None
		self.bin_thresh = None
		self.font = cv2.FONT_HERSHEY_SIMPLEX

	def get_frame(self):
		# Capture frame by frame
		ret, frame = self.cap.read()

		# Our operations on the frame come here
		image_dimensions = (960, 540)
		im = cv2.resize(frame, image_dimensions)
		#gray_image = 255-im # inverts the colors 
		gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		gray_image = 255-gray_image # inverts the colors 
		
		ret, bin_thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		thresh = bin_thresh
		#thresh = cv2.GaussianBlur(bin_thresh, (5,5), 0)
		#ret, thresh = cv2.threshold(g_blurred_image, 127, 255, 0)
		dislation_count = 10
		thresh = cv2.dilate(thresh,None,iterations = 2)
		thresh = cv2.erode(thresh,None,iterations = 0)

		self.bin_thresh = bin_thresh
		self.thresh = thresh
		return (im, bin_thresh)

	def get_contours(self):
		contours, hiearchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		contours, _ = sort_contours(contours)
		digit_contours = []
		for cont in contours:
			x, y, w, h = cv2.boundingRect(cont)
			arc_length = cv2.arcLength(cont, True)
			if arc_length > 100.0 and arc_length < 960:
				digit_contours.append(cont)
				#cv2.rectangle(im, (x,y), (x+w, y+h), (255, 0, 0), 2)
				#cv2.putText(im,str(count),(x,y), font, 1,(255,255,255),2)

		return digit_contours

	def format_contours(self, contours):
		formatted_contours = []
		for cont in contours:
			x, y, w, h = cv2.boundingRect(cont)
			digit_image = self.bin_thresh[y:y+h,x:x+w]
			digit_image = cv2.resize(digit_image, (20, 20))
			expanded_image = np.zeros((28,28), np.uint8)
			expanded_image[3:23,3:23] = digit_image	

			'''for i in xrange(0,len(expanded_image)):
				for j in xrange(0, len(expanded_image[i])):
					expanded_image[i][j] = float(expanded_image[i][j]) / 255.0'''

			formatted_contours.append(expanded_image)

		return formatted_contours

	def label_contours_on_image(self, im, contours_labled):
		count = 0

		for cl in contours_labled:
			cont, label = cl
			x, y, w, h = cv2.boundingRect(cont)
			arc_length = cv2.arcLength(cont, True)
			if arc_length > 30.0:
				cv2.rectangle(im, (x,y), (x+w, y+h), (255, 0, 0), 2)
				cv2.putText(im,str(label[0]),(x,y), self.font, 1,(255,255,255),2)
				count += 1

		return im













