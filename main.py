# import the necessary packages 
from cnn.networks.lenet import LeNet
from comp_vision.comp_vision import CompVision
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not a pre-trained mdoel should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
ap.add_argument("-c", "--camera-feed", type=int, default=-1,
	help="(optional) flag to identify images on camera stream")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# reshape the MNSIT dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLables) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=0.33)


# transform the training and testing labels into vectors in the 
# range [0, classes] -- this generates a vector for each label,
# where the indx of the label is set to '1' and all other entries
# to '0'; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLables = np_utils.to_categorical(testLables, 10)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, nb_epoch=10,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLables,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

if args["camera_feed"] < 0:
	# randomly select a few testing digits
	for i in np.random.choice(np.arange(0, len(testLables)), size=(10,)):
		# classify the digit
		probs = model.predict(testData[np.newaxis, i])
		print(len(testData[np.newaxis, i][0][0]))
		prediction = probs.argmax(axis=1)

		# resize the image from a 28 x 28 image a to 96 x 96 image so
		# we can better see it
		image = (testData[i][0] * 255).astype("uint8")
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, str(prediction[0]), (5,20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		# show the iimage and prediction
		print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
			np.argmax(testLables[i])))
		cv2.imshow("Digit", image)
		cv2.waitKey(0)


if args["camera_feed"] > 0:
	c = CompVision()
	while(True):
		im, bin_thresh= c.get_frame()
		contours = c.get_contours()
		digit_images = c.format_contours(contours)
		cont_with_labels = []

		for i in xrange(0, len(digit_images)):
			input_digit = np.array([np.array([digit_images[i]])])
			probs = model.predict(input_digit)
			prediction = probs.argmax(axis=1)
			#print("[INFO] Predicted: {}".format(prediction[0]))
			cont_with_labels.append((contours[i], prediction))

		im = c.label_contours_on_image(im, cont_with_labels)

		

		'''image = (digit_images[0] * 255).astype("uint8")
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)'''
		cv2.imshow('frame', im)
		k = cv2.waitKey(10)
		if k != -1:
			if k & 0xFF == ord('q'):
				break



	# When everything is done, release the capture
	#cap.release()
	cv2.destroyAllWindows()

