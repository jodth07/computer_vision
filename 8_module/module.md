Option #2: Face Detection and Privacy
To address privacy concerns you may want to use data anonymization.  On images, this can be achieved by hiding features that could lead to a person or personal data identification, such as the person’s facial features or a license plate number.

The goal of this project is to write algorithms for face detection and feature blurring.  Select three color images from the web that meet the following requirements:

Two images containing human subjects facing primarily to the front and one image with a non-human subject.
At least one image of a human subject should contain that person’s entire body.
At least one image should contain multiple human subjects.
At least one image should display a person’s face far away.
All images should vary in light illumination and color intensity. 
First, using the appropriate trained cascade classifierLinks to an external site., write one algorithm to detect the human faces in the gray scaled versions of the original images.  Put a red boundary box around the detected face in the image in order to see what region the classifier deemed as a human face. If expected results are not achieved on the unprocessed images, apply processing steps before implementing the classifier for optimal results.

After the faces have been successfully detected, you will want to process only the extracted faces before detecting and applying blurring to hide the eyes. Although the eye classifierLinks to an external site. is fairly accurate, it is important that all faces are centered, rotated, and scaled so that the eyes are perfectly aligned. If expected results are not achieved, implement more image processing for optimal eye recognition. Now, apply a blurring method to blur the eyes out in the extracted image.

Inspect your results and write a summary describing the techniques you used to detect and blur the eyes out of human faces in images. Reflect on the challenges you faced and how you overcame these challenges.  Furthermore, discuss in your summary, the accuracy of your results for all three images and techniques you used to improve the accuracy after each repeated experiment.