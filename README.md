[//]: # (Image References)

[image1]: ./output_images/header.png "Header"
[image2]: ./output_images/color_hist.png
[image3]: ./output_images/comb_hist_norm.png
[image4]: ./output_images/final.png
[image5]: ./output_images/heat_map1.png
[image6]: ./output_images/heat_map.png
[image7]: ./output_images/hog_vis.png
[image8]: ./output_images/spatial_hist.png
[image9]: ./output_images/svc_data.png

![alt text][image1]
## **Overview**

The purpose of this project is to use computer vision techniques to locate vehicles in an image.  Techniques include Color Histogram Template Matching, Spatial Binning and Histogram of Oriented Gradients(HOG).  Data is used to train a Support Vector Machine Classifier(SVC).  The classifier is used in a software pipeline to locate and draw bounding boxes are drawn around vehicles original image.  

Link to [project code](https://github.com/hbutler97/CarND-Vehicle-Detection/blob/master/find_vehicle.ipynb)

Link to [Result YouTube Video](https://youtu.be/3SCqGUhmyxk)



## **Feature Extraction for SVC Training**

Prior to training the classifier, serveral types features are extracted out the image to locate the object of interest(vehicles).  Differentiating features include color and shape. 

### **Color Histogram**

A histogram of the color channels are extracted from the image and concatenated to a single vector.  This is done in a function called color_hist().  Results are shown below.

![alt text][image2]

### **Spatial Binning**

To generalize the features a bit more the image resolution is reduced such that relevant features are still recognizable, but general enough to be useful to locate vehicles. This is done in a function called bin_spatial() Results are shown below.

![alt text][image8]

### **Histogram of Oriented Gradients**

Shape of the car is probably the easiest feature to recognize.  Histogram of Oriented Gradients(HOG), was used to extract information on the car's shape. This is done in a function called get_hog_features() Results are shown below.

![alt text][image7]

### **Feature Concatenation and Normalization**

The features mentioned above were combined(concatenated) and normalized prior to be used for SVC Training.  This is done in a function called extract_features().  Results are shown below.

![alt text][image3]

## **Support Vector Machine Classifier Training**

A Support Vector Machine Classifier was used to determine if features presented was an actual vehicle.  Multiple configurations for feature extraction was attempted to achieve reasonable training accuracy and ultimately good vehicle detection in the video stream.

The table below shows some notable configurations that yielded reasonable performance.  Ultimately Configuration 5 performed the best on the video stream.  In addition to the configurations below, the percentage of the data set used for training was also changed from the typical 80% to 15%.  This was due to the training data being from a video stream and many of the images were the same.  It turned out that this didn't make a big difference in my case and ultimately ended up using 80% of the training set.

A cell called Feature Extraction Globals has the configurations listed below.  Extract Features and Scale Results cell, does as titled. Lastly, there is a cell call Train Classifier. 

![alt text][image9]


## **Software Pipeline**

The software pipeline used to detect the Vehicles.  High Level functions preformed by the pipeline are as follows:

* Image Patch Scan(Sliding Window)

  -Image is scanned using a sliding window(details below) and for each patch the following operations are preformed

* HOG Feature Extraction
* Spatial Binning Extraction
* Color Histogram Extraction
* Features Concatenation
* Features Normalization
* SVC Prediction
* Bounding Box Drawing(for positive car detection)
* False Positive Filtering


### **Sliding Window**

Sliding window was used to generate the patches to scan for vehicle detection.  The approach was to limit the region of interest to be in the area where cars should be detected...ie, there are not flying cars at the moment so we can limit the search to start at about the mid point on the Y axis.  For higher values on the Y axis, represent depth and as such the patch searched is smaller(scale is reduced) and as you move down the Y axis, the depth is reduced and as such the scale and patch size is increased.  Overlapping regions are used as the depth is reduced to improve detection resolution.  

![alt text][image5]

### **False Positive Filtering**

As shown in the image above, duplicate and false positive detection can occur with this pipeline.  A heat map is created based on the number of detections and a threshold hold is applied to filter out the false positives. 

![alt text][image6]

## **Final Result**

The pipeline described above produces the following image shown below.  The image below also include lane line detection as well.

![alt text][image4]

### **Video Results**

Link to [Result YouTube Video](https://youtu.be/3SCqGUhmyxk) 

### **Discussion**

Training of the SVC seemed to produce very similar results independent of feature extraction parameters, including the amount of data used.  Ultimately the back-end filtering was adjusted to accommodate for the different behavior of the SVC.  In most cases, behavior different, but within reason.  It would have been interesting to try a different classifier to observe training behavior.

Detection of the white car at distance seemed to be an issue.  It also seemed to be associated with the contrast with the road as well.  Again, other CV techniques and/or a different classifier would be interesting to investigate. 

Improvement in detection will increase if the patch dimensions were chosen at a finer grain.  The expense would be processing time...  Techniques like using history and prediction to minimize computation would be one way to solve this problem. 



