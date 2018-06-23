# Finding Lane Lines on the Road

---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

## Reflection

### Dataset

The dataset consists of 6 static images and 3 videos including white and yellow line markings.


### Implementation

For description and details of the functions look into P1.ipynb

Provided helper functions:
- grayscale(img)
- canny(img, low_threshold, high_threshold)
- gaussian_blur(img, kernel_size)
- region_of_interest(img, vertices)
- draw_lines(img, lines, color=[255, 0, 0], thickness=10)
- hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
- weighted_img(img, initial_img, α=0.8, β=1., γ=0.)

Own functions:
- mask_hls(image)
- save_img(img, name, target_path)
- get_vertices(img)
- process_image(img)

Classes:
- Config


**My lane line detection pipeline consists of 7 main steps**

#### 1. Color masking
Extracting the lane lines via color bitmasking (white, yellow) from the images in RGB color space works good for the white and yellow test videos (solidWhiteRight.mp4 and solidYellowLeft.mp4). The video for the optional challenge (challenge.mp4) is more noisy (shadows) so the RGB color space does not yield the best results for lane line highlighting.

The best line contrast via color bitmasking is achieved with an image conversion to HLS colorspace.

A bitwise combination of white and yellow color masks for line highlighting on top of the input image is introduced. White and yellow color is extracted, everyting else gets blacked out.

Hue, Saturation, Light values are choosen with lower and upper values for the best mask result.

#### 2. Grayscaling
The color masked image is transformed to gray (only one color channel left) so that the Canny edge detector can best measure the pixel intensity changes (gradients) on a homogenous input.

#### 3. Gaussian smoothing (Blur)
Applying an additional gaussian filter with kernel size 13 to the grayscaled image surpresses noise and spurious gradients and leads to smooth lines for Canny.

Kernel sizes must be positive and odd and should be choosen with performance in mind if relevant (bigger kernels -> longer processing -> blurrier result)

#### 4. Canny edge detection
Generally speaking, edges in images exist at locations with rapid changes in pixel intensity.
Canny uses two threshold values (low and high) with no generic ratio description but best practices propose to choose a 1:2 or 1:3 ratio.

#### 5. Region of interest selection
Not the whole input picture is relevant for lane line detection so a region of interest is defined.

The region mask (4 vertices) was initially based on fixed values:
- Vertice1: Bottom left
- Vertice2: Top left
- Vertice3: Top right
- Vertice4: Bottom right

The solution worked OK for the white and yellow test videos (solidWhiteRight.mp4 and solidYellowLeft.mp4). The input image size changes with the challenge video so the vertice-values have been changed to relative values to be able to handle different input image sizes.

#### 6. Hough transformation
The transformation converts e.g. non-black points from the edge detected input image into sinusoidal curves in pθ space. Bright points (lot of curve intersection via votes) indicate the line parameters in xy space.

#### 7. Image annotation
Hough lines are feed into the draw_lines function. The draw lines function has been enhanced to draw complete lines (left and right) from bottom to top of the ROI. Hough transform retrieves multiple lines for each side (left, right) per frame. The differentiation between a left and right line is done via the line slope: 
- (y2-y1)/(x2-x1): (left line) 0 > slope >= 0 (right line)

All line points per side and frame get averaged and np.polyfit is used to retrieve the slope and intercept of the intended line. The pixel values x1 and x2 for the intended line are retrieved via: 

- int(round((line-y - line-intercept) / line-slope))

The start and end-points of the intended line relative to image size.
- line_y1 = img.shape[0]
- line_y2 = int(round(img.shape[0]*0.6))

The black output image including hough lines from hough_lines method is then added with transparency to input image via weighted_img.

### Shortcomings
**Curves:** The current implementation works OK for straight line in video detection. If a video contains sharpe curve scenes the pipeline is not sufficient yet.

**Varying light conditions e.g. night:** Implementation has not been tested against strong varying light condition driving scenes so there is no evidence that the current solution can handle this.

**Steepness:** Current ROI implemenation does not cover road descents or ascents cause it's fixed slightly below the center of the images.

### Improvements
**Curves:** Implement perspective transformations regarding curved line detection. Enhance the np.polyfit to support curve line-points.

**Varying light conditions e.g. night:** Introduce additional advanced image preprocessing steps e.g. Neighbor Average Filtering or Sobel operators to extract line lanes or road boundaries.

**Steepness:** Adapt the ROI according to the driving scene and therefor incorporate steepness driving situations.

**Implementation:** Use more Pythonic constructs within the draw_lines method. Evaluate the possibility to use a neural network (segnet) for line detections.
