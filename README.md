# README

## Purpose
This document serves as an overview of the solution developed for a technical assessment as part of a job application. The task focuses on matching and grouping websites based on the similarity of their logos.

## Task Description
The objective is to identify and cluster websites according to the visual similarities of their logos. Key aspects of the task include:

- **Understanding the Problem:** Before writing code, ensuring alignment with the correct objectives to avoid misalignment and wasted effort.
- **Clustering without ML Algorithms:** Exploring non-ML approaches instead of traditional algorithms like DBSCAN or k-means.
- **Logo Extraction and Matching:** Ensuring accurate logo extraction and effective matching, considering the challenges machines face compared to human perception.
- **Exploring Multiple Angles:** Investigating various perspectives to generate valuable insights.
- **Tech Stack Flexibility:** Using any preferred programming language, tools, or libraries best suited for the task (Veridion has a preference for Node, Python, and Scala).
- **Scalability Consideration:** While not mandatory, designing an approach that has the potential to scale is beneficial.

## General Description
The solution is designed to cluster websites based on the visual similarities of their logos, using a non-machine learning approach. I codded this application in Python.

This is a short diagram on the steps I followed: 
<pre>
+--------------------------------+  
|  Logo Extraction               |  
|  (BeautifulSoup & Selenium)    |  
+--------------------------------+
             |
             ▼
+----------------------+  
|  Image Processing    |  
| (Resizing, Noise     |  
|  Reduction, etc.)    |  
+----------------------+
              │
              ▼
+----------------------+  
|  Feature Extraction  |  
| (Keypoints, Hashing) |  
+----------------------+
             │
             ▼
+-------------------------+  
|  Similarity Measurement |  
| (SSIM, Histograms)      |  
+-------------------------+
              │
              ▼
+-------------------------+  
|   Clustering            |  
| (Graph or Heuristic,    |  
| DBSCAN (Density-Based   |  
| Spatial Clustering of   |  
| Applications with Noise)|  
+-------------------------+
           │
           ▼
+---------------------+  
| Scalability &       |  
| Performance         |  
| Optimization        |  
+---------------------+
             │
             ▼ 
+--------------------+  
|  Error Logging     |  
| (Errors stored in  |  
|   log file)        |  
+--------------------+  
</pre>
## Functionality Breakdown

### Extracting the logos

The following diagram will explain how I managed the extraction of logos:
<pre>
+----------------------------------------------------+  
| Function: extract_logo_from_domain(domain)         |  
+----------------------------------------------------+  
        │  
        ▼  
+--------------------------------------+  
| 1. Try Using Requests & BeautifulSoup |  
| - Send HTTP request to domain        |  
| - Parse HTML for <link rel="icon">   |  
| - Parse HTML for <img> tags          |  
+--------------------------------------+  
        │  
        ├─> If found, return logo URL  
        │  
        ▼  
+-------------------------------------------------+   
| 2. If BeautifulSoup Fails, Use Selenium        |  
| - Set up headless Chrome WebDriver             |  
| - Open the website using HTTPS                 |  
| - Wait for the page to load                    |  
+-------------------------------------------------+  
        │  
        ▼  
+------------------------------------------------+  
| 3. Look for Favicon and Logo Using Selenium    |  
| - Check <link> tags for "icon",                |  
|    "shortcut icon"                             |  
| - Check <img> tags with class names:           |  
|   "logo", "site-logo", "brand-logo"            |  
+------------------------------------------------+  
        │  
        ▼  
+-------------------------------+  
| 4. Close WebDriver            |  
| - Release resources           |  
+-------------------------------+  
        │  
        ▼  
+-----------------------------------+  
| 5. Return Found Logo or None      |  
| - If favicon found, return it     |  
| - Else, return extracted <img> src |  
| - Log errors if any occur         |  
+-----------------------------------+  
</pre>

Explanation:

BeautifulSoup4 [https://www.geeksforgeeks.org/beautifulsoup4-module-python/] (*a parser*) is a user-friendly Python library designed for parsing HTML and XML documents. 
However, I ran into an issue when using it alone. Some links for images were not found. I found a couple 
of issues with BeautifulSoup:  
* Parses static HTML content fetched from a webpage, which works well for pages that do not require JavaScript to render their content. However, if the page relies on JavaScript to load or update content dynamically (like loading logos or other images after page load), BeautifulSoup alone can't handle that Struggles with JavaScript-heavy websites because it can't execute scripts. If the website's content is dependent on JavaScript, BeautifulSoup will only show the static HTML content, which might be incomplete.
* Doesn’t interact with browsers at all, so you can’t use headless browser features to automate scraping tasks.  
On the small test sample I used, BeautifulSoup failed to find a logo for many domain names. This is when I started looking for another parser. I found Selenium. *Web scraping* sometimes involves extracting data from dynamic content. Selenium is a multipurpose tool that enables you to interact with a browser and grab the data you require, which is ideal for scraping dynamic content.  

![alt text](image.png)  

Steps to scrap using selenium are presented on the following diagram:  
<pre>
+--------------------------------------------+
|          Launching a Browser with         |
|                Selenium                   |
+--------------------------------------------+
                 |
                 ▼
+--------------------------------------------+
|     Set options for Selenium               |
|     Initialize WebDriver and launch        |
|     the browser                            |
+--------------------------------------------+
                 |
                 ▼
+--------------------------------------------+
|   Navigating and Interacting with Web      |
|                Pages                       |
+--------------------------------------------+
                 |
                 ▼
+--------------------------------------------+
|  Use `driver.get(url)` to load the page    |
|  Find elements with `driver.find_element`  |
|  Interact with elements (click, enter text) |
+--------------------------------------------+
                 |
                 ▼
+--------------------------------------------+
|     Handling JavaScript Rendered          |
|            Elements                       |
+--------------------------------------------+
                 |
                 ▼
+--------------------------------------------+
|  Wait for specific elements to appear      |
|  with WebDriverWait and ExpectedConditions |
|  Interact with elements once visible       |
+--------------------------------------------+

</pre>

Logo candidates explanation:

<pre>
         +------------------------------+
         |    Extract Logo from Domain  |
         +------------------------------+
                       |
       ----------------------------------------
       |                                      |
+-------------------+             +----------------------+
| BeautifulSoup     |             | Selenium Headless    |
| - <link rel="icon"> |             | - Searches <img> tags |
| - First <img> tag |             | - Looks for classes:  |
+-------------------+             |   ├── "logo"          |
       |                          |   ├── "site-logo"     |
       |                          |   ├── "brand-logo"    |
       |                          |   ├── "header-logo"   |
       |                          +----------------------+
       |                                      |
       |                                      |
       |                                      |
       +------------>  No Logo Found (Fallback)

</pre>
A favicon (short for "favorite icon") is a small icon associated with a website.Common file formats: .ico, .png, .svg, .jpg
<pre>
Format	Description
.ico	Icon file format that supports multiple resolutions and is widely used for favicons.
.png	A high-quality, lossless image format that supports transparency, commonly used for favicons.
.svg	A scalable vector format ideal for high-resolution screens, allowing infinite scalability without quality loss.
.jpg / .jpeg	A lossy compressed image format, best for simple favicons without transparency.
.gif	A format that supports animations but has a limited color palette (256 colors).
</pre>
After findind the logo, the program maps the domain name to the link of the logo:  
<pre>
map_logos_to_domains(domains)
|
|-- google.com  →  https://www.google.com/favicon.ico
|-- facebook.com  →  https://www.facebook.com/favicon.ico
|-- nonexistentwebsite.com  →  No logo found
</pre>
The image must be processed because initially, it is very colorful and each logo is of a different size / rezolution.
This is a diagram that explains how the process_image function works:
<pre>
+----------------------------+
|     process_images()       |
+----------------------------+
          |
          v
+----------------------------+
| Loop through logo_map      |
+----------------------------+
          |
          v
+----------------------------+
| Check if URL is valid (http)|
+----------------------------+
          |
          v
+----------------------------+
|  Clean URL (remove queries)|
+----------------------------+
          |
          v
+----------------------------+
| Send request to fetch image|
+----------------------------+
          |
          v
+----------------------------------------+
| Check if response is valid & image type|
+----------------------------------------+
          |
    +-----+-------------------+
    |                         |
    v                         v
+--------------------+   +--------------------+
| ICO format?       |   | Other image formats |
| (favicon .ico)    |   | (PNG, JPG, etc.)    |
+--------------------+   +--------------------+
    |                       |
    v                       v
+--------------------+   +--------------------------+
| Convert ICO to RGB |   | Decode image as grayscale |
| Extract first frame |   |                          |
+--------------------+   +--------------------------+
          |                 |
          v                 v 
+------------------------------+
| Resize image to (200x200)    |
+------------------------------+
          |
          v
+------------------------------+
| Apply thresholding (binary)  |
+------------------------------+
          |
          v
+------------------------------+
| Store processed image        |
+------------------------------+
          |
          v
+----------------------------+
| Return processed_images {} |
+----------------------------+

</pre>

Auxiliary function in program:

<pre>
+------------------------------------------------------+
|              resize_with_padding(image)             |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| Extract height & width from the input image         |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| Compute aspect ratio: aspect_ratio = width / height |
+------------------------------------------------------+
                         |
                         v
+--------------------------------------------+
| Determine new dimensions based on         |
| aspect ratio to fit within target size    |
+--------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| Resize image while maintaining aspect ratio         |
+------------------------------------------------------+
                         |
                         v
+--------------------------------------------+
| Compute padding to center the image       |
| (top, bottom, left, right)                |
+--------------------------------------------+
                         |
                         v
+--------------------------------------------+
| Add padding with white background (255)   |
+--------------------------------------------+
                         |
                         v
+----------------------------+
| Return padded image        |
+----------------------------+

</pre>
Explanation of the auxiliary function:
* Extract image dimensions

The function reads the input image’s height and width.

* Compute Aspect Ratio

Determines if the image is wider or taller to decide resizing strategy.

* Resize While Keeping Proportions

If wider → width = target width, height is scaled

If taller → height = target height, width is scaled

* Compute Padding

Determines how much space is needed above, below, left, and right to maintain centering.

* Add Padding

Uses cv2.copyMakeBorder to pad the image with white (255) pixels.

* Return Final Image

The output image has a fixed target_size, but the original aspect ratio is preserved.

### Difficulties found in process_image function:
Some URLs needed cleaning else it would have lead to an error.


## Algorithm used for matching
Object detection algorithm: HOG  
Histogram of Oriented Gradients - It is a feature descriptor commonly used in computer vision and image processing, particularly for object detection. [https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients]


<pre>
+---------------------------+
|     Input Image           |
+---------------------------+
            |
            v
+---------------------------+
|  Compute Gradients        |
|  (Magnitude and Direction) |
+---------------------------+
            |
            v
+---------------------------+
|  Divide Image into Cells  |
|                           |
+---------------------------+
            |
            v
+---------------------------+
|  Compute Histograms       |
|  (9 orientation bins per  |
|   cell, weighted by       |
|   gradient magnitude)     |
+---------------------------+
            |
            v
+---------------------------+
|  Group Cells into Blocks  |
|                           |
+---------------------------+
            |
            v
+---------------------------+
|  Normalize Block Histograms|
|  (Improves invariance to  |
|   lighting changes)       |
+---------------------------+
            |
            v
+---------------------------+
|  Concatenate Histograms   |
|  (Final HOG descriptor)   |
+---------------------------+
            |
            v
+---------------------------+
|   Output HOG Descriptor    |
+---------------------------+

</pre>

A feature is a value that encodes information about the gradients (changes in intensity) of an image, particularly the orientation and magnitude of these gradients. These features are designed to represent the shape, texture, and structure of objects within an image. 

HOG Features:

* Gradient Orientation Histogram: These are features that capture the gradient information in the image. HOG works by computing the gradient direction and magnitude at each pixel, then creating histograms of gradient orientations for small regions (cells).

* Normalization: These histograms are then normalized in blocks to improve robustness against lighting variations and contrast.

* Feature Vector: The result is a long feature vector where each element represents the histogram of gradients in a cell or block.

*SSIM(Structural Similarity Index)* is a metric used to measure the similarity between two images. It is based on the idea that the human visual system is highly sensitive to structural information (edges, textures, etc.) but less sensitive to pixel-level differences such as noise or small variations in brightness. SSIM tries to measure how similar the structural information in the two images is.

The SSIM index takes three main factors into account:

* Luminance: This compares the brightness of the images.

* Contrast: This compares the contrast (the difference in intensity) between the two images.

* Structure: This compares the patterns in the images, capturing how similar the spatial arrangement of pixels is.

![WhatsApp Image 2025-03-23 at 23 02 16_361b2204](https://github.com/user-attachments/assets/f023af99-d507-4385-97e2-762169095e18)

The SSIM score is a numerical value that quantifies the similarity between the two images. A higher score means that the images are more similar in terms of structure, contrast, and luminance.

<pre>
  +----------------------------+
  |    Function: compute_ssim   |
  +----------------------------+
               |
               V
  +-----------------------------+
  | Check if img1 or img2 is    |
  | None (empty or not loaded)  |
  +-----------------------------+
               |
     No        |        Yes
   +-----------+-----------+
   | Raise ValueError       |
   +------------------------+
               |
               V
  +-----------------------------+
  | Check if img1 is a color   |
  | (3 channels)                |
  +-----------------------------+
               |
     Yes       |        No
   +-----------+---------------+
   | Convert img1 to grayscale |
   +----------------------------+
               |
               V
  +-----------------------------+
  | Check if img2 is a color    |
  | (3 channels)                |
  +-----------------------------+
               |
     Yes       |        No
   +-----------+-----------+
   | Convert img2 to grayscale |
   +----------------------------+
               |
               V
  +-------------------------------+
  | Compute SSIM between gray1 and |
  | gray2                          |
  +-------------------------------+
               |
               V
  +----------------------------+
  | Return SSIM score          |
  +----------------------------+

</pre>

match_features(feature1, feature2) - this function calculates the similarity score between two feature vectors.
<pre>
+---------------------------------------+
|           Function: match_features    |
+---------------------------------------+
                |
                V
    +----------------------------+
    |  Input: feature1, feature2  |
    |  Example:                   |
    |  feature1 = [1, 2, 3]       |
    |  feature2 = [4, 5, 6]       |
    +----------------------------+
                |
                V
    +----------------------------------------+
    | Compute Euclidean distance between    |
    | feature1 and feature2 using           |
    | np.linalg.norm (norm of difference)   |
    | Example:                               |
    | distance = sqrt((1-4)^2 + (2-5)^2 +   |
    | (3-6)^2) = sqrt(9 + 9 + 9) = sqrt(27) |
    | = 5.2                                  |
    +----------------------------------------+
                |
                V
    +---------------------------------------+
    |  Output: Similarity score (distance)  |
    |  Example:                             |
    |  5.2                                   |
    +---------------------------------------+
                
   


</pre>

 normalize_matrix(matrix)- this function normalizes the values of a given matrix so that all the values lie within the range [0, 1].
 <pre>
              
+---------------------------------------------+
|           Function: normalize_matrix        |
+---------------------------------------------+
                |
                V
    +-----------------------------+
    |  Input: matrix              |
    |  Example:                   |
    |  matrix =                   |
    |  [[10, 20, 30],             |
    |   [40, 50, 60]]             |
    +-----------------------------+
                |
                V
    +--------------------------------------+
    |  Compute min_val = min(matrix)     |
    |  Compute max_val = max(matrix)     |
    |  Example:                          |
    |  min_val = 10, max_val = 60        |
    +--------------------------------------+
                |
                V
     +-------------------------------+
     | Check if max_val - min_val > 0 |
     +-------------------------------+
           |               |
     No    |               |    Yes
  +--------+-------+       | +----------------------+
  | Return original matrix| | Normalize matrix      |
  | (no variation)        | | using:                |
  +-----------------------+ | (matrix - min_val) /  |
                            | (max_val - min_val)   |
                            | Example:              |
                            | normalized_matrix =    |
                            | [[0.0, 0.2, 0.4],     |
                            | [0.6, 0.8, 1.0]]      |
                            +----------------------+
                |
                V
    +-----------------------------+
    |  Output: Normalized matrix   |
    |  Example:                    |
    |  [[0.0, 0.2, 0.4],          |
    |   [0.6, 0.8, 1.0]]          |
    +-----------------------------+
 </pre>

 ![WhatsApp Image 2025-03-23 at 23 12 01_f49e052f](https://github.com/user-attachments/assets/91c8211e-dad0-4899-8bcf-7fdc430d121d)


 The logos are grouped  based on the similarity matrix.
 <pre>
 +---------------------------------------------+
|        Function: group_similar_logos        |
+---------------------------------------------+
                |
                V
    +-----------------------------+
    | Input: logo_map, threshold  |
    | Example:                    |
    | logo_map = {domain1: img1,   |
    |              domain2: img2}  |
    | threshold = 0.5              |
    +-----------------------------+
                |
                V
    +------------------------------+
    | Call process_images(logo_map) |
    | Example: process_images -> {  |
    | processed_images = {domain1:  |
    | image1_processed, domain2:    |
    | image2_processed}             |
    +------------------------------+
                |
                V
    +--------------------------------------+
    | Loop through processed_images to    |
    | extract HOG features using          |
    | extract_hog_features() for each logo |
    | Example: features_list = [features1, |
    | features2]                          |
    +--------------------------------------+
                |
                V
    +------------------------------------------+
    | Compute pairwise similarity matrix:     |
    | For each pair (i, j), compute SSIM and  |
    | HOG feature distance using match_features |
    | similarity_matrix[i, j] = (ssim_score +  |
    | (1 / (1 + feature_dist))) / 2            |
    | Example: similarity_matrix = [[0.9, 0.7], |
    |                                 [0.7, 0.9]] |
    +------------------------------------------+
                |
                V
    +-------------------------------------------+
    | Normalize similarity matrix using        |
    | normalize_matrix(similarity_matrix)      |
    | Example: normalized_similarity_matrix =    |
    | [[0.0, 0.4], [0.4, 0.0]]                  |
    +-------------------------------------------+
                |
                V
    +------------------------------------------+
    | Apply DBSCAN clustering to the similarity |
    | matrix with eps=threshold, min_samples=1  |
    | Example: clustering.labels_ = [0, 0]      |
    +------------------------------------------+
                |
                V
    +-------------------------------------------+
    | Group logos by DBSCAN labels:            |
    | grouped_results = {0: [domain1, domain2]} |
    | Example: group domains with same label   |
    +-------------------------------------------+
                |
                V
    +--------------------------------------------+
    | Output: grouped_results                    |
    | Example: grouped_results = {0: [domain1,    |
    |                             domain2]}       |
    +--------------------------------------------+

 </pre>

 *DBSCAN* (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm used to group together closely packed data points while marking points that lie alone in low-density regions as outliers. It's widely used in unsupervised machine learning for clustering spatial data.

 *Key Concepts of DBSCAN:*
* Core Points: A point that has at least a minimum number of points within a specified distance (epsilon). These points are central to a cluster.

* Border Points: Points that are not core points but fall within the neighborhood of a core point.

* Noise Points: Points that are neither core points nor border points. These are outliers and don't belong to any cluster.

* Neighborhood: The set of points within a specified radius from a given point.

[https://www.datacamp.com/tutorial/dbscan-clustering-algorithm]

The flow of the main function is:
<pre>
[Parquet File] --> [Extract Domains] --> [Map Logos to Domains] --> [Process Images] --> [Extract HOG Features]
                                                                                                 |
                                                                                                 v
                                                                                         [Compute SSIM]
                                                                                                 |
                                                                                                 v
                                                                                         [Match Features]
                                                                                                 |
                                                                                                 v
                                                                                         [Normalize Matrix]
                                                                                                 |
                                                                                                 v
                                                                                         [Group Similar Logos] --> [Display Groups]

</pre>
# How to install

### Prerequisites

 ```
pip install pyarrow requests beautifulsoup4 opencv-python-headless numpy scikit-learn Pillow scikit-image selenium
```

Install ChromeDriver   
[https://youtu.be/dz59GsdvUF8]  

```
git clone <repo>
```
# How to run

```
python main.py
```
