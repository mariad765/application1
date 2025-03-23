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
The solution is designed to cluster websites based on the visual similarities of their logos, using a non-machine learning approach. I codded this application in Python. This is a short diagram on the steps I followed:

+--------------------------------+  
|  Logo Extraction               |  
|  (BeautifulSoup & Selenium)    |  
+--------------------------------+  
        │  
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

## Functionality Breakdown

### Extracting the logos

The following diagram will explain how I managed the extraction of logos:

+----------------------------------------------------+  
| Function: extract_logo_from_domain(domain)         |  
+----------------------------------------------------+  
        │  
        ▼  
+--------------------------------------+  
| 1. Try Using Requests & BeautifulSoup|  
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
| - Check <link> tags for "icon", "shortcut icon"|  
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
| - Else, return extracted <img> src|  
| - Log errors if any occur         |  
+-----------------------------------+  


Explaation:







