import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
################
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import requests
import io
from PIL import Image
from urllib.parse import urlparse, urlunparse 
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from sklearn.cluster import DBSCAN
from collections import defaultdict

#######
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
import time
from selenium.webdriver.chrome.service import Service

#####################

import logging
import sys

# Setup logging configuration
logging.basicConfig(filename='error_log.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Redirect stderr to log file
sys.stderr = open('error_log.log', 'w')

# Open a file for results
sys.stdout = open('results.txt', 'w')

# Function to extract domains from the Parquet file
def extract_domains_from_parquet(parquet_file_path):
    # Read the Parquet file into a table
    table = pq.read_table(parquet_file_path)

    # Convert the table to a pandas DataFrame
    df = table.to_pandas()

    # Extract the 'domain' column and store it in a list
    domains = df['domain'].tolist()

    return domains

def extract_logo_from_domain(domain):
    try:
        # First, try using requests and BeautifulSoup for a quick extraction
        response = requests.get(f"http://{domain}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Look for <link> tags containing 'icon' for favicon
            favicon = soup.find("link", rel="icon")
            if favicon:
                return urljoin(f"http://{domain}", favicon['href'])

            # Look for <img> tags for logos
            logo = soup.find("img")
            if logo:
                return urljoin(f"http://{domain}", logo['src'])

        # If BeautifulSoup fails to find the logo, use Selenium for dynamic content
        # Set up Chrome options for Selenium (headless mode)
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("window-size=1200x600")

        # Path to your ChromeDriver
        chrome_driver_path = r"C:\Users\Maria\Downloads\chromedriver-win32\chromedriver-win32\chromedriver.exe"  # Replace with your ChromeDriver path

        # Initialize WebDriver
        service = Service(chrome_driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Ensure we're using HTTPS
        url = f"https://{domain}"

        driver.get(url)

        # Wait for the page to load and look for images (adjust time if needed)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "img"))
        )

        # Look for favicons in <link> tags
        favicon_url = None
        link_tags = driver.find_elements(By.TAG_NAME, "link")
        for link in link_tags:
            rel = link.get_attribute("rel")
            if rel in ["icon", "shortcut icon", "apple-touch-icon"]:
                href = link.get_attribute("href")
                if href:
                    favicon_url = urljoin(url, href)
                    break

        # Look for <img> tags with potential logo class names
        logo_candidates = ["logo", "site-logo", "brand-logo"]
        logo_url = None
        if not favicon_url:
            for class_name in logo_candidates:
                logos = driver.find_elements(By.CLASS_NAME, class_name)
                for logo in logos:
                    src = logo.get_attribute("src")
                    if src:
                        logo_url = urljoin(url, src)
                        break
                if logo_url:
                    break

        # Close the WebDriver after extracting logo
        driver.quit()

        # Return the logo URL if found, else None
        return favicon_url if favicon_url else logo_url

    except Exception as e:
        logging.error(f"Error fetching logo for {domain}: {e}")
        #print(f"Error fetching logo for {domain}: {e}")
        return None

# Function to store logos in map <domain, logo>
def map_logos_to_domains(domains):
    # Create an empty dictionary to store the logos
    domain_logo_map = {}

    # Iterate over the list of domains
    for domain in domains:
        # Extract the logo for the domain
        logo = extract_logo_from_domain(domain)

        # If a logo is found, add it to the dictionary
        if logo:
            domain_logo_map[domain] = logo
        else :
            domain_logo_map[domain] = "No logo found"

    return domain_logo_map

# Preprocess Images
# Resize all logos to a standard size for consistency.
#Convert images to grayscale for feature extraction.
# Function to resize image with padding (if necessary) to maintain aspect ratio
def resize_with_padding(image, target_size):
    height, width = image.shape
    target_width, target_height = target_size

    # Compute the aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Pad the image to match the target size
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    return padded_image
def process_images(logo_map):
    """Loads and preprocesses images from URLs, ensuring they are ready for ORB feature extraction."""
    processed_images = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for domain, logo_url in logo_map.items():
        try:
            if logo_url.startswith("http"):  # If it's a URL
                # Remove query parameters from the URL (e.g., ?v2)
                parsed_url = urlparse(logo_url)
                cleaned_url = urlunparse(parsed_url._replace(query=""))

                response = requests.get(cleaned_url, headers=headers, timeout=5)

                # Check for successful response and valid image content type
                if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                    image_bytes = io.BytesIO(response.content)

                    # Handle ICO files (multi-image format)
                    if cleaned_url.endswith(".ico"):
                        try:
                            # Open the ICO file
                            img = Image.open(image_bytes)

                            # If it's an ICO file with multiple images, try to extract the first image
                            img.seek(0)  # Move to the first image in the ICO file
                            img = img.convert("RGB")  # Convert to RGB if needed
                            img = np.array(img)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        except Exception as e:
                            #print(f"Error processing ICO image for {domain}: {e}")
                            print(f"Error processing ICO image for {domain}: {e}", file=sys.stderr)
                            img = None
                    else:
                        try:
                            img_array = np.frombuffer(response.content, np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        except Exception as e:
                            #print(f"Error decoding image for {domain}: {e}")
                            print(f"Error decoding image for {domain}: {e}", file=sys.stderr)
                            img = None

                    # Preprocess the image for ORB
                    if img is not None:
                        # Resize image to a standard size for consistency
                        img = resize_with_padding(img, (200, 200))  # Resize to 200x200 pixels

                        # Apply a binary threshold to enhance the features
                        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 11, 2)

                        # Store processed image
                        processed_images[domain] = img
                    else:
                        #print(f"Failed to process image for {domain}")
                        print(f"Failed to process image for {domain}", file=sys.stderr)

                else:
                    #print(f"Failed to download {cleaned_url} (Status Code: {response.status_code}) or invalid content type")
                    print(f"Failed to download {cleaned_url} (Status Code: {response.status_code}) or invalid content type", file=sys.stderr)

        except Exception as e:
            logging.error(f"Error processing {domain}: {e}")
            #print(f"Error processing {domain}: {e}")
            print(f"Error processing {domain}: {e}", file=sys.stderr)

    return processed_images



def extract_hog_features(image):
    """Extract HOG features from an image."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image: Image is empty or not loaded properly")
    
    if len(image.shape) == 2:  # Already grayscale
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unexpected image format")
    
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, visualize=True)
    return features


def compute_ssim(img1, img2):
    """Compute the Structural Similarity Index (SSIM) between two images."""
    if img1 is None or img2 is None:
        raise ValueError("One or both images are empty or not loaded properly")
    
    # Ensure both images are grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1  # Already grayscale

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2  # Already grayscale
    
    # Compute SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return score
def match_features(feature1, feature2):
    """Compute similarity score between two feature vectors."""
    return np.linalg.norm(feature1 - feature2)

def normalize_matrix(matrix):
    """Normalize the similarity matrix to be in range [0,1]."""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val > 0:
        return (matrix - min_val) / (max_val - min_val)
    return matrix

def group_similar_logos(logo_map, threshold):
    """Group similar logos based on HOG + SSIM similarity."""
    processed_images = process_images(logo_map)
    features_list = []
    domains = list(processed_images.keys())
    
    # Extract features for each logo
    for domain in domains:
        features = extract_hog_features(processed_images[domain])
        features_list.append(features)
    
    # Compute pairwise similarity matrix
    similarity_matrix = np.zeros((len(domains), len(domains)))
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            ssim_score = compute_ssim(processed_images[domains[i]], processed_images[domains[j]])
            feature_dist = match_features(features_list[i], features_list[j])
            similarity_matrix[i, j] = (ssim_score + (1 / (1 + feature_dist))) / 2
            similarity_matrix[j, i] = similarity_matrix[i, j]
    
    # Normalize similarity matrix
    similarity_matrix = normalize_matrix(similarity_matrix)
    
    # Apply DBSCAN clustering (similar logos in one group)
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='precomputed').fit(1 - similarity_matrix)
    
    # Group domains by cluster
    grouped_results = defaultdict(list)
    
    for i, label in enumerate(clustering.labels_):
        if label == -1:  # Outliers (logos that are not similar to others)
            grouped_results[i].append(domains[i])  # Treat each outlier as its own group
        else:
            grouped_results[label].append(domains[i])  # Group similar logos together
    
    # Return the grouped results
    return grouped_results


# Main function to execute the script
def main():
    # Define the path to your Parquet file
    parquet_file_path = r"C:\Users\Maria\Downloads\logos.snappy.parquet"
    
    # Call the function to extract domains
    #domains = extract_domains_from_parquet(parquet_file_path)

    # Print the list of domains
    #print(domains)

    # Define a list of domains for testing
    domains = ["google.com","target.com", "target.com", "facebook.com", "microsoft.com", "apple.com", "ikea.com.hk", "ikea.com.cn", "ikea.com", "culliganheartland.com", "kierpensions.co.uk", "kier.co.uk", "kiergroup.com", "linde.pe"]

    # Call the function to map logos to domains
    domain_logo_map = map_logos_to_domains(domains)

    # Print the domain-logo map
    for domain, logo in domain_logo_map.items():
        print(f"Domain: {domain}, Logo: {logo}")

    # Group similar logos
    similar_logo_groups = group_similar_logos(domain_logo_map, threshold=0.1)

    # Print the final grouped result
    for group, domains in similar_logo_groups.items():
        #print(f"Group {group}: {domains}")
        # write in results file
        print(f"Group {group}: {domains}", file=sys.stdout)




# Run the main function
if __name__ == "__main__":
    main()
