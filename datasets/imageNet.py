import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np

def download_imagenet_subset(output_folder, num_samples=50000):
    # URL to ImageNet images metadata (contains URLs to images)
    metadata_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289"
    
    # Request metadata
    response = requests.get(metadata_url)
    
    if response.status_code == 200:
        # Extract image URLs
        image_urls = response.text.strip().split('\n')
        
        # Shuffle the URLs to get a random subset
        np.random.shuffle(image_urls)
        
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Download and save images
        for i, url in enumerate(image_urls[:num_samples]):
            try:
                # Download image
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Open the image using PIL
                    image = Image.open(BytesIO(response.content))
                    
                    # Save the image as PNG
                    image.save(os.path.join(output_folder, f"image_{i + 1}.png"), "PNG")
                    
                    print(f"Downloaded image {i + 1}/{num_samples}")
                else:
                    print(f"Failed to download image {i + 1}")
            except Exception as e:
                print(f"Error downloading image {i + 1}: {str(e)}")

    else:
        print(f"Failed to fetch metadata. Status code: {response.status_code}")

# Specify the output folder and the number of samples
output_folder = "imagenet"
num_samples = 50000

# Call the function to download the ImageNet subset
download_imagenet_subset(output_folder, num_samples)
