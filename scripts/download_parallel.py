import requests
import os
from multiprocessing import Pool, freeze_support
# Define your NASA Earthdata username and password

# Define the function to download HDF files
def download_from_url(url, output_dir):
    # Extract filename from URL
    last_slash_index = url.split('/')[-1]
    filename = last_slash_index.split('?')[0]
    # Define the output path
    output_path = os.path.join(output_dir, filename)
    # Download the file
    response = requests.get(url)
    # Check if request was successful (status code 200)
    print(response.status_code)
    if response.status_code == 200:
        # Save the HDF file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Download complete for", filename)
    else:
        print("Failed to download", filename)
        print("Error message:", response.text)

# Define the function to download HDF files from a single link
def download(link_output_dir):
    link, output_dir = link_output_dir
    download_from_url(link, output_dir)

# Define the function to read links from a text file and download HDF files using multiprocessing
def download_from_links_file(file_path, output_dir):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Read links from the text file
    with open(file_path, 'r') as f:
        links = f.read().splitlines()
    
    # Create a list of link-output_dir pairs
    link_output_dirs = [(link, output_dir) for link in links]
    # Download HDF files using multiprocessing
    with Pool(4) as pool:
        pool.map(download, link_output_dirs)

# Input feature



if __name__ == '__main__':
    feature = input("Enter the feature: ")

    # Specify the path to the text file containing links
    links_file = f'dataset/{feature}/download_links.txt'

    # Specify the output directory
    output_directory = f'dataset/{feature}/downloaded_files'
    freeze_support()
    # Call the function to download HDF files from the links in the text file using multiprocessing
    download_from_links_file(links_file, output_directory)
