import requests
import urllib.request
import json
import shutil

access_key = "3908370aa920af41f5b19158e25bc6cf588f43b9d8d86d51590d30f2997a096c"
secret = "aa1dc506312641e6fd954eb78846be0e53aad70f7b78303a121dbe7a7d3dfd0a"
auth = "?client_id=" + access_key

# Get a random image from the unsplash api	
def get_random_image():
	url = "https://api.unsplash.com/photos/random"	
	resource = requests.get(url + auth)
	resource.raise_for_status()
	image = json.loads(resource.text)
	return image
	
# Build a list of random images
def build_random_image_list(num_of_images):
	images = []
	for i in range(num_of_images):
		image = get_random_image()
		images.append(image)
	return images 
	
# Pass in list of unsplash images and the size you wish to save. Valid size options are: raw, full, regular and thumb
def save_images(images, size="regular"):
	images_saved = 0
	for image in images:
		image_url = image['urls'][size]
		filename = str(image['id']) + ".jpg"
		urllib.request.urlretrieve(image_url, filename)
		print("Saving: ", filename)
		images_saved +=1
	print()
	print(images_saved, "images saved")


def main():
	print("Fetching Images...\n")
	images = build_random_image_list(10000)
	save_images(images)
	
	

main()