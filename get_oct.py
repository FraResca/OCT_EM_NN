from PIL import Image
import matplotlib.pyplot as plt
import os

octdir = 'OCT'
cropdir = 'OCT_crops'

for filename in os.listdir(octdir):
    im = Image.open(os.path.join(octdir, filename))
    
    width, height = im.size
    left = height
    top = height / 6
    right = width
    bottom = height * (5/6)
    
    im1 = im.crop((left, top, right, bottom))

    im1.save(os.path.join(cropdir, filename))    


# Opens a image in RGB mode
im = Image.open('02_0.tif')
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
 
# Setting the points for cropped image
left = height
top = height / 6
right = width

bottom = height * (5/6)
 
# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))

print(im1.size)
# Shows the image in image viewer
im1.save('02_0_cropped.tif')