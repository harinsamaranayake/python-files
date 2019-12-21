import os
print ("root prints out directories only from what you specified")
print ("dirs prints out sub-directories from root")
print ("files prints out all files from root and directories")

path="/Users/harinsamaranayake/Documents/Research/Datasets/drone_images"
print(next(os.walk(path))[2])

# for root, dirs, files in os.walk("/var/log"):
#     print (root)
#     print (dirs)
#     print (files)