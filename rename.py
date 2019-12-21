import os
path='/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/'
arr = os.listdir(path)

if '.DS_Store' in arr:
    arr.remove('.DS_Store')

# for i in arr:
#     old_name=i
#     old_name_part=old_name.split("_")
#     # new_name="img"+"_"+old_name_part[2]
#     new_name="img"+"_"+old_name_part[1]
#     print(i)
#     print(old_name_part)
#     print(new_name)
#     # os.rename(old_name,new_name)

for i in arr:
    old_name=i
    old_name_part=old_name.split(".")
    new_name=old_name_part[0]+".png"
    print(i,'\t',old_name,'\t',new_name)
    os.rename(path+i,path+new_name)
