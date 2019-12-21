import numpy as np

# ar=[1,2,3,4,5]
# ar=[[1,2,3,4,5],[1,2,3,4,5]]
# ar=[[1,2,3,4,[5,5]],[1,2,3,4,5]]
ar=np.array(ar)
s=np.sum(ar,axis=[1,2,3])
print(s)