def RA_unit(x, h, w, n):
    print("h and n :",h,n)
    # x-input, h-height, w-width, n-slice count
    # if the image height is h and required slice count is n, then the slice height is h/n. So the stride is also h/n.
    # tf.nn.avg_pool(input,ksize,strides,padding,data_format=None,name=Non) one image one channel
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    x_2 = tf.zeros([1, h, w, 0], tf.float32)
    x_t_small = tf.zeros([1, x_1.shape[1].value, w/2, 0], tf.float32)  # h = x_1.shape[1].value, w = w/2
    for k in range(n):
	    x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,w/2,x.shape[3].value]) # tf.slice(input,begin,size,name=None) check further !
	    x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1) # resize back to normal h,w
        x_2 = tf.concat([x_2, x_t_2], axis=3) # stacking resized images to get X`
	    x_t_3 = tf.abs(x - x_t_2) # i`-x`
    	x_t = tf.concat([x_t, x_t_3], axis=3) # staking to get I`-X`
    x_out = tf.concat([x, x_t], axis=3)# D`
    return x_out , x_2