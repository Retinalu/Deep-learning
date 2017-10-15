import tensorflow as tf


### 图像识别问题   像素  28 * 28 = 784
### 十分类

############## 定义读入数据得函数  ############################

def read_and_decode (file_queue):                #读取一张图片
    reader = tf.TFRecordReader()                 #tf规定的一种图片格式
### 省略
    return  image,label


def  read_image_batch(file_queue,batch_size):    #读取一组图片
### 省略   
    return image_batch,one_hot_labels



################   定义参数   ############################
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

### 训练文件的路径
train_file_path = "oss://my-tf-test/mnist_train/train.tfrecords"
### 读取
train_image_filename_queue = tf.train.string_input_producer([train_file_path])
### 函数 read_image_batch    
train_images,train_labels  = read_image_batch(train_image_filename_queue,128)   #128张训练数据

### 重塑X
x  = tf.reshape(train_images,[-1,784])    # -1 代表此纬度未知且不限制
### 计算Y 模型      Y = X(T)W + b
y  = tf.nn.softmax（tf.matmul(x,w) + b）
### 标签Y
y_ = tf.to_float(train_labels)

### 计算损失函数  (采用交叉熵)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
### 训练模型
train_step    = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


################   执行   ############################
init = tf.global_variables_initializer()

with tif.Session as sess:
    sess.run(init)

    coord   = tf.train.coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(101):
        sess.run(train_step)
        if i % 10 = 0:
            print ("step:%d accuracy:%f" %(i,sess.run(accuracy)))

    coord.request_stop()
    coord.join(threads)
    print ("done")

    














