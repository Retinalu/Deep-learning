import tensorflow as tf



### 文件的路径
test_file_path = "oss://my-tf-test/mnist_train/test.tfrecords"
### 读取
test_image_filename_queue = tf.train.string_input_producer([train_file_path])
### 函数 read_image_batch
test_images,test_labels  = read_image_batch(test_image_filename_queue,10000)

### 重塑X
x_test  = tf.reshape(train_images,[-1,784])    # -1 代表此纬度未知且不限制
### 计算Y 模型      Y = X(T)W + b
y_pred  = tf.nn.softmax（tf.matmul(x_test,w) + b）
### 标签Y
y_test = tf.to_float(test_labels)


### 准确率定义  布尔运算，按照最大拟然算法返回的分类目录是否 相等
correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_test,1))
### 准确率预测
accuracy           = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


