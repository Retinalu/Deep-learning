import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Model_name  =  './Saver/infoGAN_02_G_01'

batch_size  = 32   # batch size
cat_dim     = 10   # total categorical factor
con_dim     = 2    # total continuous factor
rand_dim    = 38


############# 定义生成器 和 鉴别器 #################
def generator(data_in, training=True, reuse=False):
    with tf.variable_scope('generator',reuse=reuse):
        Layer1    = tf.layers.dense(data_in,1024)
        Layer1_BN = tf.contrib.layers.batch_norm(Layer1,activation_fn=tf.nn.relu,is_training=training)

        Layer2    = tf.layers.dense(Layer1_BN ,7*7*128)
        Layer2_BN = tf.contrib.layers.batch_norm(Layer2,activation_fn=tf.nn.relu,is_training=training)
        Layer3    = tf.reshape(Layer2_BN, [-1, 7, 7, 128])

        Layer4    = tf.layers.conv2d_transpose(Layer3,filters=64,kernel_size=[4,4],strides=2,padding='same',activation=None)
        Layer4_BN = tf.contrib.layers.batch_norm(Layer4,activation_fn=tf.nn.relu,is_training=training)

        out       = tf.layers.conv2d_transpose(Layer4_BN,filters=1,kernel_size=[4,4],strides=2,padding='same',activation=tf.nn.sigmoid)
  
    return out

def leaky_relu(x):
     return tf.where(tf.greater(x, 0), x, 0.01 * x)

def discriminator(data_in, num_category=10, batch_size=32, num_cont=2,reuse=False):
    with tf.variable_scope('discriminator',reuse=reuse):
        Layer1   = tf.layers.conv2d(data_in,  filters=64,  kernel_size=[4,4], strides=2, activation=leaky_relu)
        Layer2   = tf.layers.conv2d(Layer1,   filters=128, kernel_size=[4,4], strides=2, activation=leaky_relu)
        Layer2_f = tf.reshape(Layer2,[batch_size,-1])
        Share_Layer3 = tf.layers.dense(Layer2_f,1024,activation=leaky_relu)
    #with tf.variable_scope('dis_for_dis',reuse=reuse):
        Disc_Layer_1 = tf.layers.dense(Share_Layer3, 128, activation=leaky_relu)        
        Disc_Layer_2 = tf.layers.dense(Disc_Layer_1,  1,   activation=None)
        Disc_Layer_3 = tf.squeeze(Disc_Layer_2, -1)
    #with tf.variable_scope('dis_for_cat',reuse=reuse):
        Rec_cat = tf.layers.dense(Share_Layer3, num_category, activation=None)
    #with tf.variable_scope('dis_for_con',reuse=reuse):
        Rec_con = tf.layers.dense(Share_Layer3, num_cont,     activation=tf.nn.sigmoid)   
    return Disc_Layer_3, Rec_cat, Rec_con

def cre_data(batch_size, cat_dim, con_dim, rand_dim):  
    # get random class number
    z_cat = tf.multinomial(tf.ones((batch_size, cat_dim), dtype=tf.float32) / cat_dim, 1)
    z_cat = tf.squeeze(z_cat, -1)
    z_cat = tf.cast(z_cat, tf.int32)

    # continuous latent variable
    z_con = tf.random_normal((batch_size, con_dim))
    z_rand = tf.random_normal((batch_size, rand_dim))

    z = tf.concat(axis=1, values=[tf.one_hot(z_cat, depth = cat_dim), z_con, z_rand])
    return z,z_cat,z_con  

############### 网络结构 ########################
tf.reset_default_graph()

with tf.variable_scope('input'):  
    show_real   = tf.placeholder(tf.float32, [batch_size,28,28,1], name='show_real')     #真实图片
    answer_real = tf.placeholder(tf.float32, [batch_size,10], name='answer_real')        #真实类别
    
##### 流动 ######
x_fake,x_fake_cat,x_fake_con = cre_data(batch_size, cat_dim, con_dim, rand_dim)
    
show_fake = generator(x_fake, training=True, reuse=False)

y_fake, cat_fake, con_fake = discriminator(show_fake, cat_dim, batch_size, con_dim)
y_real, cat_real, _        = discriminator(show_real, cat_dim, batch_size, con_dim, reuse=True)

############### Loss  ###########################
# BN的参数更新
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_real, labels=tf.ones_like(y_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_fake, labels=tf.zeros_like(y_fake)))


loss_cat = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cat_fake, labels=x_fake_cat))
loss_con = tf.reduce_mean(tf.square(con_fake-x_fake_con))

loss_detail = loss_cat    # 改动

d_loss = 1/2*tf.add(d_loss_real, d_loss_fake)
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_fake,labels=tf.ones_like(y_fake)))

D_loss = d_loss + loss_detail
G_loss = g_loss + loss_detail


#############  Accuracy  #####################################

with tf.name_scope('accuracy_fake'):
    correct_prediction_fake = tf.equal(tf.argmax(cat_fake,1,output_type=tf.int32),x_fake_cat)
    accuracy_fake           = tf.reduce_mean(tf.cast(correct_prediction_fake,tf.float32))

pred_real   = tf.argmax(cat_real,1,output_type=tf.int32)
answer      = tf.argmax(answer_real,1,output_type=tf.int32)


############### 优化器  ###########################
train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]


d_train_opt = tf.train.AdamOptimizer(0.0001).minimize(D_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=g_vars)


############### Train #############################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.ion()
plt.show()

print ("Training _______________")

epochs = 100
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            show_real_all = mnist.train.next_batch(batch_size)
            show_real_in  = show_real_all[0].reshape((batch_size,28,28,1))
            answer_real_in   = show_real_all[1].reshape((batch_size,10))
            
            sess.run(d_train_opt, feed_dict={show_real:show_real_in})
            sess.run(g_train_opt, feed_dict={show_real:show_real_in})
        A_cat_fake,loss_category = sess.run([accuracy_fake,loss_cat], feed_dict={show_real:show_real_in})
        print ("After %d  the category fake and loss are:"%(e),A_cat_fake,loss_category)
        #目视分类
        print("______________________________________________")
        A_cat_fake = sess.run(accuracy_fake, feed_dict={show_real:show_real_in})
        pred_y,answer_y = sess.run([pred_real,answer],feed_dict={show_real:show_real_in, answer_real:answer_real_in})
        print ("After %d  the pred is:"%(e), pred_y)
        print ("After %d  the answ is:"%(e), answer_y)
        #可视化数字
        Value_g_outputs = sess.run(show_fake, feed_dict={})
        img = Value_g_outputs[10]
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass         
        lines  = ax.imshow(img.reshape((28,28)),cmap="Greys_r")
        plt.pause(0.3)
        #保存
        saver.save(sess, Model_name)
        

 
