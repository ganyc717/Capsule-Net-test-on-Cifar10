import tensorflow as tf
import config as cfg
import tensorflow.contrib.slim as slim

epsilon = 1e-9

class CapsuleNet():
    def __init__(self,is_training = True):
        self.image_size = cfg.IMAGE_SIZE
        self.category_num = cfg.CATEGORY_NUM
        self.input_img = tf.placeholder(tf.float32,[None,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3],"input_img")
        self.input_label = tf.placeholder(tf.uint8,[None],"input_label")
        self.buildNet(self.input_img,is_training = is_training)
        if is_training:
            self.getloss(self.input_label)
            self.summary()


    def getloss(self,input_label):
        #The margin loss
        positive = tf.square(tf.maximum(0., cfg.MAX_POSITIVE_THRES - self.output_len))
        negative = tf.square(tf.maximum(0., self.output_len - cfg.MIN_NEGATIVE_THRES))
        positive_label = tf.cast(tf.one_hot(input_label,cfg.CATEGORY_NUM),tf.float32)
        negative_label = 1. - positive_label
        margin_loss = tf.reduce_sum(positive_label * positive + cfg.NEGATIVE_LOSS_WEIGHT + negative_label * negative,axis=1)
        self.margin_loss = tf.reduce_mean(margin_loss)

        #The reconstruction loss
        raw_img = tf.reshape(self.input_img,[cfg.BATCH_SIZE,-1])
        reconstruct_loss = tf.reduce_sum(tf.square(self.decoded - raw_img),axis=1)
        self.reconstruct_loss = tf.reduce_mean(reconstruct_loss)

        self.total_loss = self.margin_loss + cfg.REGULARIZATION_WEIGHT * self.reconstruct_loss

    #Build the network, input the image data, output 10 categories possibility.
    def buildNet(self,input_img,is_training = True):
        with tf.variable_scope("variable_scope"):
            net = slim.conv2d(input_img, 256, 13, 1,padding='VALID',
                              weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),scope='conv0')
            net = slim.conv2d(net,cfg.CAPSULE1_VEC_NUM * cfg.CAPSULE1_VEC_LEN, 9, 2,
                              padding='VALID',weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),scope='capsule1')

            featureMap_size = net.shape[1].value

            capsule1 = tf.reshape(net, [-1, featureMap_size * featureMap_size * cfg.CAPSULE1_VEC_NUM,
                                        cfg.CAPSULE1_VEC_LEN])
            capsule1 = self.squash(capsule1)
            '''
            # 2 layer route takes too much time training, so just takes one layer routing.
            capsule2 = self.route(capsule1, cfg.CAPSULE2_VEC_NUM, cfg.CAPSULE2_VEC_LEN,
                                  scope="capsule2")
            capsule3 = self.route(capsule2, cfg.CAPSULE3_VEC_NUM, cfg.CAPSULE3_VEC_LEN,
                                  scope="capsule3")
            '''
            capsule3 = self.route(capsule1, cfg.CAPSULE3_VEC_NUM, cfg.CAPSULE3_VEC_LEN,
                                  scope="capsule3")
            self.output_capsule = capsule3
            self.output_len = tf.sqrt(tf.reduce_sum(tf.square(capsule3),axis=2) + epsilon)
            #The length of each vector indicate the possiblity of the category for each capsule
            if is_training:
                mask = tf.one_hot(self.input_label, depth=cfg.CATEGORY_NUM)
            else:
                mask = tf.argmax(self.output_len, axis=1)
                mask = tf.one_hot(mask, depth=cfg.CATEGORY_NUM)
            mask = tf.reshape(mask,[-1,cfg.CAPSULE3_VEC_NUM,1])
            mask = tf.cast(mask,tf.float32)
            masked_capsule = tf.matmul(capsule3,mask,transpose_a=True)
            masked_capsule = tf.squeeze(masked_capsule)

            with tf.variable_scope('Decoder'):
                fc1 = slim.fully_connected(masked_capsule, 512, scope='fc1')
                fc2 = slim.fully_connected(masked_capsule, 1024, scope='fc2')
                self.decoded = slim.fully_connected(masked_capsule, cfg.IMAGE_SIZE*cfg.IMAGE_SIZE*3, scope='decode',activation_fn=tf.sigmoid)


    def route(self,input,output_vector_num,output_vector_len,scope):
        input_vec_num = input.shape[1].value
        input_vec_len = input.shape[2].value
        with tf.variable_scope(scope):
            W = tf.get_variable('Weight', shape=(1, input_vec_num, output_vector_num, input_vec_len, output_vector_len),
                                dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(0.0, 0.01))
            W = tf.tile(W, [cfg.BATCH_SIZE, 1, 1, 1, 1])
            input = tf.reshape(input,[cfg.BATCH_SIZE,input_vec_num,1,input_vec_len,1])
            input = tf.tile(input,[1,1,output_vector_num,1,1])
            # from input vector to output vector,
            # each input vector will generate output_vector_num output vector,
            # total input_vec_num input vectors generate input_vec_num * output_vector_num output vector,
            # the output vector lenght is output_vector_len
            # W_mul_input shape should be [batch, input_vec_num, output_vector_num, output_vector_len,1]
            # the last dimension keeps there as we will need dot product later
            output_vectors = tf.matmul(W,input,transpose_a=True)# from input vec to output vec

            #routing_mat, the last 2 dimension is just for calculation convenience
            routing_mat = tf.zeros([cfg.BATCH_SIZE, input_vec_num, output_vector_num,1,1])
            u_stopped = tf.stop_gradient(output_vectors, name='stop_gradient')

            for iter in range(cfg.ROUTE_ITER_COUNT):
                with tf.variable_scope('iter_' + str(iter)):
                    #weight_routing_mat is the weight(possiblity) from each input vector routing to output_vector_num output vectors
                    weight_routing_mat = tf.nn.softmax(routing_mat, dim=2)
                    if iter < cfg.ROUTE_ITER_COUNT - 1:
                        output = u_stopped * weight_routing_mat

                        output = tf.reduce_sum(output, axis=1, keep_dims=True)
                        output = self.squash(output)
                        #in order to make vector dot multiplication, expand the last dimension
                        output = tf.reshape(output,[cfg.BATCH_SIZE,1,output_vector_num,output_vector_len,1])

                        v_tile = tf.tile(output, [1, input_vec_num, 1, 1, 1])
                        #u_dot_v dimension is [batch,input_vector_num,output_vector_num,1,1]
                        #Add it to routing_mat
                        u_dot_v = tf.matmul(u_stopped, v_tile, transpose_a=True)
                        routing_mat = routing_mat + u_dot_v
                    #The last iteration
                    elif iter == cfg.ROUTE_ITER_COUNT - 1:
                        output_vec = output_vectors * weight_routing_mat
                        output_vec = tf.reduce_sum(output_vec, axis=1)
                        output_vec = tf.squeeze(output_vec)
                        output_vec = self.squash(output_vec)
            #output_vec shape [batch,output_vector_num,output_vector_len]
            return output_vec

    def squash(self, vec):
        vec_L2 = tf.reduce_sum(tf.square(vec), -1, keep_dims=True)
        scalar_factor = vec_L2 / (1 + vec_L2)
        unit_vec = vec  / tf.sqrt(vec_L2 + epsilon)
        vec_squashed = scalar_factor * unit_vec
        return vec_squashed

    def summary(self):
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('reconstruct_loss', self.reconstruct_loss)
        tf.summary.scalar('margin_loss', self.margin_loss)

        correct_prediction = tf.equal(tf.cast(self.input_label,tf.int64), tf.argmax(self.output_len, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))