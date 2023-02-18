# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf

def weight_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()(shape) , trainable=True, name=name)  
    return initial
def bias_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()(shape) , trainable=True, name=name) 
    return initial

def knowledge_representation_matrix(
        queries, 
        keys,
        num_units,
        dropout_prob,
        is_training,
        scope = 'krm'):
    
    Q = tf.compat.v1.layers.dense(queries, num_units) 
    K = tf.compat.v1.layers.dense(keys, num_units) 
    V = tf.compat.v1.layers.dense(keys, num_units)#, activation=tf.nn.relu) 
    
 #   Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
 #   K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) 
 #   V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
    
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    
    outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
    
    outputs = tf.nn.softmax(outputs)

    outputs = tf.nn.dropout(outputs, dropout_prob)
    outputs = tf.matmul(outputs, V)
    
  #  outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    
    outputs += Q
    outputs =  tf.compat.v1.layers.batch_normalization(outputs, momentum=0.9, epsilon=1e-5, training=is_training)#
    
    return outputs




def multi_span(x, drop_rate, is_training):
  #  x = tf.layers.dense(x, 512)
    with tf.name_scope("krm_{}".format(6)):
        x = knowledge_representation_matrix(queries = x, 
                                                    keys = x, 
                                                    num_units = 128, 
                                                    dropout_prob = drop_rate,
                                                    is_training = is_training,
                                                    )
        x = tf.compat.v1.layers.dense(x, 128, activation=tf.nn.relu) #+ x
        
    print('x output: ', np.shape(x))

    
    return x
class NCR(object):

    def __init__(self, batch_size, num_steps, num_skills, hidden_size):
        


        self.student = tf.compat.v1.placeholder(tf.int32, [batch_size, ], name="student")
        self.his_pro = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="his_pro")
        self.his_kc = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="his_kc")
        self.his_a = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="his_a")
        self.target_id = tf.compat.v1.placeholder(tf.int32, [batch_size,], name="target_id")
        self.target_kc = tf.compat.v1.placeholder(tf.int32, [batch_size,], name="target_kc")
        self.target_a = tf.compat.v1.placeholder(tf.float32, [batch_size], name="target_a")

        
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.compat.v1.keras.initializers.glorot_uniform()
        
        # student embedding
        self.student_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([18066, hidden_size]),dtype=tf.float32, trainable=True, name = 'student_w')
        student_e =  tf.nn.embedding_lookup(self.student_w, self.student)

        #answer embedding
        self.a_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([2, hidden_size]),dtype=tf.float32, trainable=True, name = 'a_w')
        his_a_e =  tf.nn.embedding_lookup(self.a_w, self.his_a)

        #pro embedding
        self.pro_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'pro_w')
        his_pro_e =  tf.nn.embedding_lookup(self.pro_w, self.his_pro)
        target_id_e =  tf.nn.embedding_lookup(self.pro_w, self.target_id)

        #kc embedding
        self.kc_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([num_skills, hidden_size]),dtype=tf.float32, trainable=True, name = 'kc_w')
        his_kc_e =  tf.nn.embedding_lookup(self.kc_w, self.his_kc)
        target_kc_e =  tf.nn.embedding_lookup(self.kc_w, self.target_kc)

        w_l = weight_variable([hidden_size, 2*hidden_size], name = 'w_l', training = self.is_training )
        b_l = bias_variable([hidden_size],  name='b_l', training = self.is_training )

        combine_his = tf.concat([his_pro_e, his_kc_e],axis = -1)
        combine_his =  tf.matmul(combine_his,  tf.transpose(w_l, [1,0])+b_l)
        #combine_his = tf.compat.v1.layers.dense(combine_his, units = hidden_size)

        # combine_plus_a = tf.concat([combine_his, his_a_e],axis = -1)
       # combine_plus_a = tf.compat.v1.layers.dense(combine_plus_a, units = hidden_size)
        w_r = weight_variable([hidden_size, hidden_size], name = 'w_r', training = self.is_training )
        b_r = bias_variable([hidden_size],  name='b_r', training = self.is_training )
        w_w = weight_variable([hidden_size, hidden_size], name = 'w_w', training = self.is_training )
        b_w = bias_variable([hidden_size],  name='b_w', training = self.is_training )
        his_a_e = tf.expand_dims(self.his_a, -1)
        his_a_e = tf.cast(his_a_e, tf.float32)
        ri = tf.matmul(combine_his,  tf.transpose(w_r, [1,0])+b_r)
        wr = tf.matmul(combine_his,  tf.transpose(w_w, [1,0])+b_w)
        combine_plus_a = his_a_e*ri + (1-his_a_e)*wr


        combine_traget = tf.concat([target_id_e, target_kc_e],axis = -1)
        combine_traget = tf.matmul(combine_traget,  tf.transpose(w_l, [1,0])+b_l)
        #combine_traget = tf.compat.v1.layers.dense(combine_traget, units = hidden_size)

        att = tf.matmul(combine_his, tf.expand_dims(combine_traget, axis = -1))
        att = tf.reduce_mean(att, axis = 1)
        att= tf.sigmoid(att)
       # att = tf.nn.softmax(att,axis = -1)


        tmp_vector =  multi_span(combine_plus_a, self.dropout_keep_prob,  self.is_training)
      #  tmp_vector = tf.compat.v1.layers.dense(tmp_vector, units = hidden_size)
        tmp_vector = tf.reduce_mean(tmp_vector, axis = 1)

        
        gate = tf.sigmoid(tf.compat.v1.layers.dense(att, units = hidden_size))
        output = tmp_vector * gate +  (1-gate)*student_e #tf.concat([tmp_vector, student_e], axis = -1) #tmp_vector #
       # output = tmp_vector+student_e
        sent_vector = output * combine_traget #tf.concat([output, combine_traget], axis = -1)# 
        #sent_vector = student_e * combine_traget
        sent_vector = tf.compat.v1.layers.dense(sent_vector, units = hidden_size)
        finals = tf.compat.v1.layers.dense(sent_vector, units = 1)
        print(np.shape(finals))

        finals = tf.reshape(finals, [-1])
        
        #make prediction
        self.pred = tf.sigmoid(finals, name="pred")

        # loss function
        losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=finals, labels=self.target_a), name="losses")

        l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.compat.v1.trainable_variables()],
                                 name="l2_losses") * 0.000001
        self.loss = tf.add(losses, l2_losses, name="loss")
        
        self.cost = self.loss