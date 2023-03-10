# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import tensorflow as tf
from datetime import datetime 
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from model import NCR
from utils import checkmate as cm
from utils import data_helpers as dh


TRAIN_OR_RESTORE = 'T' 

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()).replace(':', '_'))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()).replace(':', '_'))

#number = str(sys.argv[1])
tf.compat.v1.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.compat.v1.flags.DEFINE_float("norm_ratio", 10, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 128, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 512 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 3, "Number of epochs to train for.")


tf.compat.v1.flags.DEFINE_integer("decay_steps", 2, "how many steps before decay learning rate. (default: 500)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.3, "Rate of decay for learning rate. (default: 0.95)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


def train():
    """Training model."""

    logger.info("Loading data...")

    logger.info("Training data processing...")
    train_students = np.load("data/train.npy", allow_pickle=True)
    
    logger.info("Validation data processing...")
    valid_students = np.load("data/valid.npy", allow_pickle=True)

    
    print(np.shape(train_students))
    max_num_steps = int(sys.argv[1])
    max_num_skills = 1175

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            ncr = NCR(
                batch_size = FLAGS.batch_size,
                num_steps = max_num_steps,
                num_skills = max_num_skills,
                hidden_size = FLAGS.hidden_size, 
                )
            

            # Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=ncr.global_step, decay_steps=(len(train_students)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                
                train_op = optimizer.minimize(ncr.loss, global_step=ncr.global_step, name="train_op")

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.compat.v1.summary.scalar("loss", ncr.loss)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=1, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load ncr model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())



            current_step = sess.run(ncr.global_step)

            def train_step(student, his_pro, his_kc, his_a, target_id, target_kc, target_a):
                """A single training step"""
                
                feed_dict = {
                    ncr.student: student,
                    ncr.his_pro: his_pro,
                    ncr.his_kc: his_kc,
                    ncr.his_a: his_a,
                    ncr.target_id: target_id,
                    ncr.target_kc: target_kc,
                    ncr.target_a: target_a,
                    ncr.dropout_keep_prob: FLAGS.keep_prob,
                    ncr.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, ncr.global_step, train_summary_op, ncr.pred, ncr.loss], feed_dict)

                if step % 100 == 0:
                    logger.info("step {0}: loss {1:g} ".format(step,loss))
                train_summary_writer.add_summary(summaries, step)
                return pred

            def validation_step(student, his_pro, his_kc, his_a, target_id, target_kc, target_a):
                """Evaluates model on a validation set"""

                feed_dict = {
                    ncr.student: student,
                    ncr.his_pro: his_pro,
                    ncr.his_kc: his_kc,
                    ncr.his_a: his_a,
                    ncr.target_id: target_id,
                    ncr.target_kc: target_kc,
                    ncr.target_a: target_a,
                    ncr.dropout_keep_prob: 0.0,
                    ncr.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [ncr.global_step, validation_summary_op, ncr.pred, ncr.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                
                return pred
            
            run_time = []
            m_rmse = 1
            m_r2 = 0
            m_acc = 0
            m_auc = 0
            for iii in range(FLAGS.epochs):
                np.random.seed(iii*100)
                np.random.shuffle(train_students)
                a=datetime.now()
                data_size = len(train_students)
                index = 0
                actual_labels = []
                pred_labels = []
                while(index+FLAGS.batch_size <= data_size):
                    student = np.zeros((FLAGS.batch_size, ))
                    his_pro = np.zeros((FLAGS.batch_size, max_num_steps))
                    his_kc = np.zeros((FLAGS.batch_size, max_num_steps))
                    his_a = np.zeros((FLAGS.batch_size, max_num_steps))
                    target_id = np.zeros((FLAGS.batch_size, ))
                    target_kc = np.zeros((FLAGS.batch_size, ))
                    target_a = []
                    for i in range(FLAGS.batch_size):
                        item = train_students[index+i]
                    #    print(item[4])
                        student[i] = item[0]
                        his_pro[i,:] = item[4][-max_num_steps:]
                        his_kc[i,:] = item[5][-max_num_steps:]
                        his_a[i,:] = item[6][-max_num_steps:]
                        target_id[i] = item[1]
                        target_kc[i] = item[2]
  
                        target_a.append(item[3])

                        actual_labels.append(item[3])

                    index += FLAGS.batch_size
                    
                    pred = train_step(student, his_pro, his_kc, his_a, target_id, target_kc, target_a)
                    for p in pred:
                        pred_labels.append(p)
                    current_step = tf.compat.v1.train.global_step(sess, ncr.global_step)
                
                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)
                rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                auc = metrics.roc_auc_score(actual_labels, pred_labels)
                #calculate r^2
                r2 = r2_score(actual_labels, pred_labels)
                
                pred_score = np.greater_equal(pred_labels,0.5) 
                pred_score = pred_score.astype(int)
                pred_score = np.equal(actual_labels, pred_score)
                acc = np.mean(pred_score.astype(int))

                logger.info("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g}".format((iii +1),rmse, auc, r2, acc))

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")
                    
                    data_size = len(valid_students)
                    index = 0
                    actual_labels = []
                    pred_labels = []
                    while(index+FLAGS.batch_size <= data_size):
                        student = np.zeros((FLAGS.batch_size, ))
                        his_pro = np.zeros((FLAGS.batch_size, max_num_steps))
                        his_kc = np.zeros((FLAGS.batch_size, max_num_steps))
                        his_a = np.zeros((FLAGS.batch_size, max_num_steps))
                        target_id = np.zeros((FLAGS.batch_size, ))
                        target_kc = np.zeros((FLAGS.batch_size, ))
                        target_a = []
                        for i in range(FLAGS.batch_size):
                            item = valid_students[index+i]
                            student[i] = item[0]
                            his_pro[i,:] = item[4][-max_num_steps:]
                            his_kc[i,:] = item[5][-max_num_steps:]
                            his_a[i,:] = item[6][-max_num_steps:]
                            target_id[i] = item[1]
                            target_kc[i] = item[2]
    
                            target_a.append(item[3])
                            actual_labels.append(item[3])
                        index += FLAGS.batch_size
                        pred  = validation_step(student, his_pro, his_kc, his_a, target_id, target_kc, target_a)
                        for p in pred:
                            pred_labels.append(p)
                    

                    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                    auc = metrics.roc_auc_score(actual_labels, pred_labels)
                    #calculate r^2
                    r2 = r2_score(actual_labels, pred_labels)
                    
                    pred_score = np.greater_equal(pred_labels,0.5) 
                    pred_score = pred_score.astype(int)
                    pred_score = np.equal(actual_labels, pred_score)
                    acc = np.mean(pred_score.astype(int))

                    logger.info("VALIDATION {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g} ".format((iii +1)/FLAGS.evaluation_interval,rmse, auc, r2, acc))
                    

                    if rmse < m_rmse:
                        m_rmse = rmse
                    if auc > m_auc:
                        m_auc = auc
                    if acc > m_acc:
                        m_acc = acc
                    if r2 > m_r2:
                        m_r2 = r2
                    best_saver.handle(auc, sess, current_step)
                if ((iii+1) % FLAGS.checkpoint_every == 0):
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))

                logger.info("Epoch {0} has finished!".format(iii + 1))
            
            logger.info("running time analysis: epoch{0}, avg_time{1}".format(len(run_time), np.mean(run_time)))
            logger.info("max: rmse {0:g}  auc {1:g}  r2 {2:g}   acc{3:g} ".format(m_rmse, m_auc, m_r2, m_acc))
            with open('results.txt', 'a') as fi:
                fi.write("max: rmse\t{0:g}\tauc\t{1:g}\tr2\t{2:g}\tacc\t{3:g}".format(m_rmse, m_auc, m_r2, m_acc))
                fi.write('\n')


    logger.info("Done.")


if __name__ == '__main__':
    train()
