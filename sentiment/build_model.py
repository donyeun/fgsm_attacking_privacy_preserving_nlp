#! /usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import time
import datetime
# import cPickle
import pickle

import tensorflow as tf
import numpy as np

import data_helpers
from text_cnn import TextCNN, CopiedClassifier
from art.estimators.classification import TensorFlowV2Classifier

from tensorflow.contrib import learn
import sklearn.metrics
import csv


tf.reset_default_graph()
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate alpha")
tf.flags.DEFINE_float("lr_lambda", 0.5, "lr lambda")

tf.flags.DEFINE_string("mode", "attacking_privacy", "baseline, privacy_preserving, attacking_privacy")

tf.flags.DEFINE_string("baseline_attr", "location", "location, gender, age")
tf.flags.DEFINE_string("privacy_preserving_attr", "location", "all, location, gender, age")
tf.flags.DEFINE_string("attacking_privacy_preserving_attr", "location", "location, gender, age")

#  parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200 --> 100 by lrank)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200 --> 100 by lrank)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



# Data Preparation
# ==================================================

np.random.seed(1001)
# Load data
FLAGS = tf.flags.FLAGS

max_doc_length, vocab_size, \
    text_x, locations, genders, ages, ratings, \
    emb_size, w_embs = data_helpers.load_trustpilot()


FLAGS.embedding_dim = emb_size
FLAGS.batch_size
print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.iteritems()):
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value) )
print("")


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      intra_op_parallelism_threads=2,
      inter_op_parallelism_threads=2)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=text_x.shape[1],
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            num_ratings=ratings.shape[1],
            num_locations=locations.shape[1],
            num_genders=genders.shape[1],
            num_ages=ages.shape[1],
            hidden_size=300,
            # mode=FLAGS.mode,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            )
        
        lanjutan_cnn_copied = CopiedClassifier(
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_locations=locations.shape[1],
                hidden_size=300,
            )

        # Define Training procedure
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        adv_lambda = tf.placeholder(tf.float32, shape=[], name="adversarial_lambda")

        
        # phase = tf.Variable("train", name="phase", trainable=False)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        all_var_list = tf.trainable_variables()

        optimizer_n = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.rating_loss,
                global_step=global_step
                )

        var_loca = [var for var in all_var_list if 'location' in var.name]
        var_gend = [var for var in all_var_list if 'gender' in var.name]
        var_age  = [var for var in all_var_list if 'age' in var.name]
        
        print("&"*30)
        print(len(var_loca))
        print(len(var_gend))
        print(len(var_age))
        assert( len(var_loca) == 4 and len(var_gend) == 4 and len(var_age) == 4 )

        if FLAGS.privacy_preserving_attr == "age":
            var_d = var_age
            disc_loss = cnn.age_loss
        elif FLAGS.privacy_preserving_attr == "gender":
            var_d = var_gend
            disc_loss = cnn.gender_loss
        elif FLAGS.privacy_preserving_attr == "location":
            var_d = var_loca
            disc_loss = cnn.location_loss
        elif FLAGS.privacy_preserving_attr == "all":
            var_d = var_loca + var_gend + var_age
            disc_loss = cnn.location_loss + cnn.gender_loss + cnn.age_loss  
        else:
            assert(False)

        optimizer_d = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                adv_lambda * disc_loss,
                var_list=var_d
                )

        #define attackers opts
        # Dony: attacker lebih ke baseline performance utk private attr
        var_attack_l = [var for var in all_var_list if 'l_attacker' in var.name]
        optimizer_attack_l = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.location_attacker_loss,
                var_list=var_attack_l,
                global_step=global_step
                )

        var_attack_g = [var for var in all_var_list if 'g_attacker' in var.name]
        optimizer_attack_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.gender_attacker_loss,
                var_list=var_attack_g,
                global_step=global_step
                )

        var_attack_a = [var for var in all_var_list if 'a_attacker' in var.name]
        optimizer_attack_a = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.age_attacker_loss,
                var_list=var_attack_a,
                global_step=global_step
                )

        print("&"*30)
        print(len(var_attack_l))
        print(len(var_attack_g))
        print(len(var_attack_a))
        print(var_attack_l)
        # assert( len(var_attack_l) == 4 and len(var_attack_g) == 4 and len(var_attack_a) == 4 )

        #representation opt
        var_g = [var for var in all_var_list if var not in (var_loca + var_gend + var_age + var_attack_l + var_attack_g + var_attack_a)]
        optimizer_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.rating_loss - adv_lambda * disc_loss,
                var_list=var_g,
                global_step=global_step
                )

        def incomp_acc(logit, label):
            c_cor = 0.
            c_total = 0.
            gt = np.argmax(label, axis=1)
            for i in range( len( logit )):
                if gt[i] == 0 and np.abs(label[i][gt[i]]) < 1e-5:
                    continue
                c_total += 1.
                if gt[i] == logit[i]:
                    c_cor += 1.
            if np.abs(c_total) < 1e-5:
                return -1
            return c_cor / c_total

        def fgsm_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=4, current_mode="with_fgm", attack_attr="", print_result=True, injected_h=None, return_h_drop=False):
        # def fgsm_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=4, current_mode="without_fgm", attack_attr="", print_result=True, injected_h=None, return_h_drop=False):
            """1
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              adv_lambda: 0.,
              cnn.dropout_keep_prob: 1.,
              cnn.current_mode: current_mode,
              cnn.attack_attr: attack_attr,
            }
            
            # if injected_h is not None:
            #     feed_dict[cnn.injected_h] = injected_h

            step, l_rat, a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_attacker_loss, cnn.location_attacker_pred,
                cnn.gender_attacker_loss, cnn.gender_attacker_pred,
                cnn.age_attacker_loss, cnn.age_attacker_pred],
                feed_dict)

                # rating_loss, rating_accuracy, rating_pred, rating_score, \
                # location_attacker_loss, location_attacker_accuracy, location_attacker_pred, location_attacker_score, \
                # gender_attacker_loss, gender_attacker_accuracy, gender_attacker_pred, gender_attacker_score, \
                # age_attacker_loss, age_attacker_accuracy, age_attacker_pred, age_attacker_score
            # print('lossss shouldnt be zero below')
            # print(l_rat)
            # print(l_loc)
            # print(l_gen)
            # print('rating~~~~')
            # print(p_rat)
            # # print(np.argmax(p_rat, axis=1))
            # print(batch_rat)
            # # print(np.argmax(batch_rat, axis=1))
            # print('masalah')
            # print(p_loc)
            # # print(np.argmax(p_loc, axis=1))
            # print(batch_loc)
            # # print(np.argmax(batch_loc, axis=1))

            # # p_loc = np.argmax(p_loc, axis=1)
            # # batch_loc = np.argmax(batch_loc, axis=1)
            # print('more')
            # print(p_gen)
            # print(p_age)

            a_loc = incomp_acc(p_loc, batch_loc)
            a_gen = incomp_acc(p_gen, batch_gen)
            a_age = incomp_acc(p_age, batch_age)
            f1_rat = sklearn.metrics.f1_score(
                y_true = np.argmax(batch_rat, axis=1),
                y_pred = p_rat,
                average= 'micro'
            )
            f1_rat = sklearn.metrics.f1_score(y_true=np.argmax(batch_rat, axis=1), y_pred=p_rat, average='micro')

            if print_result:
                print("9\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
                    step,
                    l_rat, a_rat, f1_rat,
                    l_loc, a_loc,
                    l_gen, a_gen,
                    l_age, a_age )
                )
            
            # y_true_rat = np.argmax(batch_rat, axis=1)
            # y_pred_rat = p_rat

            y = {
                'true' : {
                    'rat' : np.argmax(batch_rat, axis=1),
                    'loc' : np.argmax(batch_loc, axis=1),
                    'gen' : np.argmax(batch_gen, axis=1),
                    'age' : np.argmax(batch_age, axis=1),
                },
                'pred' : {
                    'rat' : p_rat,
                    'loc' : p_loc,
                    'gen' : p_gen,
                    'age' : p_age,
                },
            }
            # return a_rat, p_loc, p_gen, p_age
            if return_h_drop:
                return a_rat, a_loc, a_gen, a_age, y, original_h_drop,
            return a_rat, a_loc, a_gen, a_age, y


        def dev_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=1):
            """
            Evaluates model on a dev set
            """
            # if data_id == 1:
            #     current_phase = "dev"
            # else:
            #     current_phase = "test"

            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              adv_lambda: 0.,
              cnn.dropout_keep_prob: 1.,
            #   cnn.phase: current_phase,
            }


            step, l_rat, a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
            # step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)

            # print("~"*100)
            # print(cnn.phase)
            # print("~"*100)

            # time_str = datetime.datetime.now().isoformat()
            ''' data_id: train:0 dev:1 test:2 '''
            a_loc = incomp_acc(p_loc, batch_loc)
            a_gen = incomp_acc(p_gen, batch_gen)
            a_age = incomp_acc(p_age, batch_age)
            f1_rat = sklearn.metrics.f1_score(
                y_true = np.argmax(batch_rat, axis=1),
                y_pred = p_rat,
                average= 'micro'
            )
            print("{}\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
            # print("{}\t{}|rate: \t{:0.5g}\t{:0.5g}\t{:0.5g}|loc: \t{:0.5g}\t{:0.5g}|gen: \t{:0.5g}\t{:0.5g}|age: \t{:0.5g}\t{:0.5g}".format(
                data_id, step,
                l_rat, a_rat, f1_rat,
                l_loc, a_loc,
                l_gen, a_gen,
                l_age, a_age )
            )

            y_true_rat = np.argmax(batch_rat, axis=1)
            y_pred_rat = p_rat

            y = {
                'true' : {
                    'rat' : np.argmax(batch_rat, axis=1),
                    'loc' : np.argmax(batch_loc, axis=1),
                    'gen' : np.argmax(batch_gen, axis=1),
                    'age' : np.argmax(batch_age, axis=1),
                },
                'pred' : {
                    'rat' : p_rat,
                    'loc' : p_loc,
                    'gen' : p_gen,
                    'age' : p_age,
                },
            }
            return l_rat, a_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y

        def train_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer, adv_lam=0, lr = 1e-4):
            """1
            Dony: train non-attacked step
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              learning_rate: lr,
              adv_lambda: adv_lam,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            #   cnn.phase: "train",
            }

            _, step, l_rat, a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [optimizer, global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)
            
            # print("~"*100)
            # print(current_phase)
            # print("~"*100)
            
            # time_str = datetime.datetime.now().isoformat()
            f1_rat = sklearn.metrics.f1_score(y_true=np.argmax(batch_rat, axis=1), y_pred=p_rat, average='micro')
            print("0\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
            # print("0\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
                step, l_rat - adv_lam * (l_loc + l_gen + l_age),
                l_rat, a_rat, f1_rat,
                l_loc, incomp_acc(p_loc, batch_loc),
                l_gen, incomp_acc(p_gen, batch_gen),
                l_age, incomp_acc(p_age, batch_age)
                )
            )

        def train_attacker_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizers, adv_lam=0, lr = 1e-4, attack_attr="all"):
            """1
            Dony: train attacker step
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              learning_rate: lr,
              adv_lambda: adv_lam,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            #   cnn.phase: "train",
            }
            
            # optimizer_attack_l = optimizers["location"]
            # optimizer_attack_g = optimizers["gender"]
            # optimizer_attack_a = optimizers["age"]
            

            if (attack_attr == "all"):
                _, step, l_rat, a_rat, l_loc, p_loc = sess.run(
                    [optimizer_attack_l, global_step,
                    cnn.rating_loss, cnn.rating_accuracy,
                    cnn.location_attacker_loss, cnn.location_attacker_pred],
                    feed_dict)
                _, step, l_gen, p_gen = sess.run(
                    [optimizer_attack_g, global_step,
                    cnn.gender_attacker_loss, cnn.gender_attacker_pred],
                    feed_dict)
                _, step, l_age, p_age = sess.run(
                    [optimizer_attack_a, global_step,
                    cnn.age_attacker_loss, cnn.age_attacker_pred],
                    feed_dict)
            elif (attack_attr == "location"):
                _, step, l_rat, a_rat, l_loc, p_loc = sess.run(
                    [optimizer_attack_l, global_step,
                    cnn.rating_loss, cnn.rating_accuracy,
                    cnn.location_attacker_loss, cnn.location_attacker_pred],
                    feed_dict)
            elif (attack_attr == "gender"):
                _, step, l_rat, a_rat, l_gen, p_gen = sess.run(
                    [optimizer_attack_g, global_step,
                    cnn.rating_loss, cnn.rating_accuracy,
                    cnn.gender_attacker_loss, cnn.gender_attacker_pred],
                    feed_dict)
            elif (attack_attr == "age"):
                _, step, l_rat, a_rat, l_age, p_age = sess.run(
                    [optimizer_attack_a, global_step,
                    cnn.rating_loss, cnn.rating_accuracy,
                    cnn.age_attacker_loss, cnn.age_attacker_pred],
                    feed_dict)
            
            # print the result
            if (attack_attr == "all"):
                print("3\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                    
                    step,
                    l_rat, a_rat,
                    l_loc, incomp_acc(p_loc, batch_loc),
                    l_gen, incomp_acc(p_gen, batch_gen),
                    l_age, incomp_acc(p_age, batch_age)
                    )
                )
            elif (attack_attr == "location"):
                print("3\t{}\t{:g}\t{:g}\t{:g}\t{:g}".format(    
                    step,
                    l_rat, a_rat,
                    l_loc, incomp_acc(p_loc, batch_loc),
                    )
                )
            elif (attack_attr == "gender"):
                print("3\t{}\t{:g}\t{:g}\t{:g}\t{:g}".format(    
                    step,
                    l_rat, a_rat,
                    l_gen, incomp_acc(p_gen, batch_gen),
                    )
                )
            elif (attack_attr == "age"):
                print("3\t{}\t{:g}\t{:g}\t{:g}\t{:g}".format(    
                    step,
                    l_rat, a_rat,
                    l_age, incomp_acc(p_age, batch_age)
                    )
                )

        def dev_attacker_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=4, current_mode="without_fgm", attack_attr="", print_result=True, injected_h=None, return_h_drop=False):
            """1
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              adv_lambda: 0.,
              cnn.dropout_keep_prob: 1.,
              cnn.current_mode: current_mode,
              cnn.attack_attr: attack_attr,
            }
            
            if injected_h is not None:
                feed_dict[cnn.injected_h] = injected_h

            step, original_h_drop, l_rat,a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [global_step, cnn.original_h_drop,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_attacker_loss, cnn.location_attacker_pred,
                cnn.gender_attacker_loss, cnn.gender_attacker_pred,
                cnn.age_attacker_loss, cnn.age_attacker_pred],
                feed_dict)


            # print('rating##2')
            # print(p_rat)
            # print(batch_rat)
            # # print(np.argmax(batch_rat, axis=1))
            # print('masalah')
            # print(p_loc)
            # print(batch_loc)
            # # print(np.argmax(batch_loc, axis=1))

            
            a_loc = incomp_acc(p_loc, batch_loc)
            a_gen = incomp_acc(p_gen, batch_gen)
            a_age = incomp_acc(p_age, batch_age)
            f1_rat = sklearn.metrics.f1_score(
                y_true = np.argmax(batch_rat, axis=1),
                y_pred = p_rat,
                average= 'micro'
            )
            f1_rat = sklearn.metrics.f1_score(y_true=np.argmax(batch_rat, axis=1), y_pred=p_rat, average='micro')

            if print_result:
                print("{}\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
                    data_id, step,
                    l_rat, a_rat, f1_rat,
                    l_loc, a_loc,
                    l_gen, a_gen,
                    l_age, a_age )
                )
            
            # y_true_rat = np.argmax(batch_rat, axis=1)
            # y_pred_rat = p_rat

            y = {
                'true' : {
                    'rat' : np.argmax(batch_rat, axis=1),
                    'loc' : np.argmax(batch_loc, axis=1),
                    'gen' : np.argmax(batch_gen, axis=1),
                    'age' : np.argmax(batch_age, axis=1),
                },
                'pred' : {
                    'rat' : p_rat,
                    'loc' : p_loc,
                    'gen' : p_gen,
                    'age' : p_age,
                },
            }
            # return a_rat, p_loc, p_gen, p_age
            if return_h_drop:
                return a_rat, a_loc, a_gen, a_age, y, original_h_drop,
            return a_rat, a_loc, a_gen, a_age, y
            

        writer = csv.writer(open('output_y.csv', 'w'), delimiter='\t')
        writer.writerow(['true rat', 'pred rat', 'true loc', 'pred loc', 'true gen', 'pred gen', 'true age', 'pred age',])

        #data_split
        x_train, loc_train, gen_train, age_train, rat_train, \
        x_dev, loc_dev, gen_dev, age_dev, rat_dev, \
        x_test, loc_test, gen_test, age_test, rat_test = data_helpers.data_split_train_dev_test(
            [text_x, locations, genders, ages, ratings],
            shuffle=True
            )

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        train_writer = tf.summary.FileWriter("./logs", sess.graph)
        # print('train_writer at ', FLAGS.log_dir)
        sess.run(tf.global_variables_initializer())
        print("$"*30)
        print(print(tf.global_variables()))
        print("$"*30)
        sess.run(cnn.emb_W.assign(w_embs))

        # Training loop. For each batch...
        data_size = len(x_train)
        print('xxxxx training data length')
        print(data_size)
        print('xxxxx testing data length')
        print(len(x_test))
        best_dev_score = 0.
        bset_test_score = 0.
        best_dev_f1_score = 0.
        best_test_f1_score = 0.
        best_test = {}
        y_test = {}
        y_test['true'] = {}
        y_test['true']['rat'] = np.array([])
        y_test['true']['loc'] = np.array([])
        y_test['true']['gen'] = np.array([])
        y_test['true']['age'] = np.array([])

        y_test['pred'] = {}
        y_test['pred']['rat'] = np.array([])
        y_test['pred']['loc'] = np.array([])
        y_test['pred']['gen'] = np.array([])
        y_test['pred']['age'] = np.array([])

        training_batch_iter = data_helpers.batch_iter(
            data=[x_train, loc_train, gen_train, age_train, rat_train],
            batch_size=FLAGS.batch_size,
            is_balance=True,
            bal_label_index=4,
            )
        training_learning_rate = FLAGS.learning_rate     

        # create fgsm batch from the test set, with batch_size=1 (1 example per fgsm attack)
        fgsm_batch_iter = data_helpers.batch_iter(
            data=[x_test, loc_test, gen_test, age_test, rat_test],
            batch_size=1,
            is_balance=True,
            bal_label_index=4,
        )   
        test_size = len(x_test)

        # dony: non attack training
        # set_phase = phase.assign("train")


            # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)

            # cnn.set_phase(phase='train')
        if FLAGS.mode == "baseline":
            for _ in range(FLAGS.num_epochs * data_size // FLAGS.batch_size):
                current_step = tf.train.global_step(sess, global_step)
                # lr_lamb = (current_step / 100) / 1000.0
                
                # if lr_lamb > FLAGS.lr_lambda:
                lr_lamb = FLAGS.lr_lambda

                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()


                train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_n, adv_lam=lr_lamb, lr=training_learning_rate)
  
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    loss_rat, acc_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y_batch = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                    
                    test_loss, test_acc, test_f1, test_l_loc, test_a_loc, test_l_gen, test_a_gen, test_l_age, test_a_age, y_test_batch = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)

            print("training attack")
            for _ in range(FLAGS.num_epochs * data_size // FLAGS.batch_size):
                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()
                train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_attack_g, adv_lam=lr_lamb, lr=training_learning_rate * 0.1, attack_attr=FLAGS.baseline_attr)
            #     train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate * 0.1 )

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    test_score, a_l, a_g, a_a, y = dev_attacker_step( x_test, loc_test, gen_test, age_test, rat_test, 5)

        elif FLAGS.mode == "privacy_preserving":
            for _ in range(FLAGS.num_epochs * data_size // FLAGS.batch_size):
                current_step = tf.train.global_step(sess, global_step)
                # lr_lamb = (current_step / 100) / 1000.0
                
                # if lr_lamb > FLAGS.lr_lambda:
                lr_lamb = FLAGS.lr_lambda

                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()

                train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)
                # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=0.05, lr=training_learning_rate)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    # print("\nEvaluation:")
                    # DONY: using dev set (data_id=1)
                    loss_rat, acc_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y_batch = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                    
                    # dony: this line need to be separated from the training phase
                    # dony: using test set (data_id=2)
                    # cnn.set_phase(phase='eval')
                    test_loss, test_acc, test_f1, test_l_loc, test_a_loc, test_l_gen, test_a_gen, test_l_age, test_a_age, y_test_batch = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)

            print("training attack")
            for _ in range(FLAGS.num_epochs * data_size // FLAGS.batch_size):
                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()
                train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_attack_l, adv_lam=lr_lamb, lr=training_learning_rate * 0.1 )

                # ==============================================================================================================
                # <BEGIN ORIGINAL
                # evaluaate as usual, for the entire test set
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    test_score, a_l, a_g, a_a, y = dev_attacker_step( x_test, loc_test, gen_test, age_test, rat_test, 5)
                # END>
                # ==============================================================================================================
                # <BEGIN MODIFIED
                # current_step = tf.train.global_step(sess, global_step)
                    correct_rat = 0
                    correct_loc = 0
                    correct_gen = 0
                    correct_age = 0
                    
                    for _ in range(test_size):
                        # current_step = tf.train.global_step(sess, global_step)
                        x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum = fgsm_batch_iter.next_balanced_label_batch()
                        a_r_datum, a_l_datum, a_g_datum, a_a_datum, y = dev_attacker_step( x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum, 5, print_result=False)
                        
                        correct_rat += a_r_datum
                        correct_loc += a_l_datum
                        correct_gen += a_g_datum
                        correct_age += a_a_datum

                    print(
                        "~~~~",
                        correct_rat/test_size,
                        correct_loc/test_size,
                        correct_gen/test_size,
                        correct_age/test_size,
                    )




                # END>
                # ==============================================================================================================
                
        elif FLAGS.mode == "attacking_privacy":
            for _ in range(FLAGS.num_epochs * data_size // FLAGS.batch_size):
                current_step = tf.train.global_step(sess, global_step)
                lr_lamb = FLAGS.lr_lambda

                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()
                train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    # DONY: using dev set (data_id=1)
                    loss_rat, acc_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y_batch = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                    
                    # dony: this line need to be separated from the training phase
                    # dony: using test set (data_id=2)
                    # cnn.set_phase(phase='eval')
                    test_loss, test_acc, test_f1, test_l_loc, test_a_loc, test_l_gen, test_a_gen, test_l_age, test_a_age, y_test_batch = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)

            print("training attack")
            for _ in range(FLAGS.num_epochs * data_size // FLAGS.batch_size):
                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()
                train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_attack_l, adv_lam=lr_lamb, lr=training_learning_rate * 0.1)
                # ==============================================================================================================
                # <BEGIN ORIGINAL
                # evaluaate as usual, for the entire test set
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    # test_score, a_l, a_g, a_a, y = dev_attacker_step( x_test, loc_test, gen_test, age_test, rat_test, 2)
                # END>
                # ==============================================================================================================
                # <BEGIN MODIFIED
                # current_step = tf.train.global_step(sess, global_step)
                    correct_rat = 0
                    correct_loc = 0
                    correct_gen = 0
                    correct_age = 0
            
            print("ATTACKKKKKKKKKKKKKKKK PRIVACYYY")            

            x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum = fgsm_batch_iter.next_balanced_label_batch()
            a_r_datum, a_l_datum, a_g_datum, a_a_datum, y, original_h_drop = dev_attacker_step( x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum, 4, print_result=True, current_mode="without_fgm", attack_attr=FLAGS.attacking_privacy_preserving_attr, return_h_drop=True)

            # copy the weights and biases to other model
            print('weights and biases ************')
            weight_var_h1 = sess.run(tf.trainable_variables("l_attacker/full_connect_h1/W"))
            bias_var_h1 = sess.run(tf.trainable_variables("l_attacker/full_connect_h1/b"))

            weight_var_score = sess.run(tf.trainable_variables("l_attacker/full_connect_score/W"))
            bias_var_score = sess.run(tf.trainable_variables("l_attacker/full_connect_score/b"))

            # print(weight_var_h1.shape)
            print(type(bias_var_h1))
            # print(weight_var_score.shape)
            # print(bias_var_score.shape)

            print('weights and biases copied ************')
            # lanjutan_cnn_copied = CopiedClassifier(
            #     filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            #     num_filters=FLAGS.num_filters,
            #     num_locations=locations.shape[1],
            #     hidden_size=300,
            # )

            feed_dict = {
                lanjutan_cnn_copied.input_location : batch_loc,
                lanjutan_cnn_copied.h_drop : original_h_drop,
                lanjutan_cnn_copied.W_h1 : weight_var_h1[0],
                lanjutan_cnn_copied.b_h1 : bias_var_h1[0],
                lanjutan_cnn_copied.W_score : weight_var_score[0],
                lanjutan_cnn_copied.b_score : bias_var_score[0],
                
            }
            print('ergonomics')

            # sess.run(lanjutan_cnn_copied.W_h1.assign(weight_var_h1))

            location_attacker_score = sess.run(
                [lanjutan_cnn_copied.location_attacker_score],
                feed_dict
            )

            print('~~~~~~~~~~~~~~~~~~~~~~~')
            print(location_attacker_score)
            
            # sess.run()
            # sess.run(lanjutan_cnn_copied.W_h1.assign(weight_var_h1))
            print('bambang')

            # sess.run(lanjutan_cnn_copied.b_h1.assign(bias_var_h1))
            # sess.run(lanjutan_cnn_copied.W_score.assign(weight_var_score))
            print('bambang2')
            # sess.run(lanjutan_cnn_copied.b_score.assign(bias_var_score))
            lanjutan_in_art = TensorFlowV2Classifier(
                model = lanjutan_cnn_copied,
                nb_classes = 2,
                input_shape = (384, 300),
            )

            predictions = lanjutan_in_art.predict(original_h_drop)
            # var_scope2 = tf.trainable_variables("copied_attacker/full_connect_h1/W")
            # weight_var2 = sess.run(var_scope2)
            # print('shibuya')
            # print('original ^^^^^^^^^^^^^')
            # print(weight_var_h1)
            # print('copied ^^^^^^^^^^^^^')
            # print(weight_var2)
            # print(weight_var_h1 == weight_var2)



            # print('ori cuy @@@@@@@@@')
            # print(type(original_h_drop))
            # print(original_h_drop.shape)
            # print(original_h_drop)

            for i in range(test_size):
                a_r_datum, a_l_datum, a_g_datum, a_a_datum, y, original_h_drop = dev_attacker_step( x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum, 4, print_result=True, current_mode="without_fgm", attack_attr=FLAGS.attacking_privacy_preserving_attr, return_h_drop=True)

                if y['pred']['loc'] != y['true']['loc']:
                    fgsm_step(x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum)
                else:
                    print("8\tx\tx\t{}\t{}\tx\t{}\tx\t{}\tx\t{}".format(
                        # step,
                        int(y['pred']['rat'] == y['true']['rat']),
                        int(y['pred']['rat'] == y['true']['rat']),
                        int(y['pred']['loc'] == y['true']['loc']),
                        int(y['pred']['gen'] == y['true']['gen']),
                        int(y['pred']['age'] == y['true']['age']),
                    ))

                # proceed to the next datum/sample
                if i != test_size-2:
                    x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum = fgsm_batch_iter.next_balanced_label_batch()
            
                    # ==========kelar direview sampe sini
                        # # a_r_datum, a_l_datum, a_g_datum, a_a_datum, y, original_h_drop = dev_attacker_step( x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum, 2, print_result=False, current_mode="with_fgm", attack_attr=FLAGS.attacking_privacy_preserving_attr, injected_h=original_h_drop)
                        # # check if pred == truth label. If not, then perform the attack
                        # print('#####')
                        # print(y['pred']['loc'])
                        # print(y['true']['loc'])
                        # print(y['true']['loc'] == y['pred']['loc'])
                        # print(type(original_h_drop))
                        # # if y['pred']['loc'] != y['true']['loc']:
                        # #     # perform attack
                        # #     new_h_drop = original_h_drop - eps * sign()

                        
                        # correct_rat += a_r_datum
                        # correct_loc += a_l_datum
                        # correct_gen += a_g_datum
                        # correct_age += a_a_datum

                        # x_test_datum, loc_test_datum, gen_test_datum, age_test_datum, rat_test_datum = fgsm_batch_iter.next_balanced_label_batch()

                    # print(
                    #     "~~~~",
                    #     correct_rat/test_size,
                    #     correct_loc/test_size,
                    #     correct_gen/test_size,
                    #     correct_age/test_size,
                    # )

            # if the spotlight is on location, then watch the loss of location (from FC layers) and use that gradient to change the label.

            # fgm attack
            # attack_attr = FLAGS.attacking_privacy_preserving_attr
            # test_score, a_l, a_g, a_a, y s= dev_attacker_step( x_test, loc_test, gen_test, age_test, rat_test, 5, "with_fgm", attack_attr)