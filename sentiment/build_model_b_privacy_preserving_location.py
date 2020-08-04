#! /usr/bin/env python

import os
import time
import datetime
import cPickle

import tensorflow as tf
import numpy as np

import data_helpers
from text_cnn import TextCNN

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
tf.flags.DEFINE_float("lr_lambda", 1e-3, "lr lambda")

tf.flags.DEFINE_string("adv_attribute", "location", "age, gender, location, or all")

#  parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200 --> 100 by lrank)")
# tf.flags.DEFINE_integer("num_epochs", 3, "Number of training epochs (default: 200 --> 100 by lrank)")
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
for attr, value in sorted(FLAGS.__flags.iteritems()):
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
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            num_ratings=ratings.shape[1],
            num_locations=locations.shape[1],
            num_genders=genders.shape[1],
            num_ages=ages.shape[1],
            hidden_size=300,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            )

        # Define Training procedure
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        adv_lambda = tf.placeholder(tf.float32, shape=[], name="adversarial_lambda")

        global_step = tf.Variable(0, name="global_step", trainable=False)
        all_var_list = tf.trainable_variables()

        optimizer_n = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.rating_loss,
#                cnn.location_loss,
                global_step=global_step
                )

        var_loca = [var for var in all_var_list if 'location' in var.name]
        # print var_loca
        var_gend = [var for var in all_var_list if 'gender' in var.name]
        # print var_gend
        var_age = [var for var in all_var_list if 'age' in var.name]
        # print var_gend
        assert( len(var_loca) == 4 and len(var_gend) == 4 and len(var_age) == 4 )

        if FLAGS.adv_attribute == "age":
            var_d = var_age
            disc_loss = cnn.age_loss
        elif FLAGS.adv_attribute == "gender":
            var_d = var_gend
            disc_loss = cnn.gender_loss
        elif FLAGS.adv_attribute == "location":
            var_d = var_loca
            disc_loss = cnn.location_loss
        elif FLAGS.adv_attribute == "all":
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

        optimizer_location = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
               cnn.location_loss,
                global_step=global_step,
                # var_list=var_loca,
                )

        optimizer_gender = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
               cnn.gender_loss,
                global_step=global_step,
                # var_list=var_loca,
                )

        optimizer_age = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
               cnn.age_loss,
                global_step=global_step,
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
                var_list=var_attack_g
                )

        var_attack_a = [var for var in all_var_list if 'a_attacker' in var.name]
        optimizer_attack_a = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.age_attacker_loss,
                var_list=var_attack_a
                )
        assert( len(var_attack_l) == 4 and len(var_attack_g) == 4 and len(var_attack_a) == 4 )

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

        def dev_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=1):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              adv_lambda: 0.,
              cnn.dropout_keep_prob: 1.
            }
            step, l_rat, a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
            # step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)
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
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # _, step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
            #     [optimizer, global_step,
            #     cnn.rating_loss, cnn.rating_accuracy,
            #     cnn.location_loss, cnn.location_pred,
            #     cnn.gender_loss, cnn.gender_pred,
            #     cnn.age_loss, cnn.age_pred],
            #     feed_dict)

            _, step, l_rat, a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
            # step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [optimizer, global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)
            
            # time_str = datetime.datetime.now().isoformat()
            f1_rat = sklearn.metrics.f1_score(y_true=np.argmax(batch_rat, axis=1), y_pred=p_rat, average='micro')
            print("0\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
                step, l_rat - adv_lam * (l_loc + l_gen + l_age),
                l_rat, a_rat, f1_rat,
                l_loc, incomp_acc(p_loc, batch_loc),
                l_gen, incomp_acc(p_gen, batch_gen),
                l_age, incomp_acc(p_age, batch_age)
                )
            )

        def train_attacker_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer, adv_lam=0, lr = 1e-4):
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
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [optimizer, global_step,
                cnn.rating_loss, cnn.rating_accuracy,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)

            # _, step, l_rat, a_rat, l_loc, p_loc = sess.run(
            #     [optimizer_attack_l, global_step,
            #     cnn.rating_loss, cnn.rating_accuracy,
            #     cnn.location_attacker_loss, cnn.location_attacker_pred],
            #     feed_dict)

            # _, step, l_gen, p_gen = sess.run(
            #     [optimizer_attack_g, global_step,
            #     cnn.gender_attacker_loss, cnn.gender_attacker_pred],
            #     feed_dict)

            # _, step, l_age, p_age = sess.run(
            #     [optimizer_attack_a, global_step,
            #     cnn.age_attacker_loss, cnn.age_attacker_pred],
            #     feed_dict)
            
            # time_str = datetime.datetime.now().isoformat()
            # print("0\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
            #     step,
            #     l_rat, a_rat,
            #     l_loc, incomp_acc(p_loc, batch_loc),
            #     l_gen, incomp_acc(p_gen, batch_gen),
            #     l_age, incomp_acc(p_age, batch_age)
            #     )
            # )

        def dev_attacker_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=1):
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
              cnn.dropout_keep_prob: 1.
            }
            
            step, l_rat, a_rat, p_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)

            a_loc = incomp_acc(p_loc, batch_loc)
            a_gen = incomp_acc(p_gen, batch_gen)
            a_age = incomp_acc(p_age, batch_age)
            f1_rat = sklearn.metrics.f1_score(
                y_true = np.argmax(batch_rat, axis=1),
                y_pred = p_rat,
                average= 'macro'
            )
            f1_rat = sklearn.metrics.f1_score(y_true=np.argmax(batch_rat, axis=1), y_pred=p_rat, average='micro')
            print("{}\t{}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}\t{:0.5g}".format(
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
            

        writer = csv.writer(open('output_y.csv', 'w'), delimiter='\t')
        writer.writerow(['true rat', 'pred rat', 'true loc', 'pred loc', 'true gen', 'pred gen', 'true age', 'pred age',])

        #data_split
        x_train, loc_train, gen_train, age_train, rat_train, \
        x_dev, loc_dev, gen_dev, age_dev, rat_dev, \
        x_test, loc_test, gen_test, age_test, rat_test = data_helpers.data_split_train_dev_test(
            [text_x, locations, genders, ages, ratings],
            shuffle=True
            )

        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        print('train_writer at ', FLAGS.log_dir)
        sess.run(tf.global_variables_initializer())
        sess.run(cnn.emb_W.assign(w_embs))

        # Training loop. For each batch...
        data_size = len(x_train)
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

        # dony: non attack training
        for _ in range(FLAGS.num_epochs * data_size / FLAGS.batch_size):

            current_step = tf.train.global_step(sess, global_step)
            lr_lamb = (current_step / 100) / 1000.0
            
            if lr_lamb > FLAGS.lr_lambda:
                lr_lamb = FLAGS.lr_lambda

            batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()

            # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)

            # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=0.0001, lr=training_learning_rate * 0.1)
            # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)
            train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)
            

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                # print("\nEvaluation:")
                print("3\t{:0.5g}".format(lr_lamb))
                # DONY: using dev set (data_id=1)
                loss_rat, acc_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y_batch = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                # dony: using test set (data_id=2)
                test_loss, test_acc, test_f1, test_l_loc, test_a_loc, test_l_gen, test_a_gen, test_l_gen, test_a_age, y_test_batch = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)
                # y_test['true']['rat'] = np.concatenate((y_test['true']['rat'], y_test_batch['true']['rat']), axis=0)
                # y_test['true']['loc'] = np.concatenate((y_test['true']['loc'], y_test_batch['true']['loc']), axis=0)
                # # print(y_test['true']['loc'])
                # y_test['true']['gen'] = np.concatenate((y_test['true']['gen'], y_test_batch['true']['gen']))
                # y_test['true']['age'] = np.concatenate((y_test['true']['age'], y_test_batch['true']['age']))

                # y_test['pred']['rat'] = np.concatenate((y_test['pred']['rat'], y_test_batch['pred']['rat']))
                # y_test['pred']['loc'] = np.concatenate((y_test['pred']['loc'], y_test_batch['pred']['loc']))
                # y_test['pred']['gen'] = np.concatenate((y_test['pred']['gen'], y_test_batch['pred']['gen']))
                # y_test['pred']['age'] = np.concatenate((y_test['pred']['age'], y_test_batch['pred']['age']))
                
                # test_score = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)
                # dony: determine the best model using dev set
        #         if acc > best_dev_score:
        #             best_dev_score = acc
        #             best_test_score = test_score
        # print(best_dev_score, best_test_score)

        # print("training attack")
        # for _ in range(FLAGS.num_epochs * data_size / FLAGS.batch_size):
        #     batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()
        #     # train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_attack_l, adv_lam=lr_lamb, lr=training_learning_rate * 0.1 )
        #     train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate * 0.1 )

        #     current_step = tf.train.global_step(sess, global_step)
        #     if current_step % FLAGS.evaluate_every == 0:
        #         test_score, a_l, a_g, a_a = dev_attacker_step( x_test, loc_test, gen_test, age_test, rat_test, 2)

        # calculating performance of the entire test set
        # text_file = open("Output.txt", "w")
        # text_file.write(str(y_test['true']) + '\n')
        # # writer.writerow(y_test['true'])
        # for i in range(len(y_test['true']['rat'])):
        #     csv_row = [
        #         y_test['true']['rat'][i],
        #         y_test['pred']['rat'][i],

        #         y_test['true']['loc'][i],
        #         y_test['pred']['loc'][i],

        #         y_test['true']['gen'][i],
        #         y_test['pred']['gen'][i],

        #         y_test['true']['age'][i],
        #         y_test['pred']['age'][i],
        #     ]
        #     writer.writerow(csv_row)

        # f1_rat = sklearn.metrics.f1_score(y_test['true']['rat'], y_test['pred']['rat'], average='macro')
        # acc_rat = sklearn.metrics.accuracy_score(y_test['true']['rat'], y_test['pred']['rat'])

        # acc_loc = sklearn.metrics.accuracy_score(y_test['true']['loc'], y_test['pred']['loc'])
        # acc_gen = sklearn.metrics.accuracy_score(y_test['true']['gen'], y_test['pred']['gen'])
        # acc_age = sklearn.metrics.accuracy_score(y_test['true']['age'], y_test['pred']['age'])

        # res = [6, acc_rat, f1_rat, acc_loc, acc_gen, acc_age]
        # print("\t".join(str(x) for x in res))
        # print('hmm')
        # print(len(y_test['true']['rat']))
# --------------------------------------------------------------
# Notes:
# - optimizer_n : normal utility (minimise rating)
# - optimizer_d : sepertinya tidak dipakai privacy adv (with the lambda) (trying to get bad at l/g/a)
# - optimizer_attack_l/g/a : normal privacy (trying to learn l/g/a)
# - optimizer_g : the formula