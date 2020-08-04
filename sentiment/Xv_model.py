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
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate alpha")
tf.flags.DEFINE_float("lr_lambda", 1e-3, "LR lambda")

#  parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200 --> 100 by lrank)")
# tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200 --> 100 by lrank)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



# Data Preparatopn
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
    print("{}={}".format(attr.upper(), value))
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
        # print("all_var_list", vars(all_var_list))

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
        # assert( len(var_loca) == 2 and len(var_gend) == 2 and len(var_age) == 2 )
        var_d = var_loca + var_gend + var_age

        disc_loss = cnn.location_loss + cnn.gender_loss + cnn.age_loss
        # disc_loss = cnn.age_loss
        optimizer_d = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                adv_lambda * disc_loss,
                var_list=var_d,
                global_step=global_step,
                )

        optimizer_loc = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.location_loss,
                # var_list=var_loca,
                var_list=var_d,
                global_step=global_step,
                )

        optimizer_gend = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.gender_loss,
                var_list=var_gend,
                global_step=global_step,
                )

        optimizer_age = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.age_loss,
                var_list=var_age,
                global_step=global_step,
                )


        var_g = [var for var in all_var_list if var not in var_d]
        optimizer_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.rating_loss - adv_lambda * (cnn.gender_loss + cnn.age_loss + cnn.location_loss),
                # cnn.rating_loss - adv_lambda * (cnn.gender_loss),
                # cnn.rating_loss - adv_lambda * (cnn.age_loss),
                # cnn.rating_loss - adv_lambda * (cnn.location_loss),
                # cnn.rating_loss,
                var_list=var_g,
                global_step=global_step
                )

        def incomp_acc(logit, label):
            c_cor = 0.
            c_total = 0.
            gt = np.argmax(label, axis=1)
            # print("logit: ")
            # print(logit)
            # print("label: ")
            # print(label)
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
                [global_step,
                cnn.rating_loss, cnn.rating_accuracy, cnn.rating_pred,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            ''' data_id: train:0 dev:1 test:2 '''
            
            a_loc = incomp_acc(p_loc, batch_loc)
            a_gen = incomp_acc(p_gen, batch_gen)
            a_age = incomp_acc(p_age, batch_age)
            f1_rat = sklearn.metrics.f1_score(y_true=np.argmax(batch_rat, axis=1), y_pred=p_rat, average='macro')
            print("{}\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                data_id, step,
                l_rat, a_rat, f1_rat,
                l_loc, a_loc,
                l_gen, a_gen,
                l_age, a_age )
            )
            # return a_rat, f1_rat
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
            Evaluates model on a dev set
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
            
            # time_str = datetime.datetime.now().isoformat()
            # print("0\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
            #     step,
            #     l_rat, a_rat,
            #     l_loc, incomp_acc(p_loc, batch_loc),
            #     l_gen, incomp_acc(p_gen, batch_gen),
            #     l_age, incomp_acc(p_age, batch_age)
            #     )
            # )

        writer = csv.writer(open('output_y.csv', 'w'), delimiter='\t')
        writer.writerow(['k-th fold', 'true rat', 'pred rat', 'true loc', 'pred loc', 'true gen', 'pred gen', 'true age', 'pred age',])

        #data_split
        Xv_iter = data_helpers.X_validation_iter(
            data=[text_x, locations, genders, ages, ratings],
            use_dev=True,
            fold=10,
        )
        
        for iter_fold in range( Xv_iter.fold):
            x_train, loc_train, gen_train, age_train, rat_train, \
                x_dev, loc_dev, gen_dev, age_dev, rat_dev, \
                x_test, loc_test, gen_test, age_test, rat_test = Xv_iter.next_fold()

#             print x_train.shape, loc_train.shape, gen_train.shape, age_train.shape, rat_train.shape
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

            for _ in range(FLAGS.num_epochs * data_size / FLAGS.batch_size):

                current_step = tf.train.global_step(sess, global_step)
                lr_lamb = (current_step / 100) / 10000.0
                if lr_lamb > FLAGS.lr_lambda:
                    lr_lamb = FLAGS.lr_lambda

                batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()

                # baseline (optimizer_n)
                # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_n, adv_lam=0, lr=training_learning_rate )

                # attacking method (optimizer_d)
                # adv_lam need to be updated into a thing/double
                train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_n, adv_lam=1e-3, lr=training_learning_rate ) 
                # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=1, lr=training_learning_rate ) 
                # train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_loc, adv_lam=1, lr=training_learning_rate ) 

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    # print("\nEvaluation:")
                    print("3\t{:g}".format(lr_lamb))
                    # acc, f1 = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                    # l_rat, a_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y
                    loss_rat, acc_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y_batch = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                    # test_acc, test_f1 = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)



                    # l_rat, a_rat, f1_rat, l_loc, a_loc, l_gen, a_gen, l_age, a_age, y
                    test_loss, test_acc, test_f1, test_l_loc, test_a_loc, test_l_gen, test_a_gen, test_l_gen, test_a_age, y_test_batch = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)
                    y_test['true']['rat'] = np.concatenate((y_test['true']['rat'], y_test_batch['true']['rat']), axis=0)
                    y_test['true']['loc'] = np.concatenate((y_test['true']['loc'], y_test_batch['true']['loc']), axis=0)
                    # print(y_test['true']['loc'])
                    y_test['true']['gen'] = np.concatenate((y_test['true']['gen'], y_test_batch['true']['gen']))
                    y_test['true']['age'] = np.concatenate((y_test['true']['age'], y_test_batch['true']['age']))

                    y_test['pred']['rat'] = np.concatenate((y_test['pred']['rat'], y_test_batch['pred']['rat']))
                    y_test['pred']['loc'] = np.concatenate((y_test['pred']['loc'], y_test_batch['pred']['loc']))
                    y_test['pred']['gen'] = np.concatenate((y_test['pred']['gen'], y_test_batch['pred']['gen']))
                    y_test['pred']['age'] = np.concatenate((y_test['pred']['age'], y_test_batch['pred']['age']))
                

                    


                    # if acc > best_dev_score:
                    #     best_dev_score = acc
                    #     best_test_score = test_acc
                    # if f1 > best_dev_f1_score:
                    # # if acc > best_dev_f1_score:
                    #     best_dev_f1_score = f1
                    #     # best_dev_f1_score = acc
                    #     best_test_f1_score = test_f1
                    #     best_dev_result = [4, loss, acc, f1, l_loc, a_loc, l_gen, a_gen,l_age, a_age]
                    #     best_test_result = [5, test_loss, test_acc, test_f1, test_l_loc, test_a_loc, test_l_gen, test_a_gen, test_l_gen, test_a_age]

            # calculating performance of the entire test set
            text_file = open("Output.txt", "w")
            text_file.write(str(y_test['true']) + '\n')
            # writer.writerow(y_test['true'])
            for i in range(len(y_test['true']['rat'])):
                csv_row = [
                    iter_fold,

                    y_test['true']['rat'][i],
                    y_test['pred']['rat'][i],

                    y_test['true']['loc'][i],
                    y_test['pred']['loc'][i],

                    y_test['true']['gen'][i],
                    y_test['pred']['gen'][i],

                    y_test['true']['age'][i],
                    y_test['pred']['age'][i],
                ]
                writer.writerow(csv_row)

            f1_rat = sklearn.metrics.f1_score(y_test['true']['rat'], y_test['pred']['rat'], average='macro')
            acc_rat = sklearn.metrics.accuracy_score(y_test['true']['rat'], y_test['pred']['rat'])

            acc_loc = sklearn.metrics.accuracy_score(y_test['true']['loc'], y_test['pred']['loc'])
            acc_gen = sklearn.metrics.accuracy_score(y_test['true']['gen'], y_test['pred']['gen'])
            acc_age = sklearn.metrics.accuracy_score(y_test['true']['age'], y_test['pred']['age'])

            res = [6, f1_rat, acc_rat, acc_loc, acc_gen, acc_age]
            print("\t".join(str(x) for x in res))
            print('hmm')
            print(len(y_test['true']['rat']))
            # print('f1_rat  : ', f1_rat)
            # print('acc_rat : ', acc_rat)
            # print('acc_loc : ', acc_loc)
            # print('acc_gen : ', acc_gen)
            # print('acc_age : ', acc_age)
            # if f1_rat > best_test['f1_rat']:
            #     best_test['f1_rat'] = f1_rat
            #     best_test['acc_rat'] = acc_rat

            #     best_test['acc_loc'] = acc_loc
            #     best_test['acc_gen'] = acc_gen
            #     best_test['acc_age'] = acc_age


            

            # print("\t".join(str(x) for x in best_dev_result))
            # print("\t".join(str(x) for x in best_test_result))
            #print best_dev_score, best_test_score
            # Xv_iter.dev_scores.append(best_dev_score)
            # Xv_iter.test_scores.append(best_test_f1_score)

            # for i in range(len(test_y_pred_rat)):
            #     print(str(test_y_true_rat[i]) + '\t' + str(test_y_pred_rat[i]))

# if Xv_iter.use_dev:
#     print np.average(Xv_iter.dev_scores), Xv_iter.dev_scores
# print np.average(Xv_iter.test_scores), Xv_iter.test_scores
