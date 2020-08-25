import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class TextCNN(object):
    
    def fgm(self,
            x, attack_attr, 
            filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size,
            eps=0.25, epochs=100, sign=True,
            # eps=0.01, epochs=5, sign=True,
        ):
        # return x
        x_adv = tf.identity(x, name="x_adv")

        if sign:
            noise_fn = tf.sign
        else:
            noise_fn = tf.identity

        eps = -tf.abs(eps)

        def _cond(x_adv, i):
            return tf.less(i, epochs)

# # begin
        # def _body(x_adv, i, attack_attr, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size):
        def _body(x_adv, i):
            rating_loss, rating_accuracy, rating_pred, rating_score, \
                location_attacker_loss, location_attacker_accuracy, location_attacker_pred, location_attacker_score, \
                gender_attacker_loss, gender_attacker_accuracy, gender_attacker_pred, gender_attacker_score, \
                age_attacker_loss, age_attacker_accuracy, age_attacker_pred, age_attacker_score = self.privacy_simple_classifier(x_adv, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size)

            logits = tf.identity(location_attacker_score)  #TODO: change the hardcoding: use tf.case instead
            # logits = tf.identity(eval(tf.py_func(lambda val: str(val), [attack_attr], tf.string) + "_score"))
            # y_bar = tf.identity(eval("self.input_" + attack_attr.split("_")[0]))
            target = tf.identity(self.input_location)    #TODO: change the hardcoding
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=target
            )
            dy_dx, = tf.gradients(loss, x_adv)
            x_adv = tf.stop_gradient(x_adv + eps * noise_fn(dy_dx))
            # x_adv = tf.stop_gradient(x_adv)
            # x_adv = tf.reshape(x_adv, [-1, 384])
            # x_adv = tf.squeeze(x_adv, axis=[0])
            return x_adv, i+1 

        # with tf.variable_scope("fgsm_attack"):
        x_adv, _ = tf.while_loop(
                        # lambda i: tf.less(i, epochs),
                        _cond,
                        _body,
                        # [i],
                        (x_adv, 0),
                        back_prop=False,
                        name='fgm',
                    )
        return x_adv

    def simple_classifier(self, h_drop, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size):
    # def after_cnn(self, h_drop, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size):
        with tf.variable_scope("location"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            location_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_locations]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=location_score,
                    labels=self.input_location
                    )
                location_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                location_pred = tf.argmax(location_score, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(location_pred, tf.argmax(self.input_location, 1) ),
                    "float"
                    )
                location_accuracy = tf.reduce_mean(cor_pred, name="acc")


        with tf.variable_scope("gender"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            gender_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_genders]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=gender_score,
                    labels=self.input_gender
                    )
                gender_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                gender_pred = tf.argmax( gender_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(gender_pred, tf.argmax(self.input_gender, 1) ),
                    "float"
                    )
                gender_accuracy = tf.reduce_mean(cor_pred, name='acc')


        with tf.variable_scope("age"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            age_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_ages]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=age_score,
                    labels=self.input_age
                    )
                age_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                age_pred = tf.argmax( age_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(age_pred, tf.argmax(self.input_age, 1) ),
                    "float"
                    )
                age_accuracy = tf.reduce_mean(cor_pred, name='acc')

        return location_loss, location_accuracy, location_pred, location_score, \
            gender_loss, gender_accuracy, gender_pred, gender_score, \
            age_loss, age_accuracy, age_pred, age_score

    def privacy_simple_classifier(self, h_drop, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size):
        with tf.variable_scope("rating", reuse=tf.AUTO_REUSE):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            rating_score = self.wx_plus_b(
                scope_name='score',
                x=h1,
                size=[hidden_size, num_ratings]
                )
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=rating_score,
                    labels=self.input_rating
                    )
                rating_loss = tf.reduce_mean(losses, name="task_loss")

            with tf.name_scope("accuracy"):
                rating_pred = tf.argmax(rating_score, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal( rating_pred, tf.argmax(self.input_rating, 1) ),
                    "float"
                    )
                rating_accuracy = tf.reduce_mean( cor_pred, name="accuracy" )

        with tf.variable_scope("l_attacker", reuse=tf.AUTO_REUSE):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            location_attacker_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_locations]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=location_attacker_score,
                    labels=self.input_location
                    )
                location_attacker_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                location_attacker_pred = tf.argmax(location_attacker_score, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(location_attacker_pred, tf.argmax(self.input_location, 1) ),
                    "float"
                    )
                location_attacker_accuracy = tf.reduce_mean(cor_pred, name="acc")


        with tf.variable_scope("g_attacker", reuse=tf.AUTO_REUSE):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            gender_attacker_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_genders]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=gender_attacker_score,
                    labels=self.input_gender
                    )
                gender_attacker_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                gender_attacker_pred = tf.argmax( gender_attacker_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(gender_attacker_pred, tf.argmax(self.input_gender, 1) ),
                    "float"
                    )
                gender_attacker_accuracy = tf.reduce_mean(cor_pred, name='acc')


        with tf.variable_scope("a_attacker", reuse=tf.AUTO_REUSE):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            age_attacker_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_ages]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=age_attacker_score,
                    labels=self.input_age
                    )
                age_attacker_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                age_attacker_pred = tf.argmax( age_attacker_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(age_attacker_pred, tf.argmax(self.input_age, 1) ),
                    "float"
                    )
                age_attacker_accuracy = tf.reduce_mean(cor_pred, name='acc')

        return rating_loss, rating_accuracy, rating_pred, rating_score, \
            location_attacker_loss, location_attacker_accuracy, location_attacker_pred, location_attacker_score, \
            gender_attacker_loss, gender_attacker_accuracy, gender_attacker_pred, gender_attacker_score, \
            age_attacker_loss, age_attacker_accuracy, age_attacker_pred, age_attacker_score,

    def cnn(self, scope_number, embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters):
        with tf.variable_scope("cnn%s" % scope_number) as scope:
        # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable(
                        name="W",
                        shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                        )
                    b = tf.get_variable(
                        name="b",
                        shape=[num_filters],
                        initializer=tf.constant_initializer(0.1)
                        )
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            pooled = tf.concat(pooled_outputs, 3)
            num_filters_total = num_filters * len(filter_sizes)
            return tf.reshape(pooled, [-1, num_filters_total])

        
    def wx_plus_b(self, scope_name, x, size):
        with tf.variable_scope("full_connect_%s" % scope_name) as scope:
            W = tf.get_variable(
                name="W",
                shape=size,
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                name="b",
                shape=[size[1]],
                initializer=tf.constant_initializer(0.1, )
                )
            y = tf.nn.xw_plus_b(x, W, b, name="hidden")
            return y

        
    def gaussian_noise_layer(self, input_layer, std = 0.001):
        noise = tf.random_normal(shape=tf.shape(input_layer) , mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def dont_run_fgsm(self, h_drop):
        # print_op = tf.Print('not an eval', ["~~~~~~~~~~~~~~~~~~~~~~~~~"])
        return h_drop

    #main enter
    def __init__(self, sequence_length, vocab_size,
            embedding_size, filter_sizes, num_filters,
            num_ratings, num_locations, num_genders, num_ages,
            hidden_size, l2_reg_lambda=0.0, injected_h=None):
            
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_rating = tf.placeholder(tf.float32, [None, num_ratings], name="input_rating_truth")
        self.input_location = tf.placeholder(tf.float32, [None, num_locations], name="input_location_truth")
        self.input_gender = tf.placeholder(tf.float32, [None, num_genders], name="input_gender_truth")
        self.input_age = tf.placeholder(tf.float32, [None, num_ages], name="input_age_truth")
        
        self.current_mode = tf.placeholder_with_default("without_fgm", name="current_mode", shape=[])
        self.attack_attr = tf.placeholder(tf.string, name="attack_attr", shape=[])

        l2_loss = tf.constant(0.0)

        with tf.variable_scope("embedding"):
            self.emb_W = tf.get_variable(
                name="lookup_emb",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=False
                )
            embedded_chars = tf.nn.embedding_lookup(self.emb_W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            self.embedded_chars_expanded = self.gaussian_noise_layer( self.embedded_chars_expanded )
        
        #hidden 0 = cnn+pooling output
        self.pub_h_pool = self.cnn("shared", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.original_h_drop = tf.nn.dropout(self.pub_h_pool, self.dropout_keep_prob)
            # self.injected_h = tf.placeholder_with_default(self.original_h_drop, shape=[None,384], name="injected_h")
            # self.injected_h = tf.placeholder_with_default(self.original_h_drop, shape=[None,384], name="injected_h")
            # self.injected_h = tf.placeholder(tf.float32, shape=[None,384], name="injected_h")
            # self.injected_h = self.fgm(self.original_h_drop, self.attack_attr, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size, eps=0.01, epochs=1, sign=True),

            self.h_drop = tf.cond(
                tf.math.equal(self.current_mode, "with_fgm"),
                # lambda: tf.identity(self.injected_h),
                # lambda: tf.identity(self.injected_h),
                # lambda: self.fgm(self.original_h_drop, self.attack_attr, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size, eps=0.01, epochs=1, sign=True),
                # lambda: tf.identity(self.injected_h),
                lambda: tf.identity(self.fgm(self.original_h_drop, self.attack_attr, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size, eps=0.01, epochs=1, sign=True)),
                lambda: tf.identity(self.original_h_drop),
            )

        
        self.location_loss, self.location_accuracy, self.location_pred, self.location_score, \
            self.gender_loss, self.gender_accuracy, self.gender_pred, self.gender_score, \
            self.age_loss, self.age_accuracy, self.age_pred, self.age_score = self.simple_classifier(self.h_drop, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size)
        
        
        self.rating_loss, self.rating_accuracy, self.rating_pred, self.rating_score, \
            self.location_attacker_loss, self.location_attacker_accuracy, self.location_attacker_pred, self.location_attacker_score, \
            self.gender_attacker_loss, self.gender_attacker_accuracy, self.gender_attacker_pred, self.gender_attacker_score, \
            self.age_attacker_loss, self.age_attacker_accuracy, self.age_attacker_pred, self.age_attacker_score = self.privacy_simple_classifier(self.h_drop, filter_sizes, num_filters, num_ratings, num_locations, num_genders, num_ages, hidden_size)
        # print(self.h_drop)

        # check if it is eval mode.
        # tf.case([(tf.math.equal(self.current_mode, "eval"), lambda: self.run_fgsm())])
        # current_phasexx = tf.cond(tf.math.equal(self.current_mode, self.current_mode), lambda: self.run_fgsm(), lambda: self.dont_run_fgsm())
        # x = tf.constant(1, name='x')
        # y = tf.constant(2, name='y')
        # self.current_mode = tf.cond(tf.math.equal(self.current_mode, "test"), lambda: self.run_fgsm(), lambda: self.dont_run_fgsm())
        # self.current_mode = tf.Print(self.current_mode, [self.current_mode], "fase~~~~~~")

        # self.current_mode = tf.Print(self.current_mode, [self.current_mode], message="bambang~~~~")
        # loss.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        # print(sess.run(self.current_mode))
        # current_phasexx = tf.cond(tf.math.less(y, x), lambda: tf.add(x, y), lambda: tf.square(y))
        
        # current_phase = tf.cond(tf.math.equal(self.current_mode, self.current_mode), self.current_mode, self.current_mode)

class CopiedClassifier(Model):
    def wx_plus_b_h1(self, scope_name, x, size):
        with tf.variable_scope("full_connect_%s" % scope_name) as scope:
            self.W_h1 = tf.placeholder(tf.float32, [384, 300], name="W")
            self.b_h1 = tf.placeholder(tf.float32, [300], name="b")
            # self.W_h1 = tf.get_variable(
            #     name="W",
            #     shape=size,
            #     initializer=tf.contrib.layers.xavier_initializer(),
            #     trainable=False,
            # )
            # self.b_h1 = tf.get_variable(
            #     name="b",
            #     shape=[size[1]],
            #     initializer=tf.constant_initializer(0.1, ),
            #     trainable=False,
            # )
            y = tf.nn.xw_plus_b(x, self.W_h1, self.b_h1, name="hidden")
            return y

    def wx_plus_b_score(self, scope_name, x, size):
        with tf.variable_scope("full_connect_%s" % scope_name) as scope:
            # self.W_score = tf.get_variable(
            #     name="W",
            #     shape=size,
            #     initializer=tf.contrib.layers.xavier_initializer(),
            #     trainable=False
            # )
            # self.b_score = tf.get_variable(
            #     name="b",
            #     shape=[size[1]],
            #     initializer=tf.constant_initializer(0.1, ),
            #     trainable=False,
            # )
            y = tf.nn.xw_plus_b(x, self.W_score, self.b_score, name="hidden")
            return y

    def __init__(self, filter_sizes, num_filters, num_locations, hidden_size):
        super(CopiedClassifier).__init__()

        self.h_drop = tf.placeholder(tf.float32, [1, num_filters * len(filter_sizes)], name="h_drop")
        self.input_location = tf.placeholder(tf.float32, [None, num_locations], name="input_location_truth")
        self.W_h1 = tf.placeholder(tf.float32, [384, 300], name="W")
        self.b_h1 = tf.placeholder(tf.float32, [300], name="b")
        self.W_score = tf.placeholder(tf.float32, [300, 2], name="W")
        self.b_score = tf.placeholder(tf.float32, [2], name="b")

        with tf.variable_scope("copied_attacker", reuse=tf.AUTO_REUSE):
            h1 = self.wx_plus_b_h1(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.location_attacker_score = self.wx_plus_b_score(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_locations]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.location_attacker_score,
                    labels=self.input_location
                    )
                self.location_attacker_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                self.location_attacker_pred = tf.argmax(self.location_attacker_score, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(self.location_attacker_pred, tf.argmax(self.input_location, 1) ),
                    "float"
                    )
                self.location_attacker_accuracy = tf.reduce_mean(cor_pred, name="acc")


    # def call(self, x):
    #     pass

    # def train_step(model, inputs, labels):
    #     pass


    # lanjutan_cnn_cp = TensorFlowV2Classifier(
    #     model = model_cp
    #     loss_object = tf.keras.losses.BinaryCrossEntropy(from_logits=True)
        # input_ph = tf.placeholder(tf.float32, shape=[1, 384]),
        # labels_ph = tf.placeholder(tf.int32, shape=[1, 2]),
        # output = model_cp.location_attacker_score,
        # train = original_h_drop,
        # loss = model.cp.location_attacker_loss,

    # )

    # predictions = classifier.predict(x_test)

