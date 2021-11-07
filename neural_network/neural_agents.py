from tensorflow.keras import Sequential, layers, activations, Model
import numpy as np
from neural_network.helper_functions.calculation_helpers import *


#######################################################################################################################

class BaseAgent(Model):

    def __init__(self,
                 name,
                 vocab_size=20,
                 message_length=1,
                 alpha=5.,
                 encoding_dim=32,
                 n_distractors=1,
                 epsilon=1e-16,
                 messages=None,
                 message_encoder=None,
                 state_encoder=None):
        """
        Constructor of the base agent class.

        :param name: name of the agent
        :param vocab_size: size of the vocabulary
        :param message_length: message length
        :param alpha: RSA optimality parameter
        :param encoding_dim: encoding dimension, so dimension of the embedding vectors
        :param n_distractors: number of distractors
        :param epsilon: value to be added for numerical stability at critical places
        :param messages: input messages
        :param message_encoder: message encoding network
        :param state_encoder: state encoding network
        """

        super(BaseAgent, self).__init__(name=name)

        self.vocab_size = vocab_size
        self.message_length = message_length
        self.alpha = alpha
        self.encoding_dim = encoding_dim
        self.n_distractors = n_distractors
        self.epsilon = epsilon
        self.messages = messages
        if message_encoder is None:
            message_encoder = Sequential(
                [layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=1),
                 layers.Flatten(),
                 layers.Dense(32, activation=None)],
                name='message_encoder')
        self.message_encoder = message_encoder

        if state_encoder is None:
            self.state_encoder = Sequential(
                [layers.Dense(32, activation=None)],
                name=name + '_state_encoder')
        self.state_encoder = state_encoder

    def build(self, vision_dim=64, language_dim=1):
        """ Build function. Keras models must be build to initialize the weights before training.

        :param vision_dim: dimension of the visual input data
        :param language_dim: message length
        """

        self.state_encoder(np.zeros((1, vision_dim)))
        self.message_encoder(np.zeros((1, language_dim)))

    def __call__(self, inputs, **kwargs):
        """ call function. Must be implemented by Keras Model
        """
        pass

    def listener_forward_pass(self, inputs, expand_messages=False):
        """ Listener's forward pass generating the policy.
        :param inputs: list of
                                1 - message input (batch size x message length)
                                2 - target objects (batch size x image feature dim (64))
                                3 - list of distractor objects (batch size x image feature dim (64))
                                4 - target message proposals: messages the agent reasons about for the target object
                                5 - list of distractor messages proposals
        A note on the message proposals: in our implementation they always correspond to either
        1) the messages the agent is familiar with, so the messages in the training set
        2) or all messages, if the negative sampling of words is used
        In addition, the message proposals for target and distractor are identical in our case.
        :param  expand_messages: whether the messages to reason over should be expanded, this is only the case if
                the agent encounters a new message (e.g. in the ME bias evaluation) and does not already reason about
                this message (so does not use negative sampling of words)
        """

        pass

    def listener_action(self, inputs, expand_messages=False):
        """ Listener's action given the inputs.

        :param inputs: see forward pass
        :param expand_messages: see forward pass
        :return selection sampled from policy, log_policy
        """

        policy = self.listener_forward_pass(inputs, expand_messages=expand_messages)
        log_policy = tf.math.log(policy + self.epsilon)
        state = tf.squeeze(tf.one_hot(tf.random.categorical(log_policy, 1),
                                      depth=self.n_distractors + 1))
        return state, log_policy

    def speaker_forward_pass(self, inputs):
        """ Speaker's forward pass (speaker as in the RSA model) generating the policy.

        :param inputs: target object (batch size x visual feature size (64))
        :return: policy over messages
        """

        target_enc = self.state_encoder(inputs[0])
        distractors = tf.concat(inputs[1], axis=0)
        distractor_encs = self.state_encoder(distractors)
        message_proposals = inputs[2]
        batch_size = target_enc.shape[0]
        n_proposals = message_proposals.shape[1]

        message_encodings = self.message_encoder(
            np.reshape(message_proposals, (batch_size * n_proposals, self.message_length)))

        target_enc_tiled = tf.reshape(tf.tile(target_enc, tf.constant([1, n_proposals], dtype=tf.int32)),
                                      (batch_size * n_proposals, self.encoding_dim))
        distractor_encs_tiled = tf.reshape(tf.tile(distractor_encs, tf.constant([1, n_proposals], dtype=tf.int32)),
                                           (batch_size * self.n_distractors * n_proposals, self.encoding_dim))

        target_similarities = dot_product(target_enc_tiled, message_encodings)
        target_similarities = tf.reshape(target_similarities, (batch_size, n_proposals))
        distractor_similarities = dot_product(distractor_encs_tiled,
                                              tf.tile(message_encodings, [self.n_distractors, 1]))
        distractor_similarities = tf.reshape(distractor_similarities, (batch_size * self.n_distractors, n_proposals))
        distractor_similarities = tf.split(distractor_similarities, self.n_distractors)

        policy_literal_listener = activations.softmax(tf.stack([target_similarities] +
                                                               [distractor_sim
                                                                for distractor_sim in distractor_similarities],
                                                               axis=2),
                                                      axis=2)

        policy_literal_listener_exponentiated = tf.math.pow(policy_literal_listener, self.alpha)
        policy = (policy_literal_listener_exponentiated[:, :, 0] /
                  (tf.reduce_sum(policy_literal_listener_exponentiated[:, :, 0] + self.epsilon, axis=1, keepdims=True))
                  )

        return policy

    def speaker_action(self, inputs):
        """ Sample speaker action given visual input.

        :param inputs: target object (batch size x visual feature size (64))
        :return: sampled messages, sampled messages as one hot vectors, log policy
        """

        batch_size = inputs[0].shape[0]
        message_proposals = inputs[2]
        n_proposals = message_proposals.shape[1]

        policy = self.speaker_forward_pass(inputs)

        log_policy = tf.math.log(policy + self.epsilon)
        sampled_action = tf.squeeze(tf.random.categorical(log_policy, 1))
        messages = message_proposals[np.arange(batch_size), sampled_action, :]

        return messages, tf.one_hot(sampled_action, depth=n_proposals), log_policy


#######################################################################################################################

class LiteralListener(BaseAgent):
    """
    Literal listener adapted from the RSA framework.
    """

    def __init__(self, name='literal_listener', **kwargs):
        """ Constructor.
        """

        super(LiteralListener, self).__init__(name=name, **kwargs)

    def listener_forward_pass(self, inputs, expand_messages=False):
        """ Literal listener's forward pass generating the policy.

        :param inputs: list of
                                1 - message input (batch size x message length)
                                2 - target objects (batch size x image feature dim (64))
                                3 - list of distractor objects (batch size x image feature dim (64))
        :param expand_messages: whether to add a new messages to the reasoning process
        :return policy over the target and distractor(s)
        """

        message_input = inputs[0]
        target_input = inputs[1]
        distractors = tf.concat(inputs[2], axis=0)
        distractor_encs = self.state_encoder(distractors)

        message_encoding = self.message_encoder(message_input)
        target_encoding = self.state_encoder(target_input)

        target_similarity = dot_product(target_encoding, message_encoding)
        distractor_similarity = dot_product(distractor_encs, tf.tile(message_encoding, [self.n_distractors, 1]))
        distractor_similarity = tf.split(distractor_similarity, self.n_distractors)
        policy = activations.softmax(tf.stack([target_similarity] +
                                              [distractor_sim for distractor_sim in distractor_similarity], axis=1))
        return policy


#######################################################################################################################

class PragmaticListener(BaseAgent):
    """ Pragmatic listener adapted from the RSA framework. """

    def __init__(self, name='pragmatic_listener', **kwargs):
        """ Constructor.
        """

        super(PragmaticListener, self).__init__(name=name, **kwargs)

    def listener_forward_pass(self, inputs, expand_messages=False):
        """ Pragmatic listener's forward pass generating the policy.
                :param inputs: list of
                                        1 - message input (batch size x message length)
                                        2 - target objects (batch size x image feature dim (64))
                                        3 - list of distractor objects (batch size x image feature dim (64))
                                        4 - target message proposals: messages the agent reasons about for the target object
                                        5 - list of distractor messages proposals
                A note on the message proposals: in our implementation they always correspond to either
                1) the messages the agent is familiar with, so the messages in the training set
                2) or all messages, if the negative sampling of words is used
                In addition, the message proposals for target and distractor are identical in our case.
                :param  expand_messages: whether the messages to reason over should be expanded, this is only the case if
                        the agent encounters a new message (e.g. in the ME bias evaluation) and does not already reason about
                        this message (so does not use negative sampling of words)
                :return policy over target and distractor(s)
        """

        message_input = inputs[0]
        target_input = inputs[1]
        distractor_list = inputs[2]
        message_proposals_target = inputs[3]
        message_proposals_distractor = tf.concat(inputs[4], axis=0)

        message_input_expanded = tf.expand_dims(message_input, axis=-2)
        tiled_message_input_expanded = tf.tile(message_input_expanded, (self.n_distractors, 1, 1))

        # if the incoming message is a familiar message, it will also be proposed by the message proposal system:
        # so we only need to identify its index and can use this to calculate the listener's policy
        if not expand_messages:
            filter_targets = tf.cast(
                tf.reduce_sum(message_input_expanded - message_proposals_target, axis=2) == 0,
                tf.float32)

            filter_distractors = tf.cast(
                tf.reduce_sum(tiled_message_input_expanded - message_proposals_distractor, axis=2) == 0,
                tf.float32)
            filter_distractors = tf.split(filter_distractors, self.n_distractors, axis=0)
            messages_target = message_proposals_target
            messages_distractor = tf.split(message_proposals_distractor, self.n_distractors)

        else:
            messages_target = tf.concat([message_input_expanded, message_proposals_target], axis=1)
            messages_distractor = tf.concat([tiled_message_input_expanded, message_proposals_distractor], axis=1)
            messages_distractor = tf.split(messages_distractor, self.n_distractors)

        # calculate speaker's policy for target and distractors
        speaker_policy_target = self.speaker_forward_pass([target_input, distractor_list, messages_target])
        speaker_policy_dist = []
        for i in range(self.n_distractors):
            remaining_distractors = ([target_input] +
                                     [distractor for d_index, distractor in enumerate(distractor_list) if d_index != i])
            speaker_policy_dist.append(self.speaker_forward_pass([distractor_list[i],
                                                                  remaining_distractors,
                                                                  messages_distractor[i]]))

        if not expand_messages:
            prob_message_target = tf.reduce_sum(speaker_policy_target * filter_targets, axis=1)
            prob_message_distractor = [tf.reduce_sum(speaker_policy_dist[i] * filter_distractors[i], axis=1)
                                       for i in range(self.n_distractors)]
        else:
            prob_message_target = speaker_policy_target[:, 0]
            prob_message_distractor = [speaker_policy_dist[i][:, 0] for i in range(self.n_distractors)]

        normalization_value = tf.add(prob_message_target, tf.add_n(prob_message_distractor))
        policy = tf.stack([prob_message_target / (normalization_value + self.epsilon)] +
                          [prob_message_dist / (normalization_value + self.epsilon) for
                           prob_message_dist in prob_message_distractor], axis=1)
        return policy
