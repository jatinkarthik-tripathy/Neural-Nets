import tensorflow as tf
import numpy as np
class seq2seq:
    def create_models(self, inp_len, out_len, latent_dim=256):
        self.inp_len = inp_len
        self.out_len = out_len
        self.latent_dim = latent_dim

        #training model:
        # Define an input sequence and process it.
        encoder_inputs =  tf.keras.layers.Input(shape=(None, self.inp_len))
        encoder =  tf.keras.layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = tf.keras.layers.Input(shape=(None, self.out_len))
        decoder_lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(self.out_len, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.train_model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # Inference setup:
        self.encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

        decoder_state_input_h = tf.keras.layers.Input(shape=(self.latent_dim,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def train(self, enc_in, dec_in, dec_out, batch = 100, epochs = 20):
        self.train_model.fit([enc_in, dec_in], dec_out, batch_size = batch, epochs = epochs)	 
        self.train_model.save('seq2seq_model.h5')

    def test(self, input_seq, input_token_index, target_token_index, num_decoder_tokens, max_decoder_seq_length):
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence



