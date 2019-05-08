import tensorflow as tf
import os

class seq2seq:
    def create_models(self, inp_len, out_len):
        self.inp_len = inp_len
        self.out_len = out_len
        self.latent_dim = 256

        #training model:
        # Define an input sequence and process it.
        encoder_inputs =  tf.keras.layers.Input(shape=(None, self.inp_len))
        encoder =  tf.keras.layers.CuDNNLSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = tf.keras.layers.Input(shape=(None, self.out_len))
        decoder_lstm = tf.keras.layers.CuDNNLSTM(self.latent_dim, return_sequences=True, return_state=True)
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

    def train(self, enc_in, dec_in, dec_out, batch = 100, epochs = 1):
        self.checkpoint_path = "training/cp.ckpt"
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_weights_only=True, verbose=1)
        self.train_model.fit([enc_in, dec_in], dec_out, batch_size = batch, epochs = epochs, callbacks = [cp_callback])	 
        self.train_model.save('seq2seq_model.h5')

    def load_model():
        model.load_weights(checkpoint_path)

    def test(self, input_str, start_seq_index):
        #getting the encoded str
        encoded_states = self.encoder_model.predict(input_str)

        self.target_seq = np.zeros ( (1 , 1 , self.out_len) )
        self.target_seq[ 0 , 0 , start_seq_index ] = 1	

        #getting the output str
        output_str, _, _ = self.decoder_model([target_seq] + encoded_states)
        return output_str



