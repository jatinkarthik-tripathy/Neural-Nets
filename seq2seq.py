import tensorflow as tf 
import os
from tensorflow import keras

class seq2seq:
	def create_models(self, inp_len, out_len):
		
		# encoder
		enc_inp = tf.keras.layers.Input(shape = (None, inp_len))
		encoder = tf.keras.layers.CuDNNLSTM(units = 128, return_state=True, go_backwards = True)
		enc_outputs, enc_state_h, enc_state_c = encoder(enc_inp)

		# decoder
		dec_inp = tf.keras.layers.Input(shape = (None, out_len))
		decoder = tf.keras.layers.CuDNNLSTM(units = 128, return_state=True, return_sequences = True)
		dec_outputs, dec_state_h, dec_state_c = decoder(dec_inp, initial_state = [enc_state_h, enc_state_c])

		# final layer
		final_dense = tf.keras.layers.Dense(out_len, activation = 'softmax')
		dec_outputs = final_dense(dec_outputs)
		
		# putting the model together		
		self.train_model = tf.keras.models.Model(inputs = [enc_inp, dec_inp], outputs = dec_outputs)
		self.train_model.compile(optimizer = 'adam', loss='categorical_crossentropy')

		#creating the inference model
		self.encoder_model = tf.keras.models.Model(enc_inp, [enc_state_h, enc_state_c])

		decoder_state_input_h = Input(shape=(128,))
		decoder_state_input_c = Input(shape=(128,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		decoder_outputs, state_h, state_c = decoder(dec_inp, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


	def train_fit(self, enc_in, dec_in, dec_out, batch = 100, epochs = 1):
		checkpoint_path = "training/cp.ckpt"
		checkpoint_dir = os.path.dirname(checkpoint_path)
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
		self.train_model.fit([enc_in, dec_in], dec_out, batch_size = batch, epochs = epochs, callbacks = [cp_callback])
		self.trainingModel.save('s2s.h5')


