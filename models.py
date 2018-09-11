import sys
import keras
import keras.layers as layers

def output_feedback(n_a, n_dense, n_y):
    encoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    decoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    flatter = layers.Flatten()
    dense = layers.Dense(units = n_dense, activation='tanh')
    dense_out = layers.Dense(units = 1, activation='linear')
    concatenator = layers.Concatenate()
    concatenator_out = layers.Concatenate()
    reshapor = layers.Reshape((1, n_a*2))
    

    x_pdf_input = layers.Input(shape=(None, 1))
    x_x_input = layers.Input(shape=(None, 1))
    
    encoder_input = concatenator([x_pdf_input, x_x_input])
    
    _, for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c = encoder_LSTM(encoder_input)  
    
    decoder_input = layers.Input(shape=(1,n_a*2))
    deco_input = decoder_input
    decoder_state = [for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c]
    
    out_sequence = list()
    progress_steps = 20
    for i in range(n_y):
        sys.stdout.write("\rFeedback: {0} out of {1}".format(str(i), str(n_y)))
        sys.stdout.flush()
        
        decoder_output, for_deco_h, for_deco_c, back_deco_h, back_deco_c = decoder_LSTM(deco_input, initial_state=decoder_state)
        decoder_output = reshapor(decoder_output)
        decoder_output_flat = flatter(decoder_output)
        out = dense(decoder_output_flat)
        out = dense_out(out)
        
        out_sequence.append(out)
        
        deco_input = decoder_output
        decoder_state = [for_deco_h, for_deco_c, back_deco_h, back_deco_c]
        
    print()
    out_sequence = concatenator_out(out_sequence)
    model = keras.models.Model(inputs=[x_pdf_input, x_x_input, decoder_input], outputs=out_sequence)
    return model



def state_cell_only(n_a, n_dense, drop_rate, n_y):
    encoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    decoder_LSTM_L1 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    decoder_LSTM_L2 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    flatter = layers.Flatten()
    dense_L1 = layers.Dense(units = n_dense, activation='tanh')
    dense_L2 = layers.Dense(units = n_dense, activation='tanh')
    dropout = layers.Dropout(rate=drop_rate)
    dense_out = layers.Dense(units = n_y, activation='linear')
    concatenator = layers.Concatenate()

    
    x_pdf_input = layers.Input(shape=(None, 1))
    x_x_input = layers.Input(shape=(None, 1))
    
    encoder_input = concatenator([x_pdf_input, x_x_input])
    
    _, for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c = encoder_LSTM(encoder_input)  
    
    decoder_input = layers.Input(shape=(1,n_a))
    decoder_state = [for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c]

    decoder_output = decoder_LSTM_L1(decoder_input, initial_state=decoder_state)
    decoder_output = decoder_LSTM_L2(decoder_output, initial_state=decoder_state)
    decoder_output_flat = flatter(decoder_output)
    out = dense_L1(decoder_output_flat)
    out = dropout(out)
    out = dense_L2(out)
    out = dropout(out)
    out = dense_out(out)
    
    model = keras.models.Model(inputs=[x_pdf_input, x_x_input, decoder_input], outputs=out)
    return model