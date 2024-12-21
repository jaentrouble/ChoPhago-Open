import keras_nlp
import keras
from keras import layers

def encoder_only_v0(
        intermediate_dim, 
        num_heads,
        num_encoder_layers,
        embed_dim, 
        max_idx, 
        board_size
    ):
    """
    Assumes final 2 elements of input are current weapons
    """
    seq_len = board_size**2 + 7
    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    token_embeddings = layers.Embedding(max_idx, embed_dim)(inputs)
    position_embeddings = keras_nlp.layers.PositionEmbedding(
        sequence_length=seq_len,
    )(token_embeddings)
    x = layers.Add()([token_embeddings, position_embeddings])
    for _ in range(num_encoder_layers):
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            normalize_first=True,
            dropout=0.1,
        )(x)
    cur_weapon_part = x[:, -2:,:]
    outputs = layers.Dense(board_size**2+1)(cur_weapon_part)
    outputs = layers.Flatten()(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)