import tensorflow as tf
from tensorflow.keras import layers
import math

class InputEmbeddings(tf.keras.layers.Layer):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.embedding=layers.Embedding(input_dim=vocab_size,output_dim=d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,d_model:int,seq_len:int,dropout:int) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=layers.Dropout(dropout)


        position=tf.range(start=0,limit=seq_len,dtype=tf.float32)
        position=tf.expand_dims(position,axis=-1)
        div_term=tf.range(0,d_model,dtype=tf.float32)
        base_tensor=tf.fill([d_model], 10000.)
        div_term=tf.pow(base_tensor,tf.multiply(div_term,2/d_model))
        pe=position/div_term
        cols = tf.range(tf.shape(pe)[1])  # Column indices
        odd_mask = tf.cast(cols % 2 == 1, tf.float32)  # 1 for odd indices, 0 for even
        even_mask = tf.cast(cols % 2 == 0, tf.float32)
        pe = tf.cos(pe) * odd_mask + tf.sin(pe) * even_mask

    def forward(self, x, training=False):
        seq_len = tf.shape(x)[1]  # Get current sequence length
        x = x + tf.stop_gradient(self.pe[:, :seq_len, :])  # Stop gradients for PE
        return self.dropout(x, training=training)

class LayerNormalisation(tf.keras.layers.Layer):
    def __init__(self,features:int,eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=tf.Variable(tf.ones(shape=(features,)), trainable=True, dtype=tf.float32)
        self.bias=tf.Variable(tf.zeros(shape=(features,)),trainable=True,dtype=tf.float32)

    def forward(self,x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)

        return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear_1=layers.Dense(d_ff,activation="relu")
        self.dropout=layers.Dropout(dropout)
        self.linear_2=layers.Dense(d_model)

    def forward(self,x,training=True):
        x = self.linear_1(x)  # (batch, seq_len, d_model) --> (batch, seq_len, d_ff)
        x = self.dropout(x, training=training)
        x = self.linear_2(x)  # (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return x

class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.h=h

        assert d_model%h==0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = layers.Dense(d_model,use_bias=False) # Wq
        self.w_k=layers.Dense(d_model,use_bias=False) #Wk
        self.w_v=layers.Dense(d_model,use_bias=False) #Wv
        self.w_o=layers.Dense(d_model,use_bias=False) #Wo
        self.dropout=layers.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout=layers.Dropout):
        d_k = tf.shape(query)[-1]
        attention_scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / tf.math.sqrt(tf.cast(d_k, tf.float32))
        attention_scores = tf.where(mask == 0, tf.constant(-1e9, dtype=attention_scores.dtype), attention_scores)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return tf.matmul(attention_scores, value), attention_scores

    
    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        query = tf.reshape(query, (query.shape[0], query.shape[1], self.h, self.d_k))
        query = tf.transpose(query, perm=[0, 2, 1, 3])

        key = tf.reshape(key, (key.shape[0], key.shape[1], self.h, self.d_k))
        key = tf.transpose(key, perm=[0, 2, 1, 3])

        value = tf.reshape(value, (value.shape[0], value.shape[1], self.h, self.d_k))
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = tf.transpose(x, perm=[0, 2, 1, 3])

        # Step 2: Reshape to merge attention heads -> (batch, seq_len, d_model)
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.d_model))

        return self.w_o(x)

class ResidualConnection(tf.keras.layers.Layer):
    def __init__(self,features:int,dropout:float)->None:
        super().__init__()
        self.dropout=dropout
        self.norm=LayerNormalisation(features)
    
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [ResidualConnection(features, dropout) for _ in range(2)]
    
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self,features:int,layers)->None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation(features)
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection= [ResidualConnection(features, dropout) for _ in range(3)]
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(tf.keras.layers.Layer):

    def __init__(self, features: int, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = layers.Dense(vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(tf.keras.layers.Layer):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer
    
    def build(self, input_shape):
        """Initialize weights with Xavier Uniform if they have more than 1 dimension."""
        for weight in self.trainable_variables:
            if len(weight.shape) > 1:  # Equivalent to PyTorch's `p.dim() > 1`
                weight.assign(tf.keras.initializers.GlorotUniform()(weight.shape))


    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
     # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, encoder_blocks)
    decoder = Decoder(d_model, decoder_blocks)
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    # No changes
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

