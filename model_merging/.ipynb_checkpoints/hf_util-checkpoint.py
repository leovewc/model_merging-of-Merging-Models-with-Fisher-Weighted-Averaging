"""Utilities for HuggingFace."""
from typing import Tuple, Union

import tensorflow as tf
from transformers import TFBertPreTrainedModel
from transformers import TFRobertaPreTrainedModel


# def get_body_and_head(
#     model: Union[TFBertPreTrainedModel, TFRobertaPreTrainedModel]
# ) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
#     body, *head = model.layers
#     if not head:
#         head = None
#     elif len(head) > 1:
#         raise ValueError(
#             f"Expected model to have a single 'head' layer. Instead found {len(head)}. TODO: Support this."
#         )
#     else:
#         head = head[0]
#     return body, head

# def get_body_and_head(model):
#     """
#     Extracts the body (backbone) and head (task-specific layer) of the model.
#     """
#     # 确保模型权重已初始化
#     dummy_input = tf.random.uniform(
#         [1, model.config.max_position_embeddings], dtype=tf.int32, minval=0, maxval=model.config.vocab_size
#     )
#     _ = model(dummy_input)  # 调用模型，初始化权重

#     # 获取模型的主干（body）和头（head）
#     if hasattr(model, 'bert'):
#         body = model.bert
#     elif hasattr(model, 'roberta'):
#         body = model.roberta
#     else:
#         raise ValueError("Model does not have 'bert' or 'roberta' attribute.")

#     if hasattr(model, 'classifier'):
#         head = model.classifier
#     elif hasattr(model, 'classifier'):
#         head = model.classifier
#     else:
#         raise ValueError("Model does not have 'classifier' attribute.")

#     # 初始化 body 和 head 的权重
#     body_outputs = body(dummy_input)
#     # 对于 head，我们需要传递正确的输入
#     # 通常 head 期望的是 body 的输出中的 pooler_output 或 last_hidden_state
#     if isinstance(body_outputs, (tuple, list)):
#         # 处理返回多个张量的情况
#         body_output = body_outputs[0]
#     elif isinstance(body_outputs, tf.Tensor):
#         body_output = body_outputs
#     elif hasattr(body_outputs, 'pooler_output'):
#         body_output = body_outputs.pooler_output
#     elif hasattr(body_outputs, 'last_hidden_state'):
#         body_output = body_outputs.last_hidden_state
#     else:
#         raise ValueError("Cannot find suitable output from body to pass to head.")

#     # 初始化 head 的权重
#     head(body_output)

#     return body, head

#roberta
def get_body_and_head(model):
    """
    Extracts the body (backbone) and head (task-specific layer) of the model.
    """
    # Determine the maximum input sequence length (usually 512 for BERT and RoBERTa)
    max_input_length = min(512, model.config.max_position_embeddings - 2)  # Adjust if necessary

    # Generate dummy input within valid token indices
    dummy_input = tf.random.uniform(
        [1, max_input_length],
        dtype=tf.int32,
        minval=0,
        maxval=model.config.vocab_size,  # maxval is exclusive
    )

    # Initialize model weights
    _ = model(dummy_input)

    # Get the body and head
    if hasattr(model, 'roberta'):
        body = model.roberta
        # For RoBERTa, head expects features with shape (batch_size, sequence_length, hidden_size)
        use_pooler_output = False
    elif hasattr(model, 'bert'):
        body = model.bert
        # For BERT, head expects features with shape (batch_size, hidden_size)
        use_pooler_output = True
    else:
        raise ValueError("Model does not have 'roberta' or 'bert' attribute.")

    if hasattr(model, 'classifier'):
        head = model.classifier
    else:
        raise ValueError("Model does not have 'classifier' attribute.")

    # Initialize weights for body and head
    body_outputs = body(dummy_input)

    # Select appropriate body_output based on the model type
    if use_pooler_output:
        # For BERT
        body_output = body_outputs.pooler_output  # Shape: (batch_size, hidden_size)
    else:
        # For RoBERTa
        body_output = body_outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

    # Initialize the head weights
    head(body_output)

    return body, head

def get_mergeable_variables(model):
    body, _ = get_body_and_head(model)
    return body.trainable_variables


# def get_mergeable_variables(model):
#     return get_body_and_head(model)[0].trainable_variables

def get_mergeable_variables(model):
    body, _ = get_body_and_head(model)
    return body.trainable_variables

def clone_model(model):
    cloned = model.__class__(model.config)
    cloned(model.dummy_inputs)
    cloned.set_weights(model.get_weights())
    return cloned
