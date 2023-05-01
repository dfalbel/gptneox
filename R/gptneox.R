#' GPTNeox
#' @name neox
#' @import torch
#' @importFrom zeallot %<-%
#' @include model-utils.R
NULL


#' GPTNeoX modules
#'
#' @inheritParams gpt_neox_from_config
#' @describeIn GPTNeoXAttention Attention module
#' @export
GPTNeoXAttention <- nn_module(
  "GPTNeoXAttention",
  initialize = function(config) {
    self$num_attention_heads <- config$num_attention_heads
    self$hidden_size <- config$hidden_size
    self$head_size <- floor(self$hidden_size / self$num_attention_heads)
    self$rotary_ndims <- as.integer(self$head_size * config$rotary_pct)
    max_positions <- config$max_position_embeddings

    self$register_buffer(
      "bias",
      torch_tril(torch_ones(max_positions, max_positions, dtype=torch_bool()))$view(
        list(1, 1, max_positions, max_positions)
      )
    )

    self$register_buffer("masked_bias", torch_tensor(-1e9))

    self$rotary_emb <- RotaryEmbedding(
      self$rotary_ndims, config$max_position_embeddings, base=config$rotary_emb_base
    )
    self$norm_factor <- torch_sqrt(torch_tensor(self$head_size, dtype=torch_float32()))$to(torch_get_default_dtype())
    self$query_key_value <- nn_linear(config$hidden_size, 3 * config$hidden_size)
    self$dense <- nn_linear(config$hidden_size, config$hidden_size)
  },
  forward = function(hidden_states,
                     attention_mask,
                     position_ids,
                     head_mask = NULL,
                     layer_past = NULL,
                     use_cache = FALSE,
                     output_attentions = FALSE) {
    has_layer_past <- !is.null(layer_past)

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv <- self$query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    sze <- qkv$size()
    new_qkv_shape <- c(sze[-length(sze)], self$num_attention_heads, 3 * self$head_size)
    qkv <- qkv$view(new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query <- qkv[.., 1:self$head_size]$permute(c(1, 3, 2, 4))
    key <- qkv[.., (self$head_size+1):(2 * self$head_size)]$permute(c(1, 3, 2, 4))
    value <- qkv[.., (2 * self$head_size + 1):N]$permute(c(1, 3, 2, 4))

    # Compute rotary embeddings on rotary_ndims
    query_rot <- query[.., 1:self$rotary_ndims]
    query_pass <- query[.., (self$rotary_ndims+1):N]
    key_rot <- key[.., 1:self$rotary_ndims]
    key_pass <- key[.., (self$rotary_ndims+1):N]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len <- rev(key$shape)[2]
    if (has_layer_past) {
      seq_len <- seq_len + rev(layer_past[[1]]$shape)[2]
    }

    c(cos, sin) %<-% self$rotary_emb(value, seq_len=seq_len)
    c(query, key) %<-% apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query <- torch_cat(list(query, query_pass), dim=-1)
    key <- torch_cat(list(key, key_pass), dim=-1)

    # Cache QKV values
    if (has_layer_past) {
      past_key <- layer_past[[1]]
      past_value <- layer_past[[2]]
      key <- torch_cat(list(past_key, key), dim=-2)
      value <- torch_cat(list(past_value, value), dim=-2)
    }
    present <- if (use_cache) list(key, value) else NULL

    # Compute attention
    c(attn_output, attn_weights) %<-% self$.attn(query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output <- self$.merge_heads(attn_output, self$num_attention_heads, self$head_size)
    attn_output <- self$dense(attn_output)

    outputs <- list(attn_output, present)
    if (!is.null(output_attentions) && output_attentions) {
      outputs[[3]] <- attn_weights
    }

    outputs
  },
  .split_heads = function(tensor, num_attention_heads, attn_head_size) {
    # tensor: [bs, seq_len, hidden_size]
    new_shape <- c(tensor$size()[-tensor$ndim], num_attention_heads, attn_head_size)
    # -> [bs, seq_len, num_attention_heads, attn_head_size]
    tensor <- tensor$view(new_shape)
    # -> [bs, num_attention_heads, seq_len, attn_head_size]
    tensor <- tensor$permute(c(1, 3, 2, 4))
    tensor
  },
  .merge_heads = function(tensor, num_attention_heads, attn_head_size) {
    # tensor [bs, num_attention_heads, seq_len, attn_head_size]
    tensor <- tensor$permute(c(1, 3, 2, 4))$contiguous()
    # -> [bs, seq_len, num_attention_heads, attn_head_size]
    tensor <- tensor$view(c(tensor$size(1), tensor$size(2), num_attention_heads * attn_head_size))
    # -> [bs, seq_len, hidden_size]
    tensor
  },
  .attn = function(query, key, value, attention_mask=NULL, head_mask=NULL) {
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    # compute causal mask from causal mask buffer
    c(batch_size, num_attention_heads, query_length, attn_head_size) %<-% query$size()
    key_length <- key$size(-2)

    causal_mask <- self$bias[, , (key_length - query_length + 1):key_length, 1:key_length]

    query <- query$view(c(batch_size * num_attention_heads, query_length, attn_head_size))
    key <- key$view(c(batch_size * num_attention_heads, key_length, attn_head_size))

    attn_scores <- torch_zeros(
      batch_size * num_attention_heads,
      query_length,
      key_length,
      dtype=query$dtype,
      device=key$device,
    )
    attn_scores <- torch_baddbmm(
      attn_scores,
      query,
      key$transpose(2, 3),
      beta=1.0,
      alpha=(torch_tensor(1.0, dtype=self$norm_factor$dtype, device=self$norm_factor$device) / self$norm_factor)
    )
    attn_scores <- attn_scores$view(c(batch_size, num_attention_heads, query_length, key_length))

    mask_value <- torch_finfo(attn_scores$dtype)$min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value <- torch_tensor(mask_value, dtype=attn_scores$dtype)$to(device=attn_scores$device)
    attn_scores <- torch_where(causal_mask, attn_scores, mask_value)

    if (!is.null(attention_mask)) {
      # Apply the attention mask
      attn_scores <- attn_scores + attention_mask
    }

    attn_weights <- nnf_softmax(attn_scores, dim=-1)
    attn_weights <- attn_weights$to(dtype=value$dtype)

    # Mask heads if we want to
    if (!is.null(head_mask)) {
      attn_weights <- attn_weights * head_mask
    }

    attn_output <- torch_matmul(attn_weights, value)
    list(attn_output, attn_weights)
  }
)

RotaryEmbedding <- nn_module(
  "RotaryEmbedding",
  initialize = function(dim, max_position_embeddings, base=10000, device=NULL) {
    # we might need 0-(dim-1) instead.
    inv_freq <- 1.0 / (base ^ (torch_arange(0, dim-1, 2)$float()$to(device=device) / dim))
    self$register_buffer("inv_freq", inv_freq)

    # Build here to make `torch.jit.trace` work.
    self$max_seq_len_cached <- max_position_embeddings
    t <- torch_arange(start = 0, end = self$max_seq_len_cached-1, device=self$inv_freq$device, dtype=self$inv_freq$dtype)
    freqs <- torch_einsum("i,j->ij", list(t, self$inv_freq))
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb <- torch_cat(list(freqs, freqs), dim=-1)
    self$cos_cached <- emb$cos()[newaxis, newaxis, , ]
    self$sin_cached <- emb$sin()[newaxis, newaxis, , ]
  },
  forward = function(x, seq_len = NULL) {
    if (!is.null(seq_len) && seq_len > self$max_seq_len_cached) {
      self$max_seq_len_cached <- seq_len
      t <- torch_arange(start = 1, end = self$max_seq_len_cached, device=x$device, dtype=self$inv_freq$dtype)
      freqs <- torch_einsum("i,j->ij", t, self$inv_freq)
      # Different from paper, but it uses a different permutation in order to obtain the same calculation
      emb <- torch_cat(list(freqs, freqs), dim=-1)$to(device = x$device)
      self$cos_cached <- emb$cos()[newaxis, newaxis, , ]
      self$sin_cached <- emb$sin()[newaxis, newaxis, , ]
    }
    list(
      self$cos_cached[1:seq_len, ..]$to(device=x$device),
      self$sin_cached[1:seq_len, ..]$to(device=x$device)
    )
  }
)

apply_rotary_pos_emb <- function(q, k, cos, sin, position_ids) {
  gather_indices <- position_ids[, newaxis, ,newaxis]  # [bs, 1, seq_len, 1]
  gather_indices <- gather_indices$`repeat`(c(1, cos$shape[2], 1, cos$shape[4]))
  cos <- torch_gather(cos$`repeat`(c(gather_indices$shape[1], 1, 1, 1)), 3, gather_indices)
  sin <- torch_gather(sin$`repeat`(c(gather_indices$shape[1], 1, 1, 1)), 3, gather_indices)
  q_embed <- (q * cos) + (rotate_half(q) * sin)
  k_embed <- (k * cos) + (rotate_half(k) * sin)
  list(q_embed, k_embed)
}

rotate_half <- function(x) {
  x1 <- x[.., 1:floor(x$shape[x$ndim] / 2)]
  x2 <- x[.., (floor(x$shape[x$ndim] / 2) + 1):N]
  torch_cat(list(-x2, x1), dim=-1)
}

#' @describeIn GPTNeoXAttention MLP layer module
#' @export
GPTNeoXMLP <- nn_module(
  "GPTNeoXMLP",
  initialize = function(config) {
    self$dense_h_to_4h <- nn_linear(config$hidden_size, config$intermediate_size)
    self$dense_4h_to_h <- nn_linear(config$intermediate_size, config$hidden_size)
    self$act <- ACT2FN[[config$hidden_act]]
  },
  forward = function(hidden_states) {
    hidden_states <- self$dense_h_to_4h(hidden_states)
    hidden_states <- self$act(hidden_states)
    hidden_states <- self$dense_4h_to_h(hidden_states)
    hidden_states
  }
)

#' @describeIn GPTNeoXAttention Layer module
#' @export
GPTNeoXLayer <- nn_module(
  "GPTNeoXLayer",
  initialize = function(config) {
    self$use_parallel_residual <- config$use_parallel_residual
    self$input_layernorm <- nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$post_attention_layernorm <- nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$attention <- GPTNeoXAttention(config)
    self$mlp <- GPTNeoXMLP(config)
  },
  forward = function(hidden_states,
                     attention_mask = NULL,
                     position_ids = NULL,
                     head_mask = NULL,
                     use_cache = FALSE,
                     layer_past = NULL,
                     output_attentions = FALSE) {
    attention_layer_outputs <- self$attention(
      self$input_layernorm(hidden_states),
      attention_mask=attention_mask,
      position_ids=position_ids,
      layer_past=layer_past,
      head_mask=head_mask,
      use_cache=use_cache,
      output_attentions=output_attentions
    )
    attn_output <- attention_layer_outputs[[1]]  # output_attn: attn_output, present, (attn_weights)
    outputs <- attention_layer_outputs[-1]

    if (self$use_parallel_residual) {
      # pseudocode:
      # x = x + attn(ln1(x)) + mlp(ln2(x))
      mlp_output <- self$mlp(self$post_attention_layernorm(hidden_states))
      hidden_states <- mlp_output + attn_output + hidden_states
    } else {
      # pseudocode:
      # x = x + attn(ln1(x))
      # x = x + mlp(ln2(x))
      attn_output <- attn_output + hidden_states
      mlp_output <- self$mlp(self$post_attention_layernorm(attn_output))
      hidden_states <- mlp_output + attn_output
    }
    if (use_cache) {
      outputs <- c(hidden_states, outputs)  # hidden_states, present, (attn_weights)
    } else {
      outputs <- c(hidden_states, outputs[-1])  # hidden_states, (attn_weights)
    }

    outputs
  }
)

#' @describeIn GPTNeoXAttention Model module
#' @export
GPTNeoXModel <- nn_module(
  "GPTNeoXModel",
  inherit = module_utils_mixins,
  initialize = function(config) {
    self$config <- config

    self$embed_in <- nn_embedding(config$vocab_size, config$hidden_size)
    self$layers <- nn_module_list(modules = lapply(
      seq_len(config$num_hidden_layers),
      function(i) GPTNeoXLayer(config)
    ))
    self$final_layer_norm <- nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$gradient_checkpointing <- FALSE
  },
  get_input_embeddings = function() {
    self$embed_in
  },
  set_input_embeddings = function(value) {
    self$embed_in <- value
  },
  forward = function(input_ids = NULL,
                     attention_mask = NULL,
                     position_ids = NULL,
                     head_mask = NULL,
                     inputs_embeds = NULL,
                     past_key_values = NULL,
                     use_cache = NULL,
                     output_attentions = NULL,
                     output_hidden_states = NULL,
                     return_dict = NULL) {
    output_attentions <- if (!is.null(output_attentions)) output_attentions else self$config$output_attentions
    output_hidden_states <- if (!is.null(output_hidden_states)) output_hidden_states else self$config$output_hidden_states
    return_dict <- if (!is.null(return_dict)) return_dict else self$config$return_dict
    use_cache <- if(!is.null(use_cache)) use_cache else self$config$use_cache

    if (!is.null(input_ids) && !is.null(inputs_embeds)) {
      cli::cli_abort("You cannot specify both input_ids and inputs_embeds at the same time")
    } else if (!is.null(input_ids)) {
      input_shape <- input_ids$size()
    } else if (!is.null(input_embeds)) {
      input_shape <- input_embeds$size()[-(input_embeds$ndim)]
    } else {
      cli::cli_abort("You have to specify either input_ids or inputs_embeds")
    }

    c(batch_size, seq_length) %<-% input_shape

    if (is.null(past_key_values)) {
      past_length <- 1
      past_key_values <- rep(list(NULL), self$config$num_hidden_layers)
    } else {
      past_length <- past_key_values[1][1]$size(-2) + 1
    }

    if (is.null(position_ids)) {
      device <- if (!is.null(input_ids)) input_ids$device else inputs_embeds$device
      position_ids <- torch_arange(past_length, seq_length + past_length, dtype=torch_long(), device=device)
      position_ids <- position_ids$unsqueeze(1)$view(c(-1, seq_length))
    } else {
      position_ids <- position_ids$view(c(-1, seq_length))$long()
    }

    # Attention mask.
    if (!is.null(attention_mask)) {
      if (is.null(batch_size) || !batch_size > 0) {
        cli::cli_abort("batch_size has to be defined and > 0")
      }
      attention_mask <- attention_mask$view(c(batch_size, -1))

      # We create a 3D attention mask from a 2D tensor mask.
      # Sizes are [batch_size, 1, 1, to_seq_length]
      # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
      # this attention mask is more simple than the triangular masking of causal attention
      # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
      attention_mask <- attention_mask[, newaxis, newaxis, ]

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and the dtype's smallest value for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_mask <- attention_mask$to(dtype=self$dtype)  # fp16 compatibility
      attention_mask <- (1.0 - attention_mask) * torch_finfo(attention_mask$dtype)$min
    }

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask <- super$get_head_mask(head_mask, self$config$num_hidden_layers)

    if (is.null(inputs_embeds)) {
      inputs_embeds <- self$embed_in(input_ids)
    }

    hidden_states <- inputs_embeds

    # if self.gradient_checkpointing and self.training:
    #   if use_cache:
    #   logger.warning(
    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #   )
    # use_cache = False

    presents <- if (use_cache) list() else NULL
    all_attentions = if (!is.null(output_attentions) && output_attentions) list() else NULL
    all_hidden_states = if (!is.null(output_hidden_states) && output_hidden_states) list() else NULL

    for (i in seq_len(length(self$layers))) {
      layer <- self$layers[[i]]
      layer_past <- past_key_values[[i]]

      if (!is.null(output_hidden_states) && output_hidden_states) {
        all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
      }

      outputs = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask[[i]],
        layer_past=layer_past,
        use_cache=use_cache,
        output_attentions=output_attentions
      )

      hidden_states <- outputs[[1]]
      if (use_cache) {
        presents[[length(presents) + 1]] <- outputs[[2]]
      }

      if (!is.null(output_attentions) && output_attentions) {
        all_attentions[[length(all_attentions) + 1]] <- outputs[[use_cache + 2]]
      }
    }

    hidden_states <- self$final_layer_norm(hidden_states)
    # Add last hidden state
    if (!is.null(output_hidden_states) && output_hidden_states) {
      all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
    }

    if (!is.null(return_dict) && return_dict) {
      return(list(
        hidden_states,
        presents,
        all_hidden_states,
        all_attentions
      ))
    }

    list(
      last_hidden_state=hidden_states,
      past_key_values=presents,
      hidden_states=all_hidden_states,
      attentions=all_attentions
    )
  }
)

#' @describeIn GPTNeoXAttention CausalLM
#' @export
GPTNeoXForCausalLM <- nn_module(
  "GPTNeoXForCausalLM",
  initialize = function(config) {
    self$config <- config
    self$gpt_neox <- GPTNeoXModel(config)
    self$embed_out <- nn_linear(config$hidden_size, config$vocab_size, bias = FALSE)
  },
  forward = function(input_ids = NULL,
                     attention_mask = NULL,
                     position_ids = NULL,
                     inputs_embeds = NULL,
                     head_mask = NULL,
                     past_key_values = NULL,
                     labels = NULL,
                     use_cache = NULL,
                     output_attentions = NULL,
                     output_hidden_states = NULL,
                     return_dict = NULL) {
    return_dict = if (!is.null(return_dict)) return_dict else self$config$use_return_dict

    outputs <- self$gpt_neox(
      input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      past_key_values=past_key_values,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict
    )

    hidden_states <- outputs[[1]]
    lm_logits <- self$embed_out(hidden_states)

    lm_loss <- NULL
    if (!is.null(labels)) {
      # move labels to correct device to enable model parallelism
      labels <- labels$to(device = lm_logits$device)
      # we are doing next-token prediction; shift prediction scores and input ids by one
      shift_logits <- lm_logits[, 1:(-1), ]$contiguous()
      labels <- labels[, 2:N]$contiguous()
      loss_fct <- nn_cross_entropy_loss()
      lm_loss <- loss_fct(shift_logits$view(c(-1, shift_logits$size(-1))), labels$view(-1))
    }

    if (!is.null(return_dict) && return_dict) {
      output <- c(lm_logits, outputs[-1])
      return(if (!is.null(lm_loss)) {
        list(lm_loss, output)
      } else {
        output
      })
    }

    list(
      loss=lm_loss,
      logits=lm_logits,
      past_key_values=outputs$past_key_values,
      hidden_states=outputs$hidden_states,
      attentions=outputs$attentions
    )
  }
)

ACT2FN <- list(
  "relu" = nn_relu(),
  "gelu" = nn_gelu(),
  "sigmoid" = nn_sigmoid()
)
