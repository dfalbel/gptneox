module_utils_mixins <- nn_module(
  initialize = function() {

  },
  get_head_mask = function(head_mask, num_hidden_layers, is_attention_chunked = FALSE) {
    if (!is.null(head_mask)) {
      head_mask <- self$.convert_head_mask_to_5d(head_mask, num_hidden_layers)
      if (is_attention_chunked) {
        head_mask <- head_mask$unsqueeze(-1)
      }
    } else {
      head_mask <- rep(list(NULL), num_hidden_layers)
    }
    return(head_mask)
  },
  .convert_head_mask_to_5d = function(head_mask, num_hidden_layers) {
    if (head_mask$dim() == 1) {
      head_mask <- head_mask$unsqueeze(1)$unsqueeze(1)$unsqueeze(-1)$unsqueeze(-1)
      head_mask <- head_mask$expand(c(num_hidden_layers, -1, -1, -1, -1))
    } else if (head_mask$dim() == 2) {
      head_mask <- head_mask$unsqueeze(1)$unsqueeze(-1)$unsqueeze(-1)  # We can specify head_mask for each layer
    }

    if (!(head_mask$dim() == 5)) {
      cli::cli_abort("head_mask.dim != 5, instead {head_mask$dim()}")
    }

    head_mask <- head_mask$to(dtype=self$dtype)  # switch to float if need + fp16 compatibility
    head_mask
  }
)
