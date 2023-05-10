
#' Samples tokens from GPT NeoX model
#'
#' @param model A GPTNeoX Model as created with [gpt_neox_from_pretrained()]
#' @param tokenizer A GPTNeoX tokenizer as created with [gpt_neox_tokenizer_from_pretrained()]
#' @param prompt A text prompt.
#' @param config Configuration for generation.
#' @param verbose Wether to print output during generation.
#'
#' @importFrom rlang %||%
#'
#' @export
gpt_neox_generate <- function(model, tokenizer, prompt, ..., config = list(), verbose = TRUE) {
  config$max_new_tokens <- config$max_new_tokens %||% 64
  config$do_sample <- config$do_sample %||% TRUE
  config$bos_token_id <- config$bos_token_id %||% 0
  config$eos_token_id <- config$eos_token_id %||% 0
  config$top_k <- config$top_k %||% 50
  config$temperature <- config$temperature %||% 1

  new_tokens <- list()
  new_text <- list()
  device <- model$gpt_neox$embed_in$weight$device

  model$eval() # model should be in eval mode
  for (i in seq_len(config$max_new_tokens)) {
    encoding <- tokenizer$encode(prompt)

    inputs <- torch_tensor(encoding$ids + 1L, device = device)$unsqueeze(1)
    mask <- torch_tensor(encoding$attention_mask, device = device)$unsqueeze(1)

    with_no_grad({
      out <- model(inputs, attention_mask = mask)
    })

    logits <- out$logits[,-1,]

    if (config$do_sample) {
      logits <- logits/config$temperature
      logits <- logits$topk(config$top_k)

      selected <- torch_multinomial(nnf_softmax(logits[[1]], dim = -1), num_samples = 1)
      token <- logits[[2]][,selected$item()]$item()
    } else {
      token <- as.integer(logits$argmax(dim = -1))
    }

    new <- tokenizer$decode(as.integer(token) - 1L)

    if (verbose) {
      if (i == 1) cat(prompt)
      cat(new)
    }

    new_tokens[[length(new_tokens) + 1]] <- token
    new_text[[length(new_text) + 1]] <- new

    if ((token - 1L) == config$eos_token_id) {
      break
    }

    prompt <- paste0(prompt, new)
  }

  if (verbose) cat("\n")

  list(
    prompt = prompt,
    new_tokens = new_tokens,
    new_text = new_text
  )
}
