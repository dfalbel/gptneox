test_that("gptneox 3B", {

  config <- gpt_neox_config(
      bos_token_id = 0,
      eos_token_id = 0,
      hidden_act = "gelu",
      hidden_size = 4096,
      initializer_range = 0.02,
      intermediate_size = 16384,
      layer_norm_eps = 1e-05,
      max_position_embeddings = 4096,
      model_type = "gpt_neox",
      num_attention_heads = 32,
      num_hidden_layers = 16,
      rotary_emb_base = 10000,
      rotary_pct = 0.25,
      tie_word_embeddings = FALSE,
      torch_dtype = "float32",
      transformers_version = "4.27.4",
      use_cache = TRUE,
      use_parallel_residual = TRUE,
      vocab_size = 50688
  )
  #pars: 3.637.321.728
  model <- gpt_neox_for_causal_lm(config)
  n_par <- Reduce(model$parameters, init = 0, f = function(x, y) {
    y$numel() + x
  })

  # number of parameters in https://huggingface.co/stabilityai/stablelm-base-alpha-3b
  expect_equal(n_par, 3637321728)

  tokenizer = tok::tokenizer$from_pretrained("StabilityAI/stablelm-base-alpha-3b")

  encoding <- tokenizer$encode("What's your mood today?")

  expect_equal(
    encoding$ids,
    c(1276,   434,   634, 12315,  3063,    32)
  )

  expect_equal(
    encoding$attention_mask,
    c(1, 1, 1, 1, 1, 1)
  )

  inputs <- torch_tensor(encoding$ids + 1L)$unsqueeze(1)
  mask <- torch_tensor(encoding$attention_mask)$unsqueeze(1)

  expect_equal(inputs$shape, c(1, 6))
  expect_equal(mask$shape, c(1, 6))

  model$eval()
  with_no_grad({
    out <- model(inputs, attention_mask = mask)
  })

  expect_equal(out$logits$shape, 1, 6, 50688)

  model$load_state_dict(c(
    load_state_dict("~/Downloads/pytorch_model-00001-of-00002.bin"),
    load_state_dict("~/Downloads/pytorch_model-00002-of-00002.bin")
  ))

  #model$load_state_dict(state_dict)
  #state_dict <- model$state_dict()

  #model$to(dtype = torch_half())
  #debugonce(model)

  #model$load_state_dict(weights)

#   model$eval()
#   with_no_grad({
#     out <- model$gpt_neox(inputs, attention_mask = mask)
#   })
#
#   text <- "An RMarkdown document showing how to create a plot of the <dataset> name:
# ```"
#   for (i in 1:500) {
#     encoding <- tokenizer$encode(text)
#     inputs <- torch_tensor(encoding$ids + 1L)$unsqueeze(1)
#     mask <- torch_tensor(encoding$attention_mask)$unsqueeze(1)
#     with_no_grad({
#       out <- model(inputs, attention_mask = mask)
#     })
#     token <- out$logits[,-1,]$topk(5)
#     token <- sample(as.integer(token[[2]]), 1, prob = as.numeric(token[[1]]))
#     new <- tokenizer$decode(as.integer(token) - 1L)
#     if (i == 1) cat(text)
#     cat(new)
#     text <- paste0(text, new)
#   }



})
