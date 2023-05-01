#' Generate <- nn_module(
#'   initialize = function(model, ...) {
#'     # The contract is that model when called with inputs should return a named
#'     # list containing an element called 'logits'.
#'     self$model <- model
#'   },
#'   generate = function(inputs, max_tokens, ) {
#'     config <- new_generation_config(...)
#'
#'   }
#' )
#'
#'
#' #' A very small subset of generation configurations supported in HF transformers
#' #'
#' #' @importFrom rlang %||%
#' #'
#' new_generation_config <- function(...) {
#'   args <- rlang::list2(...)
#'   config <- list()
#'
#'   # Parameters that control the length of the output
#'   config$max_length = args$max_length %||% 20
#'   config$max_new_tokens = args$max_new_tokens %||% NULL
#'   config$min_length = args$min_length %||% 0
#'
#'   # Parameters that control the generation strategy used
#'   config$do_sample = args$do_sample %||% TRUE
#'
#'   # Parameters for manipulation of the model output logits
#'   self$temperature = args$temperature %||% 1.0
#'   self$top_k = args$top_k %||% 50
#' }
