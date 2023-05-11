#' Defines a GPTNeoX model configuration
#'
#' @param vocab_size (int, optional, defaults to 50432) — Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling GPTNeoXModel.
#' @param hidden_size (int, optional, defaults to 6144) — Dimension of the encoder layers and the pooler layer.
#' @param num_hidden_layers (int, optional, defaults to 44) — Number of hidden layers in the Transformer encoder.
#' @param num_attention_heads (int, optional, defaults to 64) — Number of attention heads for each attention layer in the Transformer encoder.
#' @param intermediate_size (int, optional, defaults to 24576) — Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
#' @param hidden_act (str or function, optional, defaults to "gelu") — The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "selu" and "gelu_new" are supported.
#' @param rotary_pct (float, optional, defaults to 0.25) — percentage of hidden dimensions to allocate to rotary embeddings
#' @param rotary_emb_base (int, optional, defaults to 10000) — base for computing rotary embeddings frequency
#' @param max_position_embeddings (int, optional, defaults to 2048) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
#' @param initializer_range (float, optional, defaults to 1e-5) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#' @param layer_norm_eps (float, optional, defaults to 1e-12) — The epsilon used by the layer normalization layers.
#' @param use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if config.is_decoder=True.
#' @param use_parallel_residual (bool, optional, defaults to True) — Whether to use a “parallel” formulation in each Transformer layer, which can provide a slight training speedup at large scales (e.g. 20B). Example —
#' @param ... Additional configuration options.
#'
#' @describeIn gpt_neox_config Defines the configuration of the tokenizer.
#'
#' @export
gpt_neox_config <- function(
  vocab_size=50432,
  hidden_size=6144,
  num_hidden_layers=44,
  num_attention_heads=64,
  intermediate_size=24576,
  hidden_act="gelu",
  rotary_pct=0.25,
  rotary_emb_base=10000,
  max_position_embeddings=2048,
  initializer_range=0.02,
  layer_norm_eps=1e-5,
  use_cache=TRUE,
  bos_token_id=0,
  eos_token_id=2,
  tie_word_embeddings=FALSE,
  use_parallel_residual=TRUE,
  ...
  ) {
  self <- list(...)
  self$vocab_size = vocab_size
  self$max_position_embeddings = max_position_embeddings
  self$hidden_size = hidden_size
  self$num_hidden_layers = num_hidden_layers
  self$num_attention_heads = num_attention_heads
  self$intermediate_size = intermediate_size
  self$hidden_act = hidden_act
  self$rotary_pct = rotary_pct
  self$rotary_emb_base = rotary_emb_base
  self$initializer_range = initializer_range
  self$layer_norm_eps = layer_norm_eps
  self$use_cache = use_cache
  self$tie_word_embeddings = tie_word_embeddings
  self$use_parallel_residual = use_parallel_residual
  self
}

#' @describeIn gpt_neox_config Uses a configuration defined a HF hub repository.
gpt_neox_config_from_pretrained <- function(identifier, revision = "main") {
  path <- hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)
  do.call(gpt_neox_config, config)
}

#' GPTNeoX model
#'
#' Creates a GPTNeoX Model.
#'
#' @param identifier The HuggingFace repository to download the model from.
#' @param revision The repository revision to download from. Either a branch name
#'   or a commit hash.
#' @param config A model configuration created by [gpt_neox_config()] or obtained
#'   from [gpt_neox_config_from_pretrained()]
#' @param ... Not currently used.
#'
#' @describeIn gpt_neox_from_pretrained Creates from a configuration from a HF repository.
#'
#' @export
gpt_neox_from_pretrained <- function(identifier, ..., revision = "main") {
  config <- gpt_neox_config_from_pretrained(identifier, revision)

  # now we will get the weights for that model
  # first look for a file called `WEIGHTS_NAME()`
  # if this is not found, it means that the weights are sharded, thus we get the
  # index and then load the sharded files.
  weights_path <- try(hub_download(identifier, WEIGHTS_NAME(), revision = revision), silent = TRUE)
  if (inherits(weights_path, "try-error")) {
    index_path <- hub_download(identifier, WEIGHTS_INDEX_NAME(), revision = revision)
    filenames <- unique(unlist(jsonlite::fromJSON(index_path)$weight_map))
    weights_path <- sapply(filenames, function(fname) {
      hub_download(identifier, fname, revision = revision)
    })
    names(weights_path) <- NULL
  }
  with_device(device = "meta", {
    model <- do.call(config$architectures, list(config = config))
  })
  weights <- do.call("c", lapply(weights_path, torch::load_state_dict))
  model$load_state_dict(weights, .refer_to_state_dict = TRUE)
  model
}

#' @describeIn gpt_neox_from_pretrained Creates a GPTNeoX model from a configuration list.
#' @export
gpt_neox_from_config <- function(config) {
  arch <- if (!is.null(config$architectures)) config$architectures else "GPTNeoXForCausalLM"
  model <- do.call(arch, list(config = config))
}

#' Creates a GPTNeoX tokenizer from a pre-trained tokenizer
#'
#' @inheritParams gpt_neox_from_pretrained
#'
#' @export
gpt_neox_tokenizer_from_pretrained <- function(identifier, ..., revision = "main") {
  tok_file <- hub_download(identifier, "tokenizer.json", revision = revision)
  tok::tokenizer$from_file(tok_file)
}

# snapshot_download <- function(repo_id, ..., revision = "main", repo_type = "model") {
#   cache_dir <- HUGGINGFACE_HUB_CACHE()
#   storage_folder <- fs::path(cache_dir, repo_folder_name(repo_id, repo_type))
#
#   repo_info <- api_repo_info(repo_id, revision)
#   commit_hash <- repo_info$sha
#   if (is.null(commit_hash)) cli::cli_abort("Repo info didn't return a commit sha.")
#
#   repo_files <- sapply(repo_info$siblings, function(x) x$rfilename)
#   snapshot_folder <- fs::path(storage_folder, "snapshots", commit_hash)
#
#   # we write an alias between revision and commit-hash
#   if (revision != commit_hash) {
#     ref_path <- fs::path(storage_folder, "refs", revision)
#     fs::file_create(ref_path)
#     writeLines(commit_hash, ref_path)
#   }
# }
#
# api_repo_info <- function(repo_id, revision) {
#   repo_info <- httr::GET(glue::glue("https://huggingface.co/api/models/{repo_id}/revision/{revision}"))
#   if (repo_info$status_code != 200) {
#     cli::cli_abort(c(
#       "Could not get repository information.",
#       x = httr::content(repo_info)$error
#     ))
#   }
#   httr::content(repo_info)
# }

hub_download <- function(repo_id, filename, ..., revision = "main", repo_type = "model", local_files_only = FALSE, force_download = FALSE) {
  cache_dir <- HUGGINGFACE_HUB_CACHE()
  storage_folder <- fs::path(cache_dir, repo_folder_name(repo_id, repo_type))

  # revision is a commit hash and file exists in the cache, quicly return it.
  if (grepl(REGEX_COMMIT_HASH(), revision)) {
    pointer_path <- get_pointer_path(storage_folder, revision, filename)
    if (fs::file_exists(pointer_path)) {
      return(pointer_path)
    }
  }

  url <- hub_url(repo_id, filename, revision = revision, repo_type = repo_type)

  etag <- NULL
  commit_hash <- NULL
  expected_size <- NULL

  if (!local_files_only) {
    tryCatch({
      metadata <- get_file_metadata(url)

      commit_hash <- metadata$commit_hash
      if (is.null(commit_hash)) {
        cli::cli_abort("Distant resource does not seem to be on huggingface.co (missing commit header).")
      }

      etag <- metadata$etag
      if (is.null(etag)) {
        cli::cli_abort("Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.")
      }

      # Expected (uncompressed) size
      expected_size <- metadata$size
    })
  }

  # etag is NULL == we don't have a connection or we passed local_files_only.
  # try to get the last downloaded one from the specified revision.
  # If the specified revision is a commit hash, look inside "snapshots".
  # If the specified revision is a branch or tag, look inside "refs".
  if (is.null(etag)) {
    # Try to get "commit_hash" from "revision"
    commit_hash <- NULL
    if (grepl(REGEX_COMMIT_HASH(), revision)) {
      commit_hash <- revision
    } else {
      ref_path <- fs::path(storage_folder, "refs", revision)
      if (fs::file_exists(ref_path)) {
        commit_hash <- readLines(ref_path)
      }
    }

    # Return pointer file if exists
    if (!is.null(commit_hash)) {
      pointer_path <- get_pointer_path(storage_folder, commit_hash, filename)
      if (fs::file_exists(pointer_path)) {
        return(pointer_path)
      }
    }

    if (local_files_only) {
      cli::cli_abort(paste0(
        "Cannot find the requested files in the disk cache and",
        " outgoing traffic has been disabled. To enable hf.co look-ups",
        " and downloads online, set 'local_files_only' to False."
      ))
    } else {
      cli::cli_abort(paste0(
        "Connection error, and we cannot find the requested files in",
        " the disk cache. Please try again or make sure your Internet",
        " connection is on."
      ))
    }
  }

  if (is.null(etag)) cli::cli_abort("etag must have been retrieved from server")
  if (is.null(commit_hash)) cli::cli_abort("commit_hash must have been retrieved from server")

  blob_path <- fs::path(storage_folder, "blobs", etag)
  pointer_path <- get_pointer_path(storage_folder, commit_hash, filename)

  fs::dir_create(fs::path_dir(blob_path))
  fs::dir_create(fs::path_dir(pointer_path))

  # if passed revision is not identical to commit_hash
  # then revision has to be a branch name or tag name.
  # In that case store a ref.
  # we write an alias between revision and commit-hash
  if (revision != commit_hash) {
    ref_path <- fs::path(storage_folder, "refs", revision)
    fs::dir_create(fs::path_dir(ref_path))
    fs::file_create(ref_path)
    writeLines(commit_hash, ref_path)
  }

  if (fs::file_exists(pointer_path) && !force_download) {
    return(pointer_path)
  }

  if (fs::file_exists(blob_path) && !force_download) {
    fs::link_create(blob_path, pointer_path)
    return(pointer_path)
  }

  withr::with_tempfile("tmp", {
    lock <- filelock::lock(paste0(blob_path, ".lock"))
    on.exit({filelock::unlock(lock)})
    curl::curl_download(url, tmp, quiet = !interactive())
    fs::file_move(tmp, blob_path)
    fs::link_create(blob_path, pointer_path)
  })

  pointer_path
}

hub_url <- function(repo_id, filename, ..., revision = "main", repo_type = "model") {
  glue::glue("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}")
}

get_pointer_path <- function(storage_folder, revision, relative_filename) {
  snapshot_path <- fs::path(storage_folder, "snapshots")
  pointer_path <- fs::path(snapshot_path, revision, relative_filename)
  pointer_path
}

repo_folder_name <- function(repo_id, repo_type = "model") {
  repo_id <- gsub(pattern = "/", x = repo_id, replacement = REPO_ID_SEPARATOR())
  glue::glue("{repo_type}s{REPO_ID_SEPARATOR()}{repo_id}")
}

get_file_metadata <- function(url) {
  req <- httr::HEAD(
    url = url,
    httr::add_headers("Accept-Encoding" = "identity")
  )
  list(
    commit_hash = grab_from_headers(req$all_headers, "x-repo-commit"),
    etag = normalize_etag(grab_from_headers(req$all_headers, "etag")),
    size = as.integer(grab_from_headers(req$all_headers, "content-length"))
  )
}

grab_from_headers <- function(headers, nm) {
  for(h in headers) {
    header <- h$headers
    if (!is.null(header[[nm]]))
      return(header[[nm]])
  }
  NULL
}

normalize_etag <- function(etag) {
  if (is.null(etag)) return(NULL)
  etag <- gsub(pattern = '"', x = etag, replacement = "")
  etag <- gsub(pattern = "W/", x = etag, replacement = "")
  etag
}

REPO_ID_SEPARATOR <- function() {
  "--"
}
HUGGINGFACE_HUB_CACHE <- function() {
  path <- Sys.getenv("HUGGINGFACE_HUB_CACHE", "~/.cache/huggingface/hub")
  path.expand(path)
}
REGEX_COMMIT_HASH <- function() {
  "^[0-9a-f]{40}$"
}

WEIGHTS_NAME <- function() "pytorch_model.bin"
WEIGHTS_INDEX_NAME <- function() "pytorch_model.bin.index.json"
