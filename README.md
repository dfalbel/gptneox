
<!-- README.md is generated from README.Rmd. Please edit that file -->

# gptneox

<!-- badges: start -->
<!-- badges: end -->

gptneox is an R torch implementation of GPTNeoX. It follows closely the
[implementation](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
available in HuggingFace and can be used to load pre-trained models
defined there.

## Installation

You can install the development version of gptneox like so:

``` r
remotes::install_github("dfalbel/gptneox")
```

Note: This package requires [`tok`](https://github.com/dfalbel/tok),
which in turns requires a Rust installation to be installed. Follow
instruction in the [`tok` repository](https://github.com/dfalbel/tok) to
get it installed.

## Example

Here’s an example using the StabilityAI 3B parameters model. Note that
this model is not tuned to provide chat like completions, thus you
should write prompts that look more like autocomplete queries. You can
load other GPTNeoX models, including those that have been trained for
chat like completions.

``` r
library(gptneox)
torch::torch_manual_seed(1)

repo <- "stabilityai/stablelm-base-alpha-3b"
model <- gpt_neox_from_pretrained(repo)
tok <- gpt_neox_tokenizer_from_pretrained(repo)

gen <- gpt_neox_generate(
  model, tok, 
  "The R language was created", 
  config = list(max_new_tokens = 100)
)
#> The R language was created to assist the human brain in learning and storing and manipulating large collections of information: the brain, a computer program that can store data, and the web (Google Inc. Google Inc., 2010). The R Language was initially intended to be used in English to address the problem of storing information. This may be modified by other computer languages. However, most of this new language was created to help the brain be able to store and analyze the contents of a database. The database is a collection of records containing
```
