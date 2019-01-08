library(data.table)
library(tidytext)
library(magrittr)
library(dplyr)

options(scipen = 999)

train <- fread("train.csv", data.table = FALSE)
test <- fread("test.csv", data.table = FALSE)

#Tokenizing

max_words <- 15000

all_questions <- rbind(train %>% select(question_text), test %>% select(question_text))
question_texts <- all_questions$question_text
result <- tokenizers::tokenize_ngrams(question_texts, n = 1)

embeddings <- readLines('wiki-news-300d-1M.vec')
embeddings_index = new.env(hash = TRUE, parent = emptyenv())
embeddings <- embeddings[2:length(embeddings)]

for (i in 1:length(embeddings)){
  embedding <- embeddings[[i]]
  embedding <- strsplit(embedding, " ")[[1]]
  word <- embedding[[1]]
  embeddings_index[[word]] = as.double(embedding[-1])
}