library(data.table)
library(tidytext)
library(magrittr)
library(dplyr)

options(scipen = 999)

train <- fread("train.csv", data.table = FALSE)
test <- fread("test.csv", data.table = FALSE)

#Tokenizing

all_questions <- rbind(train %>% select(question_text), test %>% select(question_text))
question_texts <- all_questions$question_text
result <- tokenizers::tokenize_words(question_texts)

words = array(0);

for (i in 1:length(result)){
  for (word in result[[i]]) {
    words[i] <- word
  }
}

words = unique(words)

embeddings <- readLines('wiki-news-300d-1M.vec')
embeddings_index = new.env(hash = TRUE, parent = emptyenv())
embeddings <- embeddings[2:length(embeddings)]

for (i in 1:length(embeddings)){
  embedding <- embeddings[[i]]
  values <- strsplit(embedding, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] = as.double(values[-1])
}

word_vectors = array(0, c(82500, 300))
i = 1
for (word in words) {
  embedding_vector = embeddings_index[[word]]
  
  if (!is.null(embedding_vector)) {
    i = i + 1
    word_vectors[i,] <- embedding_vector
  }
}
