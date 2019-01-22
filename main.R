library(data.table)
library(tidytext)
library(magrittr)
library(dplyr)
library(kerasR)

options(scipen = 999)
set.seed(1113137)

Sys.time()

train <- fread('../input/train.csv', data.table = FALSE)
test <- fread('../input/test.csv', data.table = FALSE)

max_words <- 15000
maxlen <- 64

texts <- train$question_text
tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)
word_index <- tokenizer$word_index

data = pad_sequences(sequences, maxlen = maxlen)

embeddings <- readLines('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
embeddings_index = new.env(hash = TRUE, parent = emptyenv())
embeddings <- embeddings[2:length(embeddings)]

for (i in 1:length(embeddings)){
  embedding <- embeddings[[i]]
  values <- strsplit(embedding, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] = as.double(values[-1])
}

word_vectors = array(0, c(max_words, 300))

for (word in names(word_index)){
  index <- word_index[[word]]
  if (index < max_words){
    embedding_vector = embeddings_index[[word]]
    if (!is.null(embedding_vector))
      word_vectors[index+1,] <- embedding_vector
      # count not found words!
  }
}

labels = train$target
indices = sample(1:nrow(data))
training_indices = indices[1:nrow(data)]

x_train = data[training_indices,]
y_train = labels[training_indices]

input <- layer_input(
  shape = list(NULL),
  dtype = "int32",
  name = "input"
)

predictions <- input %>% 
    layer_embedding(input_dim = max_words, output_dim = 300, name = "embedding") %>%
    layer_lstm(units = maxlen,dropout = 0.25, recurrent_dropout = 0.25, return_sequences = FALSE, name = "lstm") %>%
    layer_dense(units = 128, activation = "relu", name = "dense") %>%
    layer_dense(units = 1, activation = "sigmoid", name = "predictions")

Sys.time()

model <- keras_model(input, predictions)

get_layer(model, name = "embedding") %>% 
  set_weights(list(word_vectors)) %>% 
  freeze_weights()

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = "binary_accuracy"
)

print(model)
Sys.time()

history <- model %>% fit(
  x_train,
  y_train,
  batch_size = 128,
  epochs = 20,
  verbose = 0
)

print(history)
plot(history)

Sys.time()
