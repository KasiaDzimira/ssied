library(data.table)
library(tidytext)
library(magrittr)
library(dplyr)

options(scipen = 999)

train <- fread("train.csv", data.table = FALSE)
test <- fread("test.csv", data.table = FALSE)

train %>% filter(target == 1) %>% sample_n(5)

#Tokenizing

max_words <- 15000
maxlen <- 64

full <- rbind(train %>% select(question_text), test %>% select(question_text))
texts <- full$question_text
result <- tokenizers::tokenize_ngrams(texts, n = 2)
data <- pad_sequences(result)