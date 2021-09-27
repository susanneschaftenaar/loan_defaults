library(tm)

title <- lc_features_imp %>% 
  filter(data_source == "train") %>% 
  select(emp_title)

title_corpus <- VCorpus(VectorSource(title))

print(title_corpus)
inspect(title_corpus[1:2])

# clean up the corpus using tm_map()
title_corpus_clean <- tm_map(title_corpus, content_transformer(tolower))

as.character(title_corpus[[1]])
as.character(title_corpus_clean[[1]])

title_corpus_clean <- tm_map(title_corpus_clean, removeNumbers) # remove numbers
title_corpus_clean <- tm_map(title_corpus_clean, removeWords, stopwords()) # remove stop words
title_corpus_clean <- tm_map(title_corpus_clean, removePunctuation) # remove punctuation


library(wordcloud)
png("wordcloud.png")
wordcloud(title_corpus_clean, min.freq = 1000, 
          random.order = FALSE, max.words = 40,
          colors=brewer.pal(8, "Dark2"))
dev.off()
