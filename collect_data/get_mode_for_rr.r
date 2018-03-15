#! /usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

#library(data.table)
library(plyr)

data <- read.table(args[1], header = FALSE, sep = ',')
colnames(data) <- c('sid', 's1', 's2', 'result')

#data <- data.table(data)
data$combined <- paste(data$sid, data$s1, data$s2)

data_mode <- ddply(data, 'combined', summarise, mode = as.integer(names(which.max(table(result)))))
data_merge <- merge(data, data_mode, by = 'combined')

data_out <-data_merge[,c('sid', 's1', 's2', 'mode')]
data_out <- unique(data_out)

write.table(data_out, file = args[1], row.names = FALSE, col.name = FALSE, sep = ',')

