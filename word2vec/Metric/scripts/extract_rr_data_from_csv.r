#! /usr/bin/env Rscript
library(data.table)
args <- commandArgs(trailingOnly = TRUE)

data <- read.table(args[1], header = TRUE, sep = ",")

data_deen = data[c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')]

colnames(data_deen) = c('src', 'trg', 'sid', 's1', 's2', 'r1', 'r2')

data_deen = subset(data_deen, data_deen$src == args[2]& data_deen$trg == args[3])

# compare the rank to get the winner
sub = data_deen$r2 - data_deen$r1
abs = abs(sub)
div = sub/abs
div[is.na(div)]= 0
data_deen$result = div

data_deen = data_deen[c('sid', 's1', 's2', 'result')]

save.file = function(data, name){
    data = data.table(data)
    tmp = data[order(data$sid)]
    write.table(tmp, file = name, row.names = FALSE, col.name = FALSE, sep = ",")
}
pwd = getwd()
setwd('./tmp/')
save.file(data_deen, 'extracted_deen_2016')
setwd(pwd)
