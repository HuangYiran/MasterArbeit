#! /usr/bin/env Rscript
library(data.table)
library('plyr')
library('stringr')

args <- commandArgs(trailingOnly = TRUE)

data <- read.table(args[1], header = TRUE, sep = ",")

data_ende <- subset(data, data$srclang == args[2] & data$trglang == args[3])
print(length(data_ende$srclang))
get_sub <- function(x){
    tmp = str_to_lower(str_sub(x, 1, 2))
    if(tmp == 'ge'){
        tmp = 'de'
    }
    tmp
}
paar = str_c(get_sub(args[2]), '-', get_sub(args[3]))

data_ende_12 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system1Id', 'system2Id', 'system1rank', 'system2rank')]
data_ende_13 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system1Id', 'system3Id', 'system1rank', 'system3rank')]
data_ende_14 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system1Id', 'system4Id', 'system1rank', 'system4rank')]
data_ende_15 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system1Id', 'system5Id', 'system1rank', 'system5rank')]
data_ende_23 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system2Id', 'system3Id', 'system2rank', 'system3rank')]
data_ende_24 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system2Id', 'system4Id', 'system2rank', 'system4rank')]
data_ende_25 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system2Id', 'system5Id', 'system2rank', 'system5rank')]
data_ende_34 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system3Id', 'system4Id', 'system3rank', 'system4rank')]
data_ende_35 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system3Id', 'system5Id', 'system3rank', 'system5rank')]
data_ende_45 = data_ende[c('srclang', 'trglang', 'srcIndex', 'system4Id', 'system5Id', 'system4rank', 'system5rank')]

colnames(data_ende_12) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_13) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_14) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_15) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_23) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_24) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_25) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_34) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_35) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')
colnames(data_ende_45) = c('srclang', 'trglang', 'segmentId', 'system1Id', 'system2Id', 'system1rank', 'system2rank')

data_ende_stacked = rbind(data_ende_12,
                          data_ende_13,
                          data_ende_14,
                          data_ende_15,
                          data_ende_23,
                          data_ende_24,
                          data_ende_25,
                          data_ende_34,
                          data_ende_35,
                          data_ende_45)
print(paar)
data_ende_stacked = data_ende_stacked[str_detect(data_ende_stacked$system1Id, paar)&str_detect(data_ende_stacked$system2Id, paar),]

write.table(data_ende_stacked, "tmp/tmp.wmt14.stacked.csv", row.names = FALSE, sep = ",")
