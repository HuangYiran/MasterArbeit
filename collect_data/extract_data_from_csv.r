#! /usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

data <- read.table(args[1], header = TRUE)
# rename the colnames, only used when the origin colnames is false
colnames(data) = c('sys_id', 'sid', 'score', 'score2', 'num')

# sys_data = subset(data, data$type == "SYSTEM", select = c(sys_id, sid, score))
sys_data = subset(data, select = c(sys_id, sid, score))

save.file = function(id){
  tmp = subset(sys_data, sys_data$sys_id == id, select = c(sid, score))
  tmp = tmp[order(tmp$sid)]
  write.table(tmp, file = id, row.names = FALSE, col.names = FALSE, sep = ",")
}

sys_id = unique(sys_data$sys_id)
sys_id = as.character(sys_id)

setwd('./tmp')
write.table(sys_id, file = "tmp_system_names", row.names = FALSE, col.names= FALSE, sep = ',')

library("data.table")
sys_data = data.table(sys_data)

counter = 1
nu = length(sys_id)
while(counter <= nu ){
  save.file(sys_id[counter])
  counter = counter + 1
}

setwd('../')
