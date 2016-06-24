require(data.table)
require(ForeCA)

load("./data/dt_data.RData")

get_hist <- function(x, breaks)
{
  a <- hist(x, breaks=breaks, plot=FALSE)
  return(a$counts / length(x)) # sum == 1
}

hist_bins <- 100
# pre alloc
hist_features     <- as.data.table( matrix(0, nrow(dt_data_origin), hist_bins))
hist_features_exp <- as.data.table( matrix(0, nrow(dt_data_exp), hist_bins))

# simulation
dp <- as.matrix(dt_data_origin[, c(-1,-2,-3), with=FALSE ]) # choose 
breaks <- seq(min(dp), max(dp), length.out = hist_bins + 1) # data range
for (i in 1:nrow(dp) )
{
  set(hist_features, 
      i, 
      names(hist_features), 
      as.list(get_hist(dp[i,], breaks)) )
}
hist_features[,c("Dim","Time","Stress") := dt_data_origin[,.(Dim,Time,Stress)] ]
setkeyv(hist_features, c("Dim","Time","Stress"))

# experiment
dp <- as.matrix(dt_data_exp[, c(-1,-2,-3), with=FALSE ]) # choose
for (i in 1:nrow(dp) )
{
  set(hist_features_exp, 
      i, 
      names(hist_features_exp), 
      as.list(get_hist(dp[i,], breaks)) )
}
hist_features_exp[,c("Dim","Time","Stress") := dt_data_exp[,.(Dim,Time,Stress)] ]
setkeyv(hist_features_exp, c("Dim","Time","Stress"))

# 从原始数据中计算标准差
tmp_matrix <- dt_data_exp[,c(-1,-2,-3), with=FALSE] # matrix is more efficient
tmp_std <- apply(tmp_matrix,1,sd)
hist_features_exp[, std_value:=tmp_std]

tmp_matrix <- dt_data_origin[,c(-1,-2,-3), with=FALSE]
tmp_std <- apply(tmp_matrix,1,sd)
hist_features[, std_value:=tmp_std]

# 计算分布密度对应的熵
tmp_matrix <- hist_features[,1:hist_bins, with=FALSE]
tmp_entropy <- apply(tmp_matrix,1,discrete_entropy)
hist_features[,entropy:=tmp_entropy]

tmp_matrix <- hist_features_exp[,1:hist_bins, with=FALSE]
tmp_entropy <- apply(tmp_matrix,1,discrete_entropy)
hist_features_exp[,entropy:=tmp_entropy]

# gc
rm(tmp_matrix)
rm(tmp_std)
gc()


# save
save(list=c("hist_features","hist_features_exp"), 
     file="./data/dt_hist.RData")