require(ggplot2)
require(sn)

# load original data
load("./data/df_data_origin.RData") #df_data_origin
load("./data/df_data_origin_detail.RData")
load("./data/df_data_exp.RData")
get_hist <- function(x, breaks)
{
  a <- hist(x, breaks=breaks, plot=FALSE)
  r <- a$density
  rm(a)
  gc()
  r
}
# get hist as features 
p_dim <- 2
dp <- as.matrix(df_data_origin[,((p_dim-1)*27000 + 2):(p_dim*27000 + 1)]) # choose matrix 
breaks <- seq(min(dp), max(dp), length.out = 101)
hist_features <- apply(dp,1, get_hist, breaks)
hist_features <- t(hist_features)

dp <- as.matrix(df_data_exp[,((p_dim-1)*27000 + 2):(p_dim*27000 + 1)]) # choose matrix for exp data 
hist_features_exp <- t(apply(dp, 1, get_hist, breaks))

# principal components analysis
pca <- prcomp(hist_features)
pca_eigenvalues <- pca$sdev**2
pca_explain_ratio <- sum(pca_eigenvalues[1:2] )/sum(pca_eigenvalues)
dfl <-hist_features %*% pca$rotation[,1:2]
dfp <- hist_features_exp %*% pca$rotation[,1:2]

dfl <- as.data.frame(dfl)
dfp <- as.data.frame(dfp)
names(dfl) <- c("x", "y")
names(dfp) <- c("x", "y")
# # read data from csv
# dim <- 1   # 
# file_point <- paste0("df_point_dim", as.character(dim), ".dat")
# file_line <- paste0("df_dim", as.character(dim), ".dat")
# dfp <- read.csv(file_point, header = TRUE)
# dfl <- read.csv(file_line, header = TRUE)

fit <- msn.mle(y=dfp[c("x","y")], opt.method = "BFGS")

# claculate the density of parameters
dim_x <- 100
dim_y <- 100
x_range <- range(dfl$x)
y_range <- range(dfl$y)
x_grid <- seq(x_range[1], x_range[2], length.out = dim_x)
y_grid <- seq(y_range[1], y_range[2], length.out = dim_y)
xy_grid <- expand.grid(x_grid,y_grid)
names(xy_grid) = c("x","y")
xy_grid$z <- dmsn(x=xy_grid, dp=fit$dp)

#the probability over time
dfl$exp <- df_data_origin$Stress
dfl$time <- df_data_origin[,1]
dfl$phi <- dmsn(x=dfl[c("x","y")], dp=fit$dp)

plot_p <- ggplot(data=dfp, aes(x,y))
plot_p_c <- plot_p + geom_point(colour="red", size=4)+ geom_contour(data=xy_grid, aes(x=x,y=y,z=z)) 

ar <- arrow(angle = 30, length = unit(0.05, "inches"), ends = "last", type = "open")
plot_p_c_l <-plot_p_c + geom_path(data=dfl, aes(x,y,colour=factor(exp) ), arrow =ar, size=1 )

ggplot(data=dfl, aes(time,phi)) + geom_path(aes(colour=factor(exp)), size=1) 


