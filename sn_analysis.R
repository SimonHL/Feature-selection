require(ggplot2)
require(sn)
require(data.table)

### load original data ----
load("./data/dt_data.RData") #dt_data

get_hist <- function(x, breaks)
{
  a <- hist(x, breaks=breaks, plot=FALSE)
  return(a$density)# / length(x)) 
}

# ----------
# get hist as features 
p_dim <- 2
dim_range <- ((p_dim-1)*27000 + 2):(p_dim*27000 + 1) # columns of the specified data

# use "set" for speed
dp <- as.matrix(dt_data_origin[, dim_range, with=FALSE ]) # choose matrix from simulation data
breaks <- seq(min(dp), max(dp), length.out = 101)
hist_features <- as.data.table( matrix(0, nrow(dp), 100  ))
for (i in 1:nrow(hist_features) )
{
  set(hist_features, i, names(hist_features), as.list(get_hist(dp[i,], breaks)) )
}

dp <- as.matrix(dt_data_exp[,dim_range, with=FALSE]) # choose matrix from exp data
hist_features_exp <- as.data.table( matrix(0, nrow(dp), 100  ))
for (i in 1:nrow(hist_features_exp) )
{
  set(hist_features_exp, i, names(hist_features_exp), as.list(get_hist(dp[i,], breaks)) )
}

# principal components analysis
pca <- prcomp(hist_features, scale. = T)
pca_eigenvalues <- pca$sdev**2
pca_explain_ratio <- sum(pca_eigenvalues[1:4] )/sum(pca_eigenvalues)
pca_keep_number <- 2L
dfl <- predict(pca, hist_features)[,1:pca_keep_number]
dfp <- predict(pca,hist_features_exp)[,1:pca_keep_number]
# dfl <- t(t(hist_features)-pca$center) %*% pca$rotation[,1:2]
# dfp <-t(t(hist_features_exp)-pca$center)  %*% pca$rotation[,1:2]

dfl <- as.data.frame(dfl)
dfp <- as.data.frame(dfp)
names(dfl)[1:2] <- c("x", "y")
names(dfp)[1:2] <- c("x", "y")

fit <- msn.mle(y=dfp, opt.method = "BFGS")

# claculate the density of parameters, only for plotting contour
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
dfl$phi <- dmsn(x=dfl, dp=fit$dp)
dfl$Exp <- dt_data_origin[,Stress]
dfl$Time <- dt_data_origin[,V1]



plot_p <- ggplot(data=dfp, aes(x,y))
plot_p_c <- plot_p + geom_point(colour="red", size=4)+ geom_contour(data=xy_grid, aes(x=x,y=y,z=z)) 

ar <- arrow(angle = 30, length = unit(0.05, "inches"), ends = "last", type = "open")
plot_p_c_l <-plot_p_c + geom_path(data=dfl, aes(x,y,colour=factor(Exp) ), arrow =ar, size=1 )

ggplot(data=dfl, aes(Time,phi))  + geom_path(data=dfl, aes(colour=factor(Exp)), size=1) 

