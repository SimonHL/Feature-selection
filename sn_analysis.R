require(ggplot2)
require(sn)

dim <- 0
file_point <- paste0("df_point_dim", as.character(dim), ".dat")
file_line <- paste0("df_dim", as.character(dim), ".dat")
dfp <- read.csv(file_point, header = TRUE)
dfl <- read.csv(file_line, header = TRUE)

plot_p <- ggplot(data=dfp, aes(x,y))

sn_data <- cbind(dfp$x, dfp$y)

fit <- msn.mle(y=sn_data, opt.method = "BFGS")

dxy <-as.data.frame(rmsn(50000, dp=fit$dp))
pp <- plot_p + geom_point(colour="red", size=4)+ geom_density2d(data=dxy, aes(V1,V2)) 

ar <- arrow(angle = 30, length = unit(0.1, "inches"), ends = "last", type = "open")
pp + geom_path(data=dfl, aes(x,y,colour=factor(exp) ), arrow =ar, size=1 )


