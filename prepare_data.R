
require(reshape2)
require(dplyr)

get_nearest_index <- function(x, mytime)
{
  abssub <- abs(x - mytime)
  ordered_index <- order(abssub)
  return(ordered_index[1])
}

filenames = c("./data/data_for_zmh_0045.dat",
               "./data/data_for_zmh_006.dat",
               "./data/data_for_zmh_008.dat",
               "./data/data_for_zmh_009.dat",
               "./data/data_for_zmh_010.dat",
               "./data/data_for_zmh_013.dat")
stress_list = c(0.045,0.06,0.08,0.09,0.10,0.13)
df_data_origin <- data.frame()
for (i in 1:length(stress_list))
{
  tmp <- dcast(melt( read.table(filenames[i]) , c("V1","V2") ), V1 ~ variable+V2)
  tmp$Stress <- stress_list[i]
  df_data_origin <- rbind(df_data_origin, tmp)
  rm(tmp)
  gc()
}

save(df_data_origin, file="./data/df_data_origin.RData")

# prepare detail data with stress = 0.045
filenames = c("./data/data_for_zmh0045-exp-9904.dat",
              "./data/data_for_zmh0045-exp-5457.dat")
stress_list = c(0.045,0.045)
df_data_origin_detail <- data.frame()
for (i in 1:length(stress_list))
{
  tmp <- dcast(melt( read.table(filenames[i]) , c("V1","V2") ), V1 ~ variable+V2)
  tmp$Stress <- stress_list[i]
  df_data_origin_detail <- rbind(df_data_origin_detail, tmp)
  rm(tmp)
  gc()
}
save(df_data_origin_detail, file="./data/df_data_origin_detail.RData")

# exp time
data_exp_stresses <- c(0.045, 0.045, 0.06, 0.08, 0.09, 0.1, 0.13, 0.13) # 实验所用的应力
data_exp_times <- c(9904, 5457, 1494, 370, 207, 354,  94,  80) # 实验所得的周期数，找对应数据时有一点误差

df_exp <- data.frame(data_exp_stresses, data_exp_times)  # exp time 

get_nearest_index(x = df_data_origin_detail[,1], 9904)

df_data_exp <- data.frame()
for (i in 1:nrow(df_exp) )
{
  if (i <= 2)
  {
    time_index <- get_nearest_index(df_data_origin_detail[df_data_origin_detail$Stress==df_exp[i,1],1], 
                                    df_exp[i,2])
    df_data_exp <- rbind(df_data_exp, 
                         df_data_origin_detail[df_data_origin_detail$Stress==df_exp[i,1],][time_index,])
  }
  else
  {
    time_index <- get_nearest_index(df_data_origin[df_data_origin$Stress==df_exp[i,1],1], 
                                    df_exp[i,2])
    df_data_exp <- rbind(df_data_exp, 
                         df_data_origin[df_data_origin$Stress==df_exp[i,1],][time_index,])
  }
  gc()
}

save(df_data_exp, file="./data/df_data_exp.RData")
