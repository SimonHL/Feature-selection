
require(data.table)

get_nearest_index <- function(x, mytime)
{
  abssub <- abs(x - mytime)
  ordered_index <- order(abssub)
  return(ordered_index[1])
}

build_data_table<-function(filenames, stress_list)
{
  data_list = list()
  for (i in 1:length(stress_list))
  {
    data_list[[i]] <- dcast(melt( fread(filenames[i]) , c("V1","V2") ), V1 ~ variable+V2)
    data_list[[i]][, Stress:=stress_list[i]]
  }
  return(rbindlist(data_list))
}

filenames = c("./data/data_for_zmh_0045.dat",
               "./data/data_for_zmh_006.dat",
               "./data/data_for_zmh_008.dat",
               "./data/data_for_zmh_009.dat",
               "./data/data_for_zmh_010.dat",
               "./data/data_for_zmh_013.dat")
stress_list = c(0.045,0.06,0.08,0.09,0.10,0.13)
dt_data_origin <- build_data_table(filenames, stress_list)
gc()

# prepare detail data with stress = 0.045
filenames = c("./data/data_for_zmh0045-exp-9904.dat",
              "./data/data_for_zmh0045-exp-5457.dat")
stress_list = c(0.045,0.045)
dt_data_origin_detail <- build_data_table(filenames, stress_list)
gc()

# exp time
data_exp_stresses <- c(0.045, 0.045, 0.06, 0.08, 0.09, 0.1, 0.13, 0.13) # 实验所用的应力
data_exp_times <- c(9904, 5457, 1494, 370, 207, 354,  94,  80) # 实验所得的周期数，找对应数据时有一点误差

dt_exp <- data.table(data_exp_stresses, data_exp_times)  # exp time
names(dt_exp) <- c("Stress", "Time")

dt_data_exp <- copy(dt_data_origin_detail[1:nrow(dt_exp), ])
for (i in 1:nrow(dt_exp) )
{
  if (i <= 2)
  {
    dt_stress <- dt_data_origin_detail[Stress == dt_exp[i, Stress],] # 选择相同的Stress
    time_index <- get_nearest_index(dt_stress[,V1], dt_exp[i,Time])  # 选择最近的位置
    set(dt_data_exp, i, 1:ncol(dt_data_exp), dt_stress[time_index,])
    # dt_data_exp <- rbind(dt_data_exp, dt_stress[time_index,])
  }
  else
  {
    dt_stress <- dt_data_origin[Stress == dt_exp[i, Stress],]
    time_index <- get_nearest_index(dt_stress[,V1], dt_exp[i,Time])
    set(dt_data_exp, i, 1:ncol(dt_data_exp), dt_stress[time_index,])
  }
  rm(dt_stress)
  gc()
}

save(list=c("dt_data_origin","dt_data_origin_detail","dt_data_exp"), 
     file="./data/dt_data.RData")
