library(readxl)
total_value_added_VAUD <- read_excel("Data/total value added VAUD.xlsx", sheet = "VAUD")

total_value_added_VAUD <- as.data.frame( t(total_value_added_VAUD))
colnames(total_value_added_VAUD)<-  as.character(unlist(total_value_added_VAUD[1,]))
total_value_added_VAUD <- total_value_added_VAUD[-1,]
#names(total_value_added_VAUD)

regressors <- read.csv("/home/youssef/EPFL/Data/DataAgg.csv")

regressors<-regressors[complete.cases(regressors), ]
regressors_var<-cbind(regressors[,-1])
library(xts)

GDP<-total_value_added_VAUD['GDP']
GDP<- as.numeric(levels(GDP$GDP))[GDP$GDP]
GDP.ts<-ts(GDP,frequency=4,start=c(1997,1))

GDP_shortened.ts<- window(GDP.ts, c(2000, 01))

library(nnfor)

fit1 <- mlp(GDP_shortened.ts,xreg=regressors_var)
print(fit1)
plot(fit1)
plot(GDP_shortened.ts)
frc <- forecast(fit1,h=1,xreg=regressors_var)
print(frc)
plot(frc)

