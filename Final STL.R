library(readxl)

library(stlplus)
library(ggplot2)
library(TSA)
library(forecast)


data0 <- read_excel("D:/Ian/NCKU/碩一/統諮/Final/失業率資料_2 .xlsx")
ue = data0$`失業率(%)_合計`

train_ts <- ts(ue, start = c(2001, 1), frequency = 12)

train_s <- stlplus(train_ts, t = as.vector(stats::time(train_ts)), n.p=frequency(train_ts),
                   l.window=13, t.window=19, s.window="periodic", s.degree=1)


plot(train_s)


trend <- train_s$data$trend
seasonal <- train_s$data$seasonal
remainder <- train_s$data$remainder
trend
write.csv(trend, "D:/Ian/NCKU/碩一/統諮/Final/mean.csv", row.names = FALSE)


#remainder = remainder + seasonal

remainder = remainder[1:(276-53)]
remainder_test = remainder[(276-53+1):276]



acf(remainder,100)
pacf(remainder,100)
eacf(remainder)


#2220-6
fitdata.2010 = stats::arima(remainder, order=c(2,0,2),seasonal = list(order=c(2,0,0), period=6))
fitdata.2010
autoplot(fitdata.2010)
B_text_p_value = rep(NA,4)
for(i in 5:25){
  B_text_p_value[i] = Box.test(fitdata.2010$residuals, lag=i, fitdf=4 , type="Ljung-Box")$p.value
}

plot(1:25, B_text_p_value[1:25], type="p", 
     main=expression(paste(M[1],"'s p values for Ljung-Box statistic")), 
     xlab="lag", ylab="p value", ylim=c(0,1))
abline(h=0.05, lty=2, col=4)

shapiro.test(fitdata.2010$residuals)

par(mfrow = c(2,1))
qqnorm(fitdata.2010$residuals)
qqline(fitdata.2010$residuals)
par(mfrow = c(1,1))
checkresiduals(fitdata.2010$residuals)

#2210-6
fitdata.201 = stats::arima(remainder, order=c(2,0,2), seasonal = list(order=c(1,0,0), period=6))
fitdata.201
autoplot(fitdata.201)
B_text_p_value = rep(NA,3)
for(i in 3:25){
  B_text_p_value[i] = Box.test(fitdata.201$residuals, lag=i, fitdf=3 , type="Ljung-Box")$p.value
}
plot(1:25, B_text_p_value[1:25], type="p", 
     main=expression(paste(M[2],"'s p values for Ljung-Box statistic")), 
     xlab="lag", ylab="p value", ylim=c(0,1))
abline(h=0.05, lty=2, col=4)
par(mfrow = c(2,1))
qqnorm(fitdata.201$residuals)
qqline(fitdata.201$residuals)
par(mfrow = c(1,1))
checkresiduals(fitdata.201$residuals)

auto.arima(remainder)



#add true data line  --  plot arima 後幾筆 + trend 後幾筆 + seasonal true 後幾筆 

# 1-step prediction

# 讀取預測的 trend 資料
tdm <- read.csv("D:/Ian/NCKU/碩一/統諮/Final/unemployment_prediction_results.csv")
trend_pred <- tdm[212:264, ncol(tdm)]  # 預測的 trend (後53筆)

# 準備資料
# 原始資料：276筆，取前223筆訓練，後53筆測試
# trend：264筆，取前211筆訓練，後53筆測試 (trend_pred)
# remainder：223筆訓練資料

# 重建完整的訓練資料 (使用原始的 trend 和 seasonal)
train_data <- trend[1:211] + seasonal[1:211] + remainder[1:211]

# 設定 SARIMA 模型參數
p <- 2
d <- 0
q <- 2
P <- 2
D <- 0
Q <- 0
period <- 6

# 從第150期開始畫圖 (索引150-276，共127個點)
plot_start <- 150
plot_range <- plot_start:276

# 初始化預測結果
pred <- numeric(53)
U <- numeric(53)
L <- numeric(53)

# 進行一步預測 (從第223期開始預測)
for(i in 1:53) {
  current_index <- 223 + i  # 212:264
  
  # 使用到目前為止的 remainder 資料建立 SARIMA 模型
  remainder_current <- remainder[1:(223 + i - 1)]
  
  # 建立 SARIMA 模型
  model <- stats::arima(remainder_current, 
                        order = c(p, d, q), 
                        seasonal = list(order = c(P, D, Q), period = period))
  
  # 預測下一期的 remainder
  x.pred <- predict(model, n.ahead = 1)
  
  # 加上對應的 trend 和 seasonal 重建完整預測
  # 注意：seasonal 是週期性的，需要取對應的季節性成分
  seasonal_index <- ((current_index - 1) %% 12) + 1
  seasonal_component <- seasonal[seasonal_index]
  
  prediction <- x.pred$pred + trend_pred[i] + seasonal_component
  se <- x.pred$se
  
  pred[i] <- prediction
  U[i] <- prediction + 1.96 * se
  L[i] <- prediction - 1.96 * se
}

# 繪製預測結果
# 從第150期開始顯示
ts.plot(c(last_column[plot_start:223], rep(NA, 53)), 
        ylim = c(2.5, 5), 
        ylab = 'unemployment rate',
        main = 'SARIMA(2,0,2)×(2,0,0)₆ One-step Ahead Prediction')

# 標記實際測試資料
points((224:276) - plot_start + 1, last_column[224:276], pch = 16)

# 繪製預測線
pred_x <- (224:276) - plot_start + 1
lines(pred_x, pred, col = 2, lwd = 2)
lines(pred_x, U, col = 4, lty = 2)
lines(pred_x, L, col = 4, lty = 2)

# 添加圖例
legend("topright", 
       legend = c('Prediction', "95% Confidence Interval", 'Actual Data'),
       col = c('red', 4, "black"), 
       lty = c(1, 2, NA), 
       pch = c(NA, NA, 16),
       cex = 0.8)

# 計算預測精度指標
actual_test <- last_column[212:264]
mae <- mean(abs(pred - actual_test))
rmse <- sqrt(mean((pred - actual_test)^2))
mape <- mean(abs((pred - actual_test) / actual_test)) * 100

cat("預測精度指標：\n")
cat("MAE:", round(mae, 4), "\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAPE:", round(mape, 2), "%\n")

# 顯示最後幾期的預測結果
cat("\n最後5期預測結果：\n")
comparison <- data.frame(
  Period = 260:264,
  Actual = actual_test[49:53],
  Predicted = pred[49:53],
  Error = actual_test[49:53] - pred[49:53]
)
print(comparison)

