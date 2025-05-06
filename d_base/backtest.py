import pandas as pd
import numpy as np
from prediction import StockPredictor

class BacktestEngine:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.predictor = StockPredictor()
        
    def run_backtest(self, test_data):

        # 初始化预测器
        self.predictor = StockPredictor()  # 现在模型文件已存在

        # 获取预测结果
        test_data = test_data.copy()
        test_data['pred'] = self.predictor.rolling_predict(test_data)
        
        # 生成交易信号
        test_data['signal'] = np.where(test_data['pred'].shift(1) > test_data['close'].shift(1), 1, -1)
        
        # 计算收益
        test_data['returns'] = test_data['close'].pct_change()
        test_data['strategy_returns'] = test_data['signal'].shift(1) * test_data['returns']
        
        # 计算累计收益
        test_data['cumulative_strategy'] = (test_data['strategy_returns'] + 1).cumprod()
        test_data['cumulative_buy_hold'] = (test_data['returns'] + 1).cumprod()
        
        self.results = test_data
        return self.results
    
    def display_results(self):
        # 输出关键指标
        cumulative_return = self.results['cumulative_strategy'].iloc[-1] - 1
        annualized_return = (1 + cumulative_return)**(252/len(self.results)) - 1
        max_drawdown = (self.results['cumulative_strategy'].cummax() - self.results['cumulative_strategy']).max()
        # 新增波动性指标
        volatility = self.results['strategy_returns'].std() * np.sqrt(252)
        # 新增风险调整收益
        sharpe_ratio = (annualized_return - 0.03) / volatility

        print(f"策略累计收益: {cumulative_return:.2%}")
        print(f"年化收益: {annualized_return:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"策略波动率: {volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")