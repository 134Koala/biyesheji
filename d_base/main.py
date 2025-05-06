import argparse
from data_processor import DataProcessor
from backtest import BacktestEngine
from gru_model import GRUModel

def main():
    # 参数解析保持不变
    parser = argparse.ArgumentParser(description='股票预测系统')
    parser.add_argument('--symbol', type=str, default='000001', help='股票代码(默认: 000001)')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='开始日期')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='结束日期')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    args = parser.parse_args()

    # 1. 数据预处理（现在接收三个数据集）
    processor = DataProcessor(args.symbol, args.start_date, args.end_date)
    train_data, val_data, test_data = processor.prepare_data()  # 新增验证集接收
    
    # 2. 准备训练/验证/测试数据集（新增验证集处理）
    X_train, y_train = processor.create_dataset(train_data)
    X_val, y_val = processor.create_dataset(val_data)  # 新增验证集处理
    X_test, y_test = processor.create_dataset(test_data)
    
    # 调整数据维度（新增验证集处理）
    X_train = X_train.reshape(X_train.shape[0], processor.look_back, 1)
    X_val = X_val.reshape(X_val.shape[0], processor.look_back, 1)  # 新增验证集维度调整
    X_test = X_test.reshape(X_test.shape[0], processor.look_back, 1)
    
    # 3. 训练GRU模型（新增验证集参数传递）
    gru_model = GRUModel(look_back=processor.look_back)
    best_params = gru_model.hyperparameter_tuning(X_train, y_train, X_val, y_val)  # 新增验证集参数
    print(f"最优参数: {best_params}")
    gru_model.train(X_train, y_train, X_val, y_val)  # 确保验证集传入
    
    # 4. 执行回测
    engine = BacktestEngine(initial_capital=args.capital)
    engine.run_backtest(test_data)  # 必须传入测试数据
    
    # 5. 显示结果
    engine.display_results()
    
if __name__ == "__main__":
    main()