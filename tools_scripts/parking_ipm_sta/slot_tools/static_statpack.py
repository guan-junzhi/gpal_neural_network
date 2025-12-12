import sys
import os
import time
import numpy as np

class PrintLogger:
    """同时将输出打印到控制台和写入文件"""
    
    def __init__(self, log_file="output.log", mode="w", timestamp=True):
        """
        初始化打印记录器
        
        参数:
            log_file: 日志文件路径
            mode: 文件打开模式 ('w' 覆盖, 'a' 追加)
            timestamp: 是否在每条日志前添加时间戳
        """
        self.terminal = sys.stdout
        self.log_file = log_file
        self.timestamp = timestamp
        
        # 创建日志文件所在目录
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        
        # 打开日志文件
        self.log = open(log_file, mode, encoding="utf-8")
        
        # 写入日志头
        if mode == "w":
            self.write_header()
    
    def write_header(self):
        """写入日志文件头"""
         
        header = "Times Date : " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S")
        self.log.write(header)
    
    def write(self, message):
        """处理写入操作"""
        # 写入控制台
        self.terminal.write(message)
        
        # 写入文件
        self.log.write(message)
        self.log.flush()  # 确保数据立即写入文件
    
    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """关闭日志文件"""
        footer = f"\n{'='*20} Log Ended at {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}\n"
        self.log.write(footer)
        self.log.close()
        # 恢复原始标准输出
        sys.stdout = self.terminal
        print(f"日志已保存到: {os.path.abspath(self.log_file)}")

def log_to_file(log_file="output.log", mode="w", timestamp=True):
    """
    装饰器：将函数的所有打印输出捕获到文件
    
    使用示例：
    @log_to_file("my_script.log")
    def main():
        print("这将同时输出到控制台和文件")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            try:
                # 重定向标准输出
                sys.stdout = PrintLogger(log_file, mode, timestamp)
                return func(*args, **kwargs)
            finally:
                # 恢复标准输出
                if isinstance(sys.stdout, PrintLogger):
                    sys.stdout.close()
                sys.stdout = original_stdout
        return wrapper
    return decorator

def printTitleInfo():
    print ("")
    print ("--------------Slot Det Model Auto Test On Image DataSet---------------")
    print ("Algorithom vrsion : ", 'torch v0')
    print ("NetWork Description : ", " (ddrnet_23_slim)")
    #print "" 

def initStatPack():
    StatPack =  {}
    StatPack['point_pixel_error_sum'] = 0.0
    StatPack['angle_error'] = 0.0
    StatPack['point_total_num'] = 0
    StatPack['point_true_num'] = 0
    StatPack['point_miss_num'] = 0
    StatPack['point_false_num'] = 0
    StatPack['line_total_num'] = 0
    StatPack['line_true_num'] = 0
    StatPack['line_miss_num'] = 0
    StatPack['line_false_num'] = 0
    StatPack['point_det_num'] = 0
    StatPack['line_det_num'] = 0
    return StatPack

def updatePack(StatPack, resultPack):
    print("StatPack['point_pixel_error_sum'] is: {}; resultPack['point_error'] is: {}".format(StatPack['point_pixel_error_sum'], resultPack['point_error']))
    print("StatPack['angle_error'] is: {}; resultPack['angle_error'] is: {}".format(StatPack['angle_error'], resultPack['angle_error']))
    print("StatPack['point_total_num'] is: {}; resultPack['point_total_num'] is: {}".format(StatPack['point_total_num'], resultPack['point_total_num']))
    print("StatPack['point_true_num'] is: {}; resultPack['point_true_num'] is: {}".format(StatPack['point_true_num'], resultPack['point_true_num']))
    print("StatPack['point_miss_num'] is: {}; resultPack['point_miss_num'] is: {}".format(StatPack['point_miss_num'], resultPack['point_miss_num']))
    print("StatPack['point_false_num'] is: {}; resultPack['point_false_num'] is: {}".format(StatPack['point_false_num'], resultPack['point_false_num']))
    print("StatPack['line_total_num'] is: {}; resultPack['line_total_num'] is: {}".format(StatPack['line_total_num'], resultPack['line_total_num']))
    print("StatPack['line_true_num'] is: {}; resultPack['line_true_num'] is: {}".format(StatPack['line_true_num'], resultPack['line_true_num']))
    print("StatPack['line_miss_num'] is: {}; resultPack['line_miss_num'] is: {}".format(StatPack['line_miss_num'], resultPack['line_miss_num']))
    print("StatPack['line_false_num'] is: {}; resultPack['line_false_num'] is: {}".format(StatPack['line_false_num'], resultPack['line_false_num']))
    
    print("StatPack['point_det_num'] is: {}; resultPack['point_det_num'] is: {}".format(StatPack['point_det_num'],resultPack['point_det_num'] ))
    print("StatPack['line_de_num'] is: {}; resultPack['line_det_num'] is: {}".format(StatPack['line_det_num'], resultPack['line_det_num']))
    
    StatPack['point_pixel_error_sum'] = StatPack['point_pixel_error_sum'] + resultPack['point_error']
    StatPack['angle_error'] = StatPack['angle_error'] + resultPack['angle_error']
    StatPack['point_total_num'] = StatPack['point_total_num'] + resultPack['point_total_num']
    StatPack['point_true_num'] = StatPack['point_true_num'] + resultPack['point_true_num']
    StatPack['point_miss_num'] = StatPack['point_miss_num'] + resultPack['point_miss_num']
    StatPack['point_false_num'] = StatPack['point_false_num'] + resultPack['point_false_num']
    StatPack['line_total_num'] = StatPack['line_total_num'] + resultPack['line_total_num']
    StatPack['line_true_num'] = StatPack['line_true_num'] + resultPack['line_true_num']
    StatPack['line_miss_num'] = StatPack['line_miss_num'] + resultPack['line_miss_num']
    StatPack['line_false_num'] = StatPack['line_false_num'] + resultPack['line_false_num']

    StatPack['point_det_num'] = StatPack['point_det_num'] + resultPack['point_det_num']
    StatPack['line_det_num'] = StatPack['line_det_num'] + resultPack['line_det_num']

def outputStat(StatPack, save_path):
    if StatPack['point_total_num'] != 0:
        StatPack['point_pixel_error_sum'] = StatPack['point_pixel_error_sum'] / StatPack['point_total_num']
    if StatPack['line_total_num'] != 0:
        StatPack['angle_error'] = StatPack['angle_error'] / StatPack['line_total_num']
    if StatPack['point_total_num'] != 0:
        StatPack['point_recall'] = StatPack['point_true_num']*1.0/ StatPack['point_total_num']
        StatPack['point_miss_rate'] = StatPack['point_miss_num'] * 1.0 / StatPack['point_total_num']
    if StatPack['point_det_num'] != 0:
        StatPack['point_precision'] = StatPack['point_true_num'] * 1.0 / StatPack['point_det_num']
        StatPack['point_FDR'] = StatPack['point_false_num'] * 1.0 / StatPack['point_det_num']
    
    if StatPack['line_total_num'] != 0:
        StatPack['line_recall'] = StatPack['line_true_num']*1.0/ StatPack['line_total_num']
        StatPack['line_miss_rate'] = StatPack['line_miss_num'] * 1.0 / StatPack['line_total_num']
    if StatPack['line_det_num'] != 0:
        StatPack['line_precision'] = StatPack['line_true_num'] * 1.0 / StatPack['line_det_num']
        StatPack['line_FDR'] = StatPack['line_false_num'] * 1.0 / StatPack['line_det_num']

    printTitleInfo()
    print ("--->point count result : ")
    
    print("point recall = ", StatPack['point_recall'])
    print("point_precision = ", StatPack['point_precision'])
    print ("point average pixel error : ", StatPack['point_pixel_error_sum'])
    print("point FDR: ", StatPack['point_FDR'])
    print("point miss rate = ", StatPack['point_miss_rate'])

    print ("total ann point numbers = ", StatPack['point_total_num'])
    print("point_det_num = ", StatPack['point_det_num']) 
    print("point_true_num = ", StatPack['point_true_num']) 
    print("point_false_num = ", StatPack['point_false_num']) 
    print("point miss num = ", StatPack['point_miss_num'])
    

    print ("--->line count result : ")
    print ("line recall = ", StatPack['line_recall'])
    print("line_precision = ", StatPack['line_precision'])
    print ("point line angle error degree : ", np.degrees(StatPack['angle_error']))
    print("line FDR = ", StatPack['line_FDR']) 
    print("line miss rate = ", StatPack['line_miss_rate'])

    print ("total ann line numbers = ", StatPack['line_total_num'])
    print("line_det_num = ", StatPack['line_det_num'])
    print("line_true_num = ", StatPack['line_true_num'])
    print("line_false_num = ", StatPack['line_false_num']) 
    print("line miss num = ", StatPack['line_miss_num'])
    
    print ("--------------------down------------------------")
    log_to_file(log_file=save_path)
    return 

if __name__== "__main__":
    staticpack = initStatPack()
    resultPack = {}
    updatePack(staticpack, resultPack)
    outputStat(staticpack)

