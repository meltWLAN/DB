import os
import platform
import logging
from pathlib import Path

class NotificationManager:
    """通知管理器，用于发送系统通知"""
    
    def __init__(self):
        """初始化通知管理器"""
        # 设置日志记录器
        self.logger = logging.getLogger("NotificationManager")
        self.logger.setLevel(logging.INFO)
        
        # 创建日志处理器
        handler = logging.FileHandler("logs/notification.log")
        handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(handler)
        
        # 确保图标目录存在
        os.makedirs("resources/icons", exist_ok=True)
        
        self.logger.info("通知管理器初始化完成")

    def send_notification(self, title="股票分析通知", message="", notification_type="info"):
        """
        发送系统通知
        
        Args:
            title (str): 通知标题，默认为"股票分析通知"
            message (str): 通知内容
            notification_type (str): 通知类型，可选值：info, warning, error，默认为info
            
        Returns:
            bool: 通知是否发送成功
        """
        self.logger.info(f"准备发送{notification_type}类型通知: {title}")
        
        # 验证通知类型
        valid_types = ["info", "warning", "error"]
        if notification_type not in valid_types:
            self.logger.warning(f"无效的通知类型: {notification_type}，使用默认类型: info")
            notification_type = "info"
            
        # 获取图标路径
        icon_path = f"resources/icons/{notification_type}.png"
        if not os.path.exists(icon_path):
            self.logger.warning(f"图标文件不存在: {icon_path}，使用默认图标")
            icon_path = "resources/icons/default.png"
            
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f"""
                    osascript -e 'display notification "{message}" with title "{title}"'
                """)
            elif system == "Linux":
                os.system(f'notify-send "{title}" "{message}" -i {icon_path}')
            elif system == "Windows":
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, message, icon_path=icon_path, duration=5)
            else:
                self.logger.error(f"不支持的操作系统: {system}")
                return False
                
            self.logger.info("通知发送成功")
            return True
            
        except Exception as e:
            self.logger.error(f"发送通知失败: {str(e)}")
            return False 