import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Union, Any
import json
import requests
from datetime import datetime

class BaseNotifier:
    """通知器基类"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        初始化通知器
        
        Args:
            config: 通知配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def send(self, title: str, content: str, **kwargs) -> bool:
        """
        发送通知
        
        Args:
            title: 通知标题
            content: 通知内容
            **kwargs: 额外参数
            
        Returns:
            发送是否成功
        """
        raise NotImplementedError("Subclasses must implement send()")
    
    def _log_success(self, method: str, recipient: str) -> None:
        """记录成功发送日志"""
        self.logger.info(f"发送{method}通知到{recipient}成功")
    
    def _log_error(self, method: str, recipient: str, error: str) -> None:
        """记录发送失败日志"""
        self.logger.error(f"发送{method}通知到{recipient}失败: {error}")

class EmailNotifier(BaseNotifier):
    """邮件通知器"""
    
    def send(self, title: str, content: str, recipients: List[str] = None, 
             html_content: Optional[str] = None, attachments: List[str] = None) -> bool:
        """
        发送邮件通知
        
        Args:
            title: 邮件标题
            content: 邮件内容
            recipients: 收件人列表，默认为None则使用配置中的收件人
            html_content: HTML格式内容，默认为None
            attachments: 附件列表，默认为None
            
        Returns:
            发送是否成功
        """
        if not self.config.get('enable_email', False):
            self.logger.info("邮件通知已禁用")
            return False
        
        try:
            email_config = self.config.get('email', {})
            sender = email_config.get('sender')
            password = email_config.get('password')
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            
            # 如果未提供收件人列表，使用配置中的默认收件人
            if not recipients:
                recipients = email_config.get('recipients', [])
                if not recipients:
                    self.logger.error("未提供收件人")
                    return False
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = title
            
            # 添加文本内容
            if html_content:
                msg.attach(MIMEText(html_content, 'html'))
            else:
                msg.attach(MIMEText(content, 'plain'))
            
            # 添加附件
            if attachments:
                for attachment in attachments:
                    try:
                        with open(attachment, 'rb') as f:
                            part = MIMEText(f.read(), 'base64', 'utf-8')
                            part['Content-Type'] = 'application/octet-stream'
                            part['Content-Disposition'] = f'attachment; filename="{attachment.split("/")[-1]}"'
                            msg.attach(part)
                    except Exception as e:
                        self.logger.warning(f"添加附件 {attachment} 失败: {str(e)}")
            
            # 连接SMTP服务器并发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            
            self._log_success("邮件", ", ".join(recipients))
            return True
            
        except Exception as e:
            self._log_error("邮件", ", ".join(recipients) if recipients else "未知", str(e))
            return False

class SMSNotifier(BaseNotifier):
    """短信通知器"""
    
    def send(self, title: str, content: str, recipients: List[str] = None) -> bool:
        """
        发送短信通知
        
        Args:
            title: 通知标题
            content: 通知内容
            recipients: 手机号列表，默认为None则使用配置中的手机号
            
        Returns:
            发送是否成功
        """
        if not self.config.get('enable_sms', False):
            self.logger.info("短信通知已禁用")
            return False
        
        try:
            sms_config = self.config.get('sms', {})
            api_key = sms_config.get('api_key')
            api_url = sms_config.get('api_url')
            
            # 如果未提供接收号码，使用配置中的默认号码
            if not recipients:
                recipients = sms_config.get('recipients', [])
                if not recipients:
                    self.logger.error("未提供短信接收号码")
                    return False
            
            # 短信内容（通常会限制长度，这里简单地合并标题和内容）
            sms_content = f"{title}: {content}"
            if len(sms_content) > 70:  # 假设短信长度限制为70个字
                sms_content = sms_content[:67] + "..."
            
            # 这里是一个模拟的短信API调用
            # 实际使用中需要替换为真实的短信API
            """
            response = requests.post(
                api_url,
                json={
                    'apiKey': api_key,
                    'phoneNumbers': recipients,
                    'content': sms_content
                }
            )
            
            if response.status_code == 200:
                self._log_success("短信", ", ".join(recipients))
                return True
            else:
                self._log_error("短信", ", ".join(recipients), f"API错误: {response.text}")
                return False
            """
            
            # 模拟发送成功
            self._log_success("短信", ", ".join(recipients))
            self.logger.info(f"模拟短信内容: {sms_content}")
            return True
            
        except Exception as e:
            self._log_error("短信", ", ".join(recipients) if recipients else "未知", str(e))
            return False

class AppNotifier(BaseNotifier):
    """APP推送通知器"""
    
    def send(self, title: str, content: str, recipients: List[str] = None,
             data: Dict[str, Any] = None, channel: str = "stock_alert") -> bool:
        """
        发送APP推送通知
        
        Args:
            title: 通知标题
            content: 通知内容
            recipients: 推送目标ID列表，默认为None则推送给所有用户
            data: 附加数据，默认为None
            channel: 通知频道，默认为"stock_alert"
            
        Returns:
            发送是否成功
        """
        if not self.config.get('enable_app_push', False):
            self.logger.info("APP推送通知已禁用")
            return False
        
        try:
            app_config = self.config.get('app_push', {})
            api_key = app_config.get('api_key')
            api_url = app_config.get('api_url')
            
            # 如果未提供接收者ID，使用配置中的默认ID
            if not recipients:
                recipients = app_config.get('recipients', ['all'])
            
            # 构建推送数据
            push_data = {
                'title': title,
                'body': content,
                'channel': channel,
                'targets': recipients,
                'timestamp': datetime.now().isoformat(),
                'data': data or {}
            }
            
            # 这里是一个模拟的APP推送API调用
            # 实际使用中需要替换为真实的推送API
            """
            response = requests.post(
                api_url,
                headers={'Authorization': f'Bearer {api_key}'},
                json=push_data
            )
            
            if response.status_code == 200:
                self._log_success("APP推送", ", ".join(recipients))
                return True
            else:
                self._log_error("APP推送", ", ".join(recipients), f"API错误: {response.text}")
                return False
            """
            
            # 模拟发送成功
            self._log_success("APP推送", ", ".join(recipients))
            self.logger.info(f"模拟APP推送内容: {json.dumps(push_data, ensure_ascii=False)}")
            return True
            
        except Exception as e:
            self._log_error("APP推送", ", ".join(recipients) if recipients else "未知", str(e))
            return False

class NotificationManager:
    """通知管理器，统一管理不同类型的通知"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        初始化通知管理器
        
        Args:
            config: 通知配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化各种通知器
        self.email_notifier = EmailNotifier(config, logger)
        self.sms_notifier = SMSNotifier(config, logger)
        self.app_notifier = AppNotifier(config, logger)
    
    def send_notification(self, title: str, content: str, 
                          notification_types: List[str] = None, **kwargs) -> Dict[str, bool]:
        """
        发送通知
        
        Args:
            title: 通知标题
            content: 通知内容
            notification_types: 通知类型列表，可以是['email', 'sms', 'app']的组合，默认为所有启用的类型
            **kwargs: 其他参数
            
        Returns:
            各类型通知发送结果的字典
        """
        # 如果未指定通知类型，使用所有启用的类型
        if notification_types is None:
            notification_types = []
            if self.config.get('enable_email', False):
                notification_types.append('email')
            if self.config.get('enable_sms', False):
                notification_types.append('sms')
            if self.config.get('enable_app_push', False):
                notification_types.append('app')
        
        results = {}
        
        # 发送邮件通知
        if 'email' in notification_types:
            email_kwargs = {k: v for k, v in kwargs.items() if k in ['recipients', 'html_content', 'attachments']}
            results['email'] = self.email_notifier.send(title, content, **email_kwargs)
        
        # 发送短信通知
        if 'sms' in notification_types:
            sms_kwargs = {k: v for k, v in kwargs.items() if k in ['recipients']}
            results['sms'] = self.sms_notifier.send(title, content, **sms_kwargs)
        
        # 发送APP推送通知
        if 'app' in notification_types:
            app_kwargs = {k: v for k, v in kwargs.items() if k in ['recipients', 'data', 'channel']}
            results['app'] = self.app_notifier.send(title, content, **app_kwargs)
        
        return results
    
    def send_stock_alert(self, stock_code: str, alert_type: str, price: float, 
                         change_pct: float, message: str, **kwargs) -> Dict[str, bool]:
        """
        发送股票预警通知
        
        Args:
            stock_code: 股票代码
            alert_type: 预警类型，如"涨停预警"、"大幅拉升"、"止损预警"等
            price: 当前价格
            change_pct: 涨跌幅
            message: 预警信息
            **kwargs: 其他参数
            
        Returns:
            各类型通知发送结果的字典
        """
        title = f"{stock_code} {alert_type}"
        content = (
            f"股票: {stock_code}\n"
            f"价格: {price:.2f}\n"
            f"涨跌幅: {change_pct*100:.2f}%\n"
            f"预警信息: {message}"
        )
        
        # 构建HTML内容
        html_content = f"""
        <html>
        <body>
            <h2>{title}</h2>
            <p><strong>股票:</strong> {stock_code}</p>
            <p><strong>价格:</strong> {price:.2f}</p>
            <p><strong>涨跌幅:</strong> <span style="color:{'red' if change_pct > 0 else 'green'}">{change_pct*100:.2f}%</span></p>
            <p><strong>预警信息:</strong> {message}</p>
        </body>
        </html>
        """
        
        # 构建推送数据
        push_data = {
            'stock_code': stock_code,
            'price': price,
            'change_pct': change_pct,
            'alert_type': alert_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.send_notification(
            title, content,
            html_content=html_content,
            data=push_data,
            channel="stock_alert",
            **kwargs
        ) 