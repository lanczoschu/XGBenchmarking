next_command="scripts/exp.sh egnn actstrack opt"
while [[ true ]]; do
 if [ `ps -ef|grep "python pipeline.py" | grep -v grep |wc -l`  -gt 1 ];then
  sleep 5m # 睡眠5分钟:每5分钟检查一次QQ是否停止运行
 else
   bash $next_command
   python send_email.py "running the command bash "$next_command
#  mail -s "QQ停止运行！" 123@qq.com <<< 'QQ停止运行了'
   break; # 退出监控
 fi
done