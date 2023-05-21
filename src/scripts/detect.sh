#next_command="scripts/exp.sh egnn actstrack opt"
while [[ true ]]; do
 if [ `ps -ef|grep "python pipeline.py" | grep -v grep |wc -l`  -gt 0 ];then
  sleep 30m
 else
#   bash $next_command
   python send_email.py "Some error occurred in the running programs."
#  mail -s "QQ停止运行！" 123@qq.com <<< 'QQ停止运行了'
   break; # 退出监控
 fi
done