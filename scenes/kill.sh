ps aux | grep "\-port 1087" | grep -v grep | awk '{print $2}' | xargs kill -9
