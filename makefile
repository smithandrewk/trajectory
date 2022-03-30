LOGPATH = data
LOGFILE = $(LOGPATH)/$(shell date "+%y-%m-%d-%H:%M:%S")

run:
	nc -ul 65432 | tee $(LOGFILE).data | ./animate_sensors_live.py
get_data:
	nc -ul 65432 | tee $(LOGFILE).data
