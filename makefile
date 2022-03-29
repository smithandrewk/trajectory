run:
	nc -l 65432 | ./animate_sensors_live.py
install:
	cowsay installing
	pip3 install -r requirements.txt
	sudo apt install graphviz
	cowsay good to go!
pip3:
	sudo apt update
	sudo apt upgrade
	sudo apt autoremove
	sudo apt install python3
	sudo apt install python3-pip
	pip3 --version
