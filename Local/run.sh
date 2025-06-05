read -p "API key de Wasender: " WAPIKEY
read -p "API key de Gemini: " GAPIKEY
read -p "URL estatica de ngrok: " NGROKURL
read -p "Authtoken ngrok: " AUTHTOKEN
read -p "Puerto para servidor local: " port
rm -f .env
printf "GEMINI_API_KEY=\"$GAPIKEY\"\nWASENDER_API_TOKEN=\"$WAPIKEY\"\n" >> .env
snap install ngrok
ngrok config add-authtoken $AUTHTOKEN
apt install python3-full
apt install python3.12-venv
apt install python3-pip
python3 -m venv venv
venv/bin/pip install -r requirements.txt
gunicorn --workers 4 --bind 0.0.0.0:$port app:app
/bin/sh -ec 'ngrok http --url=$NGROKURL $port'
