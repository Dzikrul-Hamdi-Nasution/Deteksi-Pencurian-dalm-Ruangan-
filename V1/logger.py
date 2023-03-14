import telepot
import os

token = '5751445359:AAFzaIht-oboCxA8RAuJPct4kBtDqR7a9vM'
id_penerima = 1441844129

lokasi = 'History'
if not os.path.exists(lokasi):
	print('Lokasi Cache: ', lokasi)
	os.makedirs(lokasi)


bot = telepot.Bot(token)

bot.sendMessage(id_penerima,"Terjadi pencurian")

