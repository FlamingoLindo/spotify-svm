import requests

def send_plot(webhook, file_path, plt_name):
    with open(file_path, 'rb') as file:
        payload = {
            'content': f'Aqui está o seu plot: {plt_name}'
        }
        files = {
            'file': file
        }
        response = requests.post(webhook, data=payload, files=files)
        if response.status_code == 200:
            print('Plot enviado com sucesso!')
        else: 
            print('Houve um erro ao enviar o plot para o discord!')
            print(f'{response.status_code}')

webhook_url = 'https://discord.com/api/webhooks/1296983820423135243/Dzshyqa28m8Y3hRjDwzHOlkJBBCc3C3V9BZwZuEi-_03eowyZLQ8bFlEdH0W1vuwh-Ot'
