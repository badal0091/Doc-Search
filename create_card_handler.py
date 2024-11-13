import os
import json
import yaml
from gramex.handlers import BaseHandler

class CreateCardHandler(BaseHandler):
    def post(self):
        data = json.loads(self.request.body)
        card_name = data['cardName']
        client = data['client']
        title = data['title']
        body = data['body']
        link = data['link']
        pdf_link = data['pdfLink']
        questions = data['questions']

        # Call the function to add the card to config.yaml
        add_card_to_config(card_name, client, title, body, link, pdf_link, questions)

        self.write(json.dumps({'success': True}))

def add_card_to_config(card_name, client, title, body, link, pdf_link, questions):
    # Load the existing config.yaml
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Create a new section for the card
    new_card = {
        'client': client,
        'title': title,
        'body': body,
        'link': link,
        'pdfLink': pdf_link,
        'questions': questions
    }

    # Add the new card to the config under the card name
    config[card_name] = new_card

    # Save the updated config back to config.yaml
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # Create a new folder with the name matching the Card name
    new_folder_path = os.path.join('docsearch', card_name)
    os.makedirs(new_folder_path, exist_ok=True) 