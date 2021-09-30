from transformers import pipeline

model = pipeline('question-answering', model='./model/', tokenizer='./model/')

model.save_pretrained('./model/')
main_context = """H.S.Fashion est une entreprise de marque de chaussures, située à Chlef Centre, en Algérie, nous vendons des chaussures pour hommes, femmes et enfants. Nous faisons à la fois de la vente au détail et de la vente en gros. Notre point de vente au détail est basé à Boulevard des Martyrs, Chlef, Algérie. Pour la vente en gros, nous sommes basés dans la Zone d'activités Ouled Mohamed, Chlef, Algérie. Nos produits sont principalement importés de Chine et d'Espagne"""
context_time = """Nos heures d'ouverture sont de 8h à 21h, tous les jours sauf le vendredi qui est de 8h à 12h."""
context_location = """H.S.Fashion est situé à Chlef Centre - Algérie. Pour la vente au détail, nous sommes situés 'Boulevard des Martyrs - Chlef -Algérie', et pour la vente en gros nous sommes situés dans la Zone d'activités Ouled Mohamed - Chlef - Algérie."""
context_contact = """Vous pouvez nous contacter en appelant nos numéros de téléphone 0553 79 27 48 ou 0555 67 60 47, ou nous envoyer un email sur notre email : hsfashion58@gmail.com. Vous pouvez également nous retrouver sur les réseaux sociaux via notre page Facebook : facebook.com/hsfashion02 et compte Instagram : @hsfashion02"""

user_utter = input().strip()

while user_utter != "quit":
    print(model({'question': user_utter,'context': main_context}))
    print(model({'question': user_utter,'context': context_time}))
    print(model({'question': user_utter,'context': context_location}))
    print(model({'question': user_utter,'context': context_contact}))
    user_utter = input().strip()