import pandas as pd
from bs4 import BeautifulSoup

# Leer el HTML desde el archivo local
file_path = r"D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week12\Dumplings.html"

with open(file_path, 'r', encoding='utf-8') as file:
    html = file.read()

# Parsear el HTML
soup = BeautifulSoup(html, 'html.parser')

# Extraer el nombre de la receta
name_tag = soup.find('h1', class_='article-heading type--lion')
name = name_tag.get_text().strip() if name_tag else 'No name available'

# Extraer la descripción de la receta
description_tag = soup.find('p', class_='article-subheading type--dog')
description = description_tag.get_text().strip() if description_tag else 'No description available'

# Extraer los ingredientes
ingredients_tags = soup.find_all('li', class_='mm-recipes-structured-ingredients__list-item')
ingredients = []

for ingredient in ingredients_tags:
    quantity = ingredient.find('span', {'data-ingredient-quantity': 'true'}).get_text().strip()
    unit = ingredient.find('span', {'data-ingredient-unit': 'true'}).get_text().strip()
    name = ingredient.find('span', {'data-ingredient-name': 'true'}).get_text().strip()
    
    # Combinar los elementos en un solo string
    ingredient_text = f"{quantity} {unit} {name}".strip()
    ingredients.append(ingredient_text)

if not ingredients:
    ingredients.append('No ingredients available')

# Extraer los pasos
steps_tags = soup.find_all('li', class_='comp mntl-sc-block mntl-sc-block-startgroup mntl-sc-block-group--LI')
steps = [step.find('p').get_text().strip() for step in steps_tags if step.find('p')] if steps_tags else ['No steps available']

# Crear un DataFrame para almacenar la información en una sola fila
recipe_data = {
    'id': [0],
    'name': [name],
    'description': [description],
    'ingredients': [', '.join(ingredients)],
    'steps': [' '.join(steps)]
}

df = pd.DataFrame(recipe_data)

# Guardar el DataFrame en un archivo CSV
csv_file_path = r"D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week12\recipe_data.csv"
df.to_csv(csv_file_path, index=False, encoding='utf-8')

# Imprimir el DataFrame en consola
print(df)
