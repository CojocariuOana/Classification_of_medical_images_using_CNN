import os
import random
import shutil


def image_random(string):

    # listează toate fișierele din directorul specificat, verificând dacă corepunde extensia acestora
    image_files = [f for f in os.listdir(string) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # ordonează aleatoriu fișierele
    random.shuffle(image_files)

    # creează un nou fișier în care vor fi stocate imaginile aleatorizate
    new_folder_path = os.path.join(string, 'random_order')
    os.makedirs(new_folder_path, exist_ok=True)

    # setează un nume nou fiecărei imagini și le copiază în directorul creat anterior
    for i, filename in enumerate(image_files):
        _, extension = os.path.splitext(filename)
        new_filename = f'brain_{i + 1}{extension}'
        src_path = os.path.join(string, filename)
        dest_path = os.path.join(new_folder_path, new_filename)
        shutil.copy(src_path, dest_path)

    print("Imaginile au fost ordonate aleatoriu și salvate în directorul 'random_order'.")
