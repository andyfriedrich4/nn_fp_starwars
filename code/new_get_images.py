from icrawler.builtin import BingImageCrawler
import os

heroes = [
    "Qui-Gon Jinn", "Obi-Wan Kenobi", "Anakin Skywalker", "Padme Amidala",
    "Ahsoka Tano", "Yoda", "Mace Windu", "Han Solo", "Princess Leia", "Luke Skywalker",
    "Lando Calrissian", "Rey_Star_Wars", "Poe Dameron", "Finn Star Wars", "Mon Mothma",
    "Kit Fisto", "Captain Rex", "Jar Jar Binks", "Admiral Ackbar", "Aayla Secura",
    "Jyn Erso", "Cassian Andor",
    "Wicket W Warrick", "Chewbacca", "Coleman Trebor", "Captain Panaka"
]

villains = [
    "Count Dooku", "Darth Vader", "Darth Maul", "Kylo Ren", "Darth Sidious",
    "Cad Bane", "General Grievous", "Grand Inquisitor", "Mother Talzin", "Ventress",
    "Stormtrooper", "Snoke", "Jabba", "Savage Opress", "Boba Fett", "Nute Gunray",
    "Captain Phasma", "General Hux", "Darth Plagueis", "Morgan Elsbeth", "Zam Wessel"
    "Jango Fett", "Ziro the Hutt", "Pong Krell", "Director Krennic", "Grand Moff Tarkin"
]

def download_images(character_list, base_folder):
    for name in character_list:
        folder_name = name.replace(" ", "_")
        save_path = os.path.join(base_folder, folder_name)
        os.makedirs(save_path, exist_ok=True)

        try:
            crawler = BingImageCrawler(storage={'root_dir': save_path})
            crawler.crawl(keyword=name, max_num=100)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

download_images(heroes, 'heros')

download_images(villains, 'villains')
