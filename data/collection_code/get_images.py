import simple_image_download.simple_image_download as simp


my_downloader = simp.Downloader()

# Change Direcotory
my_downloader.directory = 'heros/'
my_downloader.extensions = '.jpg'

heroes = ["Rey_Star_Wars"]
# heroes = ["Qui-Gon_Jinn", "Obi-Wan_Kenobi", "Anakin_Skywalker", "Padme_Amidala", "Ahsoka_Tano", "Yoda", "Mace_Windu", "Han_Solo", "Princess_Leia", "Luke_Skywalker", "Lando_Calrissian", "Rey_Star_Wars", "Poe_Dameron", "Finn_Star_Wars", "Mon_Mothma", "Kit_Fisto", "Captain_Rex", "Jar_Jar_Binks", "Admiral_Ackbar", "Aayla_Secura", "Jyn_Erso", "Cassian_Andor"]

for hero in heroes: 
    try:
        my_downloader.download(hero, limit=100, verbose=True)
    except Exception: 
        print("skipping for", hero)
        
        
# my_downloader.directory = 'villains/'

# # Change File extension type
# villains = ["Count_Dooku", "Darth_Vader", "Darth_Maul", "Kylo_Ren", "Darth_Sidious", "Cad_Bane", "General_Grievous", "Grand_Inquisitor", "Mother_Talzin", "Ventress", "Stormtrooper", "Snoke", "Jabba", "Savage_Opress", "Boba_Fett", "Nute_Gunray", "Captain_Phasma", "General_Hux", "Darth_Plagueis", "Morgan_Elsbeth", "Zam_Wessel"]

# for villain in villains:
#     try:
#         my_downloader.download(villain, limit=100, verbose=True)
#     except Exception: 
#         print("skipping for", villain)