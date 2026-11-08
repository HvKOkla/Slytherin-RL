import pygame
import sys
from stable_baselines3 import PPO
from games.snake.logic import SnakeLogic  # <-- MODIFIÉ ICI

# Configuration de l'affichage
TAILLE_CASE = 40
TAILLE_GRILLE = 10
LARGEUR = TAILLE_CASE * TAILLE_GRILLE
HAUTEUR = LARGEUR

# Couleurs
NOIR = (20, 20, 20)
BLANC = (255, 255, 255)
VERT = (0, 255, 0)
VERT_FONCE = (0, 150, 0)
ROUGE = (255, 50, 50)
GRIS = (50, 50, 50)
JAUNE = (255, 255, 0)

def dessiner(ecran, game, police, score_max):
    ecran.fill(NOIR)
    
    for x in range(0, LARGEUR, TAILLE_CASE):
        pygame.draw.line(ecran, GRIS, (x, 0), (x, HAUTEUR))
    for y in range(0, HAUTEUR, TAILLE_CASE):
        pygame.draw.line(ecran, GRIS, (0, y), (LARGEUR, y))

    px, py = game.food
    pygame.draw.rect(ecran, ROUGE, (px * TAILLE_CASE, py * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE))

    for i, (sx, sy) in enumerate(game.snake):
        couleur = VERT if i == 0 else VERT_FONCE
        pygame.draw.rect(ecran, couleur, (sx * TAILLE_CASE, sy * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE))

    texte_score = police.render(f"Score : {game.score}  |  Record : {score_max}", True, BLANC)
    ecran.blit(texte_score, (10, 10))
    
    texte_ia = police.render("Slytherin v1.0", True, JAUNE)
    ecran.blit(texte_ia, (LARGEUR - texte_ia.get_width() - 10, 10))

    pygame.display.flip()

def main():
    pygame.init()
    ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption("Slytherin-RL v1.0 - Live Play")
    police = pygame.font.SysFont("Arial", 24)
    horloge = pygame.time.Clock()
    
    game = SnakeLogic(TAILLE_GRILLE)
    
    # Le chemin vers le modèle (qui est à la racine du projet)
    modele_path = "Slytherin_v1" 
    print(f"Chargement du modèle : {modele_path}")
    model = PPO.load(modele_path)
    
    obs = game.get_observation()
    en_cours = True
    score_max = 0

    while en_cours:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False

        action, _ = model.predict(obs, deterministic=True)
        game.step(action)
        obs = game.get_observation()
        
        dessiner(ecran, game, police, score_max)
        horloge.tick(15)

        if game.done:
            if game.score > score_max:
                score_max = game.score
            pygame.time.wait(500)
            game.reset()
            obs = game.get_observation()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()