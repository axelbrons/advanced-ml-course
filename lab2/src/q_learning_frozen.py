import gymnasium as gym
import numpy as np
import time

def create_custom_frozen_lake():
    # 1. On crée l'environnement normalement avec les glissades activées
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # 2. On accède à la matrice des probabilités P de l'environnement
    P = env.unwrapped.P
    
    # 3. On parcourt tous les états et toutes les actions
    for state in P:
        for action in P[state]:
            transitions = P[state][action]
            
            # Quand is_slippery=True, 'transitions' contient 3 éléments :
            # Index 0 : Glisser à gauche (base 33.3%)
            # Index 1 : Action voulue / aller tout droit (base 33.3%)
            # Index 2 : Glisser à droite (base 33.3%)
            if len(transitions) == 3:
                # On recrée la liste avec nos nouvelles probabilités (0.1, 0.8, 0.1)
                P[state][action] = [
                    (0, transitions[0][1], transitions[0][2], transitions[0][3]), # 10% gauche
                    (1, transitions[1][1], transitions[1][2], transitions[1][3]), # 80% voulu
                    (0, transitions[2][1], transitions[2][2], transitions[2][3])  # 10% droite
                ]
    return env

def train_frozen_lake(episodes=10000):
    # Création de l'environnement (is_slippery=True active le côté stochastique du cours)
    env = create_custom_frozen_lake()
    
    # Paramètres issus de ton cours
    alpha = 0.1    # Taux d'apprentissage (0 < alpha <= 1) [cite: 347]
    gamma = 0.95   # Facteur d'actualisation (0 <= gamma < 1) [cite: 114]
    epsilon = 0.1  # Paramètre d'exploration epsilon-greedy [cite: 330]

    # 1. Initialisation : Q(s,a) = 0 pour tout s, a [cite: 356]
    # Ici, une matrice 16 états x 4 actions
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(episodes):
        # 4. Choisir aléatoirement l'état initial s_t
        state, _ = env.reset()
        done = False
        
        while not done:
            # 6-10. Politique Epsilon-greedy [cite: 356]
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Exploration
            else:
                action = np.argmax(q_table[state]) # Exploitation (max_a Q(s,a))

            # 11. Obtenir s_{t+1} et R_t de l'environnement
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 12. Mise à jour de la Q-table (Formule de Bellman pondérée) [cite: 345, 356]
            # Q(s,a) = (1-alpha)*Q(s,a) + alpha * [R + gamma * max Q(s', a)]
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            # 13. s_t = s_{t+1}
            state = next_state
            
    env.close()
    return q_table

def test_frozen_lake(q_table, episodes=3):
    """
    Teste l'agent en utilisant la table Q apprise.
    On active render_mode="human" pour voir l'animation.
    """
    env = create_custom_frozen_lake()
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\n--- Début de l'épisode de test {episode + 1} ---")
        
        while not done:
            # Exploitation pure : on choisit l'action avec la plus grande valeur Q
            # C'est l'équivalent de l'application de la politique optimale pi*(s)
            action = np.argmax(q_table[state])
            
            # On effectue l'action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Petite pause pour bien voir l'animation
            time.sleep(0.5)
            
        if reward == 1.0:
            print("Succès ! L'agent a atteint le cadeau.")
        else:
            print("Échec. L'agent est tombé dans un trou.")

    env.close()

def evaluate_agent(env, q_table, n_tests=100):
    success = 0
    for _ in range(n_tests):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(q_table[state]) # On prend la meilleure action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1.0:
                success += 1
    return (success / n_tests) * 100


# --- Exécution principale ---

# 1. On récupère la table Q finale après l'entraînement
final_q_table = train_frozen_lake(episodes=100000)
print("Entraînement terminé. Table Q apprise !")

# 2. On évalue l'agent sans affichage graphique pour calculer son taux de réussite
eval_env = create_custom_frozen_lake()
taux_succes = evaluate_agent(eval_env, final_q_table, n_tests=100)
print(f"Taux de succès : {taux_succes}%")
eval_env.close()

# 3. On lance le test visuel pour voir l'agent évoluer
#test_frozen_lake(final_q_table, episodes=3)