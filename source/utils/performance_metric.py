"""Metrique de performance : valeur economique du modele pour l'arbitrage batterie.

Batterie parfaite 1 MW / 2 MWh : 1 MWh charge/decharge par heure, 8 MWh vendus/jour
max, on decharge apres avoir charge, batterie vide en fin de journee. Un cycle coute
8 EUR (soit 4 EUR par MWh decharge).

Deux references de comparaison, chacune donnant un score dans (-inf, 1] :
 - ORACLE : planning optimal calcule sur les vrais prix (plafond theorique, devin parfait).
 - NAIVE : planning fixe identique tous les jours (charge/decharge a heures fixes),
   c'est-a-dire ce qu'on ferait sans aucun modele de prevision.

Pour chacune, score = profit_realise / profit_reference, ou profit_realise est le profit
d'un planning DECIDE sur les prix predits mais ENCAISSE aux vrais prix.

Entrees des fonctions de score : Y_test et Y_test_pred, arrays (n_jours, 24) de vrais
prix EUR/MWh (sortie de get_pred_df_inverted).
"""

import numpy as np

MAX_DISCHARGE = 4  # MWh vendus par jour au maximum (2 cycles)
CAPACITY = 2       # MWh stockables
CYCLE_COST = 8.0   # EUR par cycle complet (charge + decharge), soit 4 EUR/MWh decharge


# Planning naif fixe, applique a l'identique chaque jour (2 cycles de 2 MWh).
# Charge heures 1-2, decharge 7-8, charge 13-14, decharge 17-18.
NAIVE_CHARGE_HOURS = (1, 2, 13, 14)
NAIVE_DISCHARGE_HOURS = (7, 8, 17, 18)


def naive_plan():
    """Planning naif fixe (24 valeurs : +1 charge, -1 decharge, 0 rien), le meme chaque jour."""
    plan = np.zeros(24, int)
    for h in NAIVE_CHARGE_HOURS:
        plan[h] = 1
    for h in NAIVE_DISCHARGE_HOURS:
        plan[h] = -1
    return plan


def optimal_strategy(prices):
    """Renvoie le planning optimal d'un jour (24 valeurs : +1 charge, -1 decharge, 0 rien).

    Programmation dynamique sur l'etat (niveau de charge, MWh deja vendus) : a chaque
    heure on peut charger (payer le prix), decharger (encaisser le prix moins le cout de
    cyclage) ou ne rien faire ; on garde le chemin le plus rentable finissant batterie vide.

    Note : un cycle = 2 heures de decharge (2 MWh vendus). Le cout par heure de decharge
    est donc CYCLE_COST / 2.
    """
    prices = np.asarray(prices, float).ravel()
    NEG = -np.inf
    val = np.full((CAPACITY + 1, MAX_DISCHARGE + 1), NEG)
    val[0, 0] = 0.0
    back = np.zeros((24, CAPACITY + 1, MAX_DISCHARGE + 1), int)

    half_cycle_cost = CYCLE_COST / 2.0  # cout par heure de decharge (= par MWh vendu)

    for h, price in enumerate(prices):
        nxt = np.full_like(val, NEG)
        for soc in range(CAPACITY + 1):
            for sold in range(MAX_DISCHARGE + 1):
                v = val[soc, sold]
                if v == NEG:
                    continue
                moves = [(soc, sold, 0, v)]  # rien
                if soc < CAPACITY:
                    moves.append((soc + 1, sold, 1, v - price))  # charger
                if soc > 0 and sold < MAX_DISCHARGE:
                    moves.append((soc - 1, sold + 1, -1, v + price - half_cycle_cost))  # decharger
                for ns, nd, act, nv in moves:
                    if nv > nxt[ns, nd]:
                        nxt[ns, nd] = nv
                        back[h, ns, nd] = act
        val = nxt

    # On remonte le meilleur etat final a batterie vide pour reconstruire le planning.
    soc, sold = 0, int(np.argmax(val[0]))
    actions = np.zeros(24, int)
    for h in range(23, -1, -1):
        act = back[h, soc, sold]
        actions[h] = act
        soc -= act
        if act == -1:
            sold -= 1
    return actions


def strategy_profit(actions, prices):
    """Profit en EUR d'un planning donne, valorise a un jeu de prix donne.

    Un cycle = 2 heures de decharge. Le cout total = CYCLE_COST * nombre_de_cycles,
    avec nombre_cycles = nombre_heures_decharge / 2.
    """
    actions = np.asarray(actions, int).ravel()
    prices = np.asarray(prices, float).ravel()
    sells = prices[actions == -1]
    num_discharge_hours = np.count_nonzero(actions == -1)
    num_cycles = num_discharge_hours / 2.0
    profit = sells.sum() - CYCLE_COST * num_cycles - prices[actions == 1].sum()
    return float(profit)


def _daily_profits(Y_test, Y_test_pred):
    """Profits journaliers (EUR) : realise (modele), oracle (vrais prix) et naif (plan fixe)."""
    Y_test = np.asarray(Y_test, float)
    Y_test_pred = np.asarray(Y_test_pred, float)
    plan_naif = naive_plan()
    realized, oracle, naive = [], [], []
    for t, p in zip(Y_test, Y_test_pred):
        realized.append(strategy_profit(optimal_strategy(p), t))  # decide sur predit, paye au vrai
        oracle.append(strategy_profit(optimal_strategy(t), t))    # devin parfait
        naive.append(strategy_profit(plan_naif, t))               # plan fixe, paye au vrai
    return np.array(realized), np.array(oracle), np.array(naive)


def eval_performance(Y_test, Y_test_pred):
    """Taux de capture vs ORACLE, dans (-inf, 1] : part du profit optimal captee par le modele."""
    realized, oracle, _ = _daily_profits(Y_test, Y_test_pred)
    tot_oracle = oracle.sum()
    return 1.0 if tot_oracle <= 0 else realized.sum() / tot_oracle


def eval_vs_naive(Y_test, Y_test_pred):
    """Taux de capture vs NAIVE : profit du modele / profit de la strategie fixe.

    > 1 : le modele bat la strategie naive ; < 1 : il fait moins bien.
    """
    realized, _, naive = _daily_profits(Y_test, Y_test_pred)
    tot_naive = naive.sum()
    return np.nan if tot_naive == 0 else realized.sum() / tot_naive


def eval_all(Y_test, Y_test_pred):
    """Renvoie un dict de scores comparatifs complets.

    Cles :
        capture_vs_oracle : profit modele / profit oracle (dans (-inf, 1])
        capture_vs_naive  : profit modele / profit naif (>1 = bat la naive)
        days_win          : nb de jours ou le modele fait STRICTEMENT mieux que la naive
        days_lose         : nb de jours ou le modele fait STRICTEMENT moins bien que la naive
        days_equal        : nb de jours a egalite
        profit_model      : profit total du modele (EUR)
        profit_oracle     : profit total oracle (EUR)
        profit_naive      : profit total naif (EUR)
    """
    realized, oracle, naive = _daily_profits(Y_test, Y_test_pred)
    diff = realized - naive
    tot_oracle, tot_naive = oracle.sum(), naive.sum()
    return {
        "capture_vs_oracle": 1.0 if tot_oracle <= 0 else realized.sum() / tot_oracle,
        "capture_vs_naive":  np.nan if tot_naive == 0 else realized.sum() / tot_naive,
        "days_win":   int((diff > 0).sum()),
        "days_lose":  int((diff < 0).sum()),
        "days_equal": int((diff == 0).sum()),
        "profit_model":  float(realized.sum()),
        "profit_oracle": float(oracle.sum()),
        "profit_naive":  float(naive.sum()),
    }


def affichage_metrics(df_y_test_inverted, df_final_preds):
    scores = eval_all(df_y_test_inverted.values, df_final_preds.values)
    print("\n" + "="*50)
    print("RESULTATS ECONOMIQUES (ARBITRAGE BATTERIE)")
    print("="*50)
    print(f"Profit total du Modele    : {scores['profit_model']:>9.2f} EUR")
    print(f"Profit total Oracle (Max) : {scores['profit_oracle']:>9.2f} EUR")
    print(f"Profit total Naif (Fixe)  : {scores['profit_naive']:>9.2f} EUR")
    print("-" * 50)
    print(f"Taux de capture vs Oracle : {scores['capture_vs_oracle']*100:>8.2f} % (Borne sup)")
    print(f"Taux de capture vs Naif   : {scores['capture_vs_naive']*100:>8.2f} % (>100 = IA utile)")
    print("-" * 50)
    print(f"Jours gagnants (Modele > Naif) : {scores['days_win']} jours")
    print(f"Jours perdants (Modele < Naif) : {scores['days_lose']} jours")
    print(f"Jours a egalite               : {scores['days_equal']} jours")
    print("="*50 + "\n")
