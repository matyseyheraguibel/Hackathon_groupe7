import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from source.utils.performance_metric import naive_plan, optimal_strategy, strategy_profit

COLOR_CHARGE    = '#2ca02c'   # vert
COLOR_DISCHARGE = '#d62728'   # rouge

STRATEGY_NAMES = {
    'oracle': 'Oracle',
    'model':  'Modele',
    'naive':  'Naif',
}


def _add_cycle_bands(ax, actions, y_band, band_height=0.05, alpha=0.85):
    for h in range(24):
        if actions[h] == 1:
            color = COLOR_CHARGE
        elif actions[h] == -1:
            color = COLOR_DISCHARGE
        else:
            continue
        rect = mpatches.FancyBboxPatch(
            (h, y_band - band_height / 2),
            width=1,
            height=band_height,
            boxstyle="square,pad=0",
            transform=ax.get_xaxis_transform(),
            color=color,
            alpha=alpha,
            clip_on=False,
            zorder=5
        )
        ax.add_patch(rect)


def _add_strategy_label(ax, y_band, label, color='#333333'):
    ax.text(-0.8, y_band, label,
            transform=ax.get_xaxis_transform(),
            fontsize=6, va='center', ha='right',
            color=color, fontweight='bold')


def _plot_day_with_cycles(ax, n, profil_vrai, profil_predit, profil_veille,
                          df_true_aligned, title_suffix=''):
    """
    Trace un jour avec ses 3 rangees de segments (Oracle/Modele/Naif).
    Appelle _add_cycle_bands et _add_strategy_label.
    """
    heures = range(24)
    ax.plot(heures, profil_vrai,   color='#1f77b4', linewidth=2,   label='Vrai (J)')
    ax.plot(heures, profil_predit, color='#ff7f0e', linewidth=2,   label='Predit (J)',     linestyle='--')
    ax.plot(heures, profil_veille, color='gray',    linewidth=1.5, label='Baseline (J-1)', linestyle=':', alpha=0.7)

    actions_oracle = optimal_strategy(profil_vrai)
    actions_model  = optimal_strategy(profil_predit)
    actions_naive  = naive_plan()

    _add_cycle_bands(ax, actions_oracle, y_band=-0.08, band_height=0.05)
    _add_cycle_bands(ax, actions_model,  y_band=-0.16, band_height=0.05)
    _add_cycle_bands(ax, actions_naive,  y_band=-0.24, band_height=0.05)

    _add_strategy_label(ax, -0.08, STRATEGY_NAMES['oracle'])
    _add_strategy_label(ax, -0.16, STRATEGY_NAMES['model'])
    _add_strategy_label(ax, -0.24, STRATEGY_NAMES['naive'])

    try:
        date_str = df_true_aligned.index[n].strftime('%d/%m/%Y')
        titre = 'Jour ' + str(n) + ' (' + date_str + ')' + title_suffix
    except AttributeError:
        titre = 'Jour index n ' + str(n) + title_suffix

    ax.set_title(titre, fontsize=12)
    heures_list = list(heures)
    ax.set_xticks(heures_list[::4])
    ax.set_xlabel('Heure')
    ax.set_ylabel('Prix (EUR/MWh)')
    ax.grid(True, linestyle='--', alpha=0.6)


def generate_evaluation_plots(df_y_true, df_y_pred, save_path):
    """
    Version enrichie avec meilleur et pire jour par RMSE.
    """
    dossier = os.path.dirname(save_path)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    df_true_aligned  = df_y_true.loc[df_y_pred.index]
    valeurs_vraies   = df_true_aligned.values.flatten()
    valeurs_predites = df_y_pred.values.flatten()
    start_date       = df_true_aligned.index.min()
    index_horaire    = pd.date_range(start=start_date, periods=len(valeurs_vraies), freq='h')
    s_true = pd.Series(valeurs_vraies,   index=index_horaire)
    s_pred = pd.Series(valeurs_predites, index=index_horaire)

    #calcul de meilleur et pire jour
    diff_profit_par_jour = []
    profit_modele_par_jour = []
    profit_oracle_par_jour = []

    for i in range(len(df_true_aligned)):
        profil_vrai = df_true_aligned.iloc[i, :].values
        profil_predit = df_y_pred.iloc[i, :].values
        
        # 1. Obtenir les actions prévues par le modèle et l'oracle
        actions_oracle = optimal_strategy(profil_vrai)
        actions_model  = optimal_strategy(profil_predit)
        
        # 2. Calcul du profit (Action: 1 = Charge/Achat, -1 = Décharge/Vente)
        # Gain = Vente (prix * 1) quand action = -1
        # Coût = Achat (prix * -1) quand action = 1
        # Donc Flux financier = -action * prix
        profit_o = strategy_profit(actions_oracle, profil_vrai)
        profit_m = strategy_profit(actions_model, profil_vrai)
        
        profit_oracle_par_jour.append(profit_o)
        profit_modele_par_jour.append(profit_m)
        
        # On calcule le "manque à gagner" (différence entre l'oracle et le modèle)
        diff_profit_par_jour.append(profit_o - profit_m)

    # Conversion en numpy array pour faciliter la recherche d'index
    diff_profit_par_jour = np.array(diff_profit_par_jour)
    
    # Le meilleur jour est celui où le manque à gagner est le plus faible (proche de 0)
    meilleur_idx = np.argmin(diff_profit_par_jour)
    # Le pire jour est celui où le modèle a "raté" le plus de profit par rapport à l'oracle
    pire_idx = np.argmax(diff_profit_par_jour)

    #-----

    # Layout : 6 graphes -> 6 lignes. On ajoute 2 lignes pour meilleur/pire
    # Positions des 6 jours aleatoires : (3,0) a (4,2)
    # Positions meilleur/pire : (5,0) et (5,1)
    fig = plt.figure(figsize=(22, 40))  # Plus haut pour 2 rangees supplementaires
    gs  = gridspec.GridSpec(6, 3, figure=fig, hspace=0.65, wspace=0.3)
    heures = range(24)

    # ------------------------------------------------------------------
    # GRAPHIQUE 1 : Tendance generale
    # ------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :])
    prix_moyen_journalier_vrai   = s_true.resample('D').mean()
    prix_moyen_journalier_predit = s_pred.resample('D').mean()
    ax0.plot(prix_moyen_journalier_vrai.index,  prix_moyen_journalier_vrai.values,
             label='Prix Moyen Journalier Reel',  color='#1f77b4', linewidth=2)
    ax0.plot(prix_moyen_journalier_predit.index, prix_moyen_journalier_predit.values,
             label='Prix Moyen Journalier Predit', color='#ff7f0e', linewidth=2, linestyle='--')
    ax0.set_title('Tendance Generale des Prix (Moyenne Journaliere)', fontsize=14, pad=10)
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Prix Moyen (EUR/MWh)')
    ax0.grid(True, linestyle='--', alpha=0.6)
    ax0.legend()

    # ------------------------------------------------------------------
    # GRAPHIQUE 2 : Profil intra-journalier moyen + bandes strategie naive
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, :])
    profil_vrai_moyen   = df_true_aligned.mean(axis=0).values
    profil_predit_moyen = df_y_pred.mean(axis=0).values

    ax1.plot(heures, profil_vrai_moyen,   label='Vrai Profil Moyen',
             color='#1f77b4', linewidth=3, marker='o')
    ax1.plot(heures, profil_predit_moyen, label='Profil Predit',
             color='#ff7f0e', linewidth=3, marker='x', linestyle='--')

    plan_naif = naive_plan()
    _add_cycle_bands(ax1, plan_naif, y_band=0.04, band_height=0.06)
    _add_strategy_label(ax1, 0.04, 'Naif')

    ax1.set_title(
        'Profil Intra-journalier Moyen\n'
        '(segments en bas = cycles charge/decharge strategie naive)',
        fontsize=14, pad=10
    )
    ax1.set_xticks(heures)
    ax1.set_xlabel('Heure de la journee')
    ax1.set_ylabel('Prix Moyen (EUR/MWh)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    handles_lignes = [
        plt.Line2D([0], [0], color='#1f77b4', linewidth=3, marker='o',  label='Vrai Profil Moyen'),
        plt.Line2D([0], [0], color='#ff7f0e', linewidth=3, marker='x', linestyle='--', label='Profil Predit'),
        mpatches.Patch(color=COLOR_CHARGE,    label='Charge'),
        mpatches.Patch(color=COLOR_DISCHARGE, label='Decharge'),
    ]
    ax1.legend(handles=handles_lignes, fontsize=9)

    # ------------------------------------------------------------------
    # GRAPHIQUE 3 : Zoom 7 premiers jours
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2, :])
    fin_semaine = start_date + pd.Timedelta(days=7)
    s_true_zoom = s_true[:fin_semaine]
    s_pred_zoom = s_pred[:fin_semaine]
    ax2.plot(s_true_zoom.index, s_true_zoom.values,
             label='Prix Reel',   color='#1f77b4', linewidth=1.5)
    ax2.plot(s_pred_zoom.index, s_pred_zoom.values,
             label='Prix Predit', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax2.set_title('Zoom sur les 7 Premiers Jours de Test', fontsize=14, pad=10)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Prix (EUR/MWh)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # ------------------------------------------------------------------
    # GRAPHIQUES 4 a 9 : 6 jours aleatoires
    # ------------------------------------------------------------------
    jours_aleatoires = np.random.choice(range(1, df_true_aligned.shape[0]), size=6, replace=False)
    positions_grille = [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]

    shared_legend_handles = [
        plt.Line2D([0], [0], color='#1f77b4', linewidth=2,                    label='Vrai (J)'),
        plt.Line2D([0], [0], color='#ff7f0e', linewidth=2,  linestyle='--',   label='Predit (J)'),
        plt.Line2D([0], [0], color='gray',    linewidth=1.5, linestyle=':',   label='Baseline (J-1)'),
        mpatches.Patch(color=COLOR_CHARGE,    label='Charge'),
        mpatches.Patch(color=COLOR_DISCHARGE, label='Decharge'),
    ]

    for i, n in enumerate(jours_aleatoires):
        ax = fig.add_subplot(gs[positions_grille[i]])
        profil_vrai   = df_true_aligned.iloc[n, :].values
        profil_predit = df_y_pred.iloc[n, :].values
        profil_veille = df_true_aligned.iloc[n - 1, :].values

        _plot_day_with_cycles(ax, n, profil_vrai, profil_predit, profil_veille,
                              df_true_aligned)

        if i == 0:
            ax.legend(handles=shared_legend_handles, fontsize=8, loc='upper right')

    # ------------------------------------------------------------------
    # GRAPHIQUE 10 : MEILLEUR JOUR
    # ------------------------------------------------------------------
    ax_best = fig.add_subplot(gs[5, 0])
    n_best = meilleur_idx
    profil_vrai_best   = df_true_aligned.iloc[n_best, :].values
    profil_predit_best = df_y_pred.iloc[n_best, :].values
    profil_veille_best = df_true_aligned.iloc[n_best - 1, :].values if n_best > 0 else np.zeros(24)

    _plot_day_with_cycles(ax_best, n_best, profil_vrai_best, profil_predit_best, profil_veille_best,
                          df_true_aligned, title_suffix=' [MEILLEUR]')
    
    
    prof_m_best = profit_modele_par_jour[n_best]
    prof_o_best = profit_oracle_par_jour[n_best]
    ax_best.text(0.98, 0.02, f'Profit Modèle: {prof_m_best:.1f} €\nOracle: {prof_o_best:.1f} €',
                 transform=ax_best.transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    
    # ------------------------------------------------------------------
    # GRAPHIQUE 11 : PIRE JOUR
    # ------------------------------------------------------------------
    ax_worst = fig.add_subplot(gs[5, 1])
    n_worst = pire_idx
    profil_vrai_worst   = df_true_aligned.iloc[n_worst, :].values
    profil_predit_worst = df_y_pred.iloc[n_worst, :].values
    profil_veille_worst = df_true_aligned.iloc[n_worst - 1, :].values if n_worst > 0 else np.zeros(24)

    _plot_day_with_cycles(ax_worst, n_worst, profil_vrai_worst, profil_predit_worst, profil_veille_worst,
                          df_true_aligned, title_suffix=' [PIRE]')
    
    prof_m_worst = profit_modele_par_jour[n_worst]
    prof_o_worst = profit_oracle_par_jour[n_worst]
    ax_worst.text(0.98, 0.02, f'Profit Modèle: {prof_m_worst:.1f} €\nOracle: {prof_o_worst:.1f} €',
                  transform=ax_worst.transAxes, fontsize=10, ha='right', va='bottom',
                  bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Dashboard enrichi sauvegardé ici : ' + save_path)
    print(f"  Meilleur jour (indice {n_best}) : Profit Modèle = {prof_m_best:.1f} € | Oracle = {prof_o_best:.1f} €")
    print(f"  Pire jour (indice {n_worst}) : Profit Modèle = {prof_m_worst:.1f} € | Oracle = {prof_o_worst:.1f} €")