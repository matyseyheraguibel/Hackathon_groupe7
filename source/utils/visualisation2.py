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


def generate_seasonal_analysis(df_y_true, df_y_pred, save_path):
    """
    Analyse saisonnière : 3 graphes pour valider les variations de performance au cours de l'année.
    Compare par rapport à l'ORACLE (meilleure stratégie possible).
    
    1. Courbe lissée (rolling average 14j) du taux de capture vs Oracle : tendance générale
    2. Barres par mois : taux de capture moyen (vs Oracle) pour chaque mois
    3. Boxplot par mois : distribution du taux de capture pour chaque mois (voir variance)
    """
    from source.utils.performance_metric import optimal_strategy, strategy_profit
    
    dossier = os.path.dirname(save_path)
    if dossier:
        os.makedirs(dossier, exist_ok=True)
    
    df_true_aligned = df_y_true.loc[df_y_pred.index]
    
    # Calculer le taux de capture journalier vs Oracle
    daily_capture_vs_oracle = []
    daily_profits_model = []
    daily_profits_oracle = []
    
    for i in range(len(df_true_aligned)):
        profil_vrai = df_true_aligned.iloc[i, :].values
        profil_predit = df_y_pred.iloc[i, :].values
        
        # Stratégie du modèle (basée sur les prix prédits)
        actions_model = optimal_strategy(profil_predit)
        # Stratégie oracle (basée sur les vrais prix)
        actions_oracle = optimal_strategy(profil_vrai)
        
        # Calculer les profits (les deux évalués sur les vrais prix)
        profit_m = strategy_profit(actions_model, profil_vrai)
        profit_o = strategy_profit(actions_oracle, profil_vrai)
        
        daily_profits_model.append(profit_m)
        daily_profits_oracle.append(profit_o)
        
        # Taux de capture : profit_modele / profit_oracle
        if profit_o != 0:
            capture = profit_m / profit_o
        else:
            capture = np.nan
        daily_capture_vs_oracle.append(capture)
    
    # Créer une série temporelle
    s_capture = pd.Series(daily_capture_vs_oracle, index=df_true_aligned.index)
    
    # Créer figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)
    
    # ------------------------------------------------------------------
    # GRAPHIQUE 1 : Courbe lissée du taux de capture vs Oracle
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Rolling average sur 14 jours pour lisser
    s_capture_smooth = s_capture.rolling(window=14, center=True).mean()
    
    ax1.plot(s_capture.index, s_capture.values, color='#1f77b4', alpha=0.3, linewidth=0.5, label='Taux journalier')
    ax1.plot(s_capture_smooth.index, s_capture_smooth.values, color='#d62728', linewidth=2.5, label='Lissé (14j rolling mean)')
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1, label='100% du profit Oracle')
    ax1.fill_between(s_capture_smooth.index, 1.0, s_capture_smooth.values, 
                     where=(s_capture_smooth.values >= 1.0), alpha=0.2, color='green', label='> 100% Oracle')
    ax1.fill_between(s_capture_smooth.index, 1.0, s_capture_smooth.values,
                     where=(s_capture_smooth.values < 1.0), alpha=0.2, color='red', label='< 100% Oracle')
    
    ax1.set_title('Tendance du Taux de Capture vs Oracle (Année 2026)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Ratio Profit Modèle / Profit Oracle')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1.2)
    
    # ------------------------------------------------------------------
    # GRAPHIQUE 2 : Barres par mois du taux de capture moyen
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Grouper par mois
    monthly_capture = s_capture.groupby(s_capture.index.to_period('M')).mean()
    monthly_capture_monthly_str = [str(m) for m in monthly_capture.index]
    
    colors_bar = ['green' if x >= 1.0 else 'red' for x in monthly_capture.values]
    bars = ax2.bar(range(len(monthly_capture)), monthly_capture.values, color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Ajouter la valeur sur chaque barre
    for i, (bar, val) in enumerate(zip(bars, monthly_capture.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='100% Oracle')
    ax2.set_xticks(range(len(monthly_capture)))
    ax2.set_xticklabels(monthly_capture_monthly_str, rotation=45)
    ax2.set_ylabel('Taux de Capture Moyen')
    ax2.set_title('Performance Moyenne par Mois (Taux de Capture vs Oracle)', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax2.legend()
    ax2.set_ylim(0, 1.2)
    
    # ------------------------------------------------------------------
    # GRAPHIQUE 3 : Boxplot par mois
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Préparer les données pour le boxplot
    monthly_groups = []
    monthly_labels = []
    for period in s_capture.index.to_period('M').unique():
        mask = s_capture.index.to_period('M') == period
        data = s_capture[mask].dropna().values
        if len(data) > 0:
            monthly_groups.append(data)
            monthly_labels.append(str(period))
    
    bp = ax3.boxplot(monthly_groups, patch_artist=True)
    ax3.set_xticklabels(monthly_labels)
    
    # Colorer les boxplots
    for patch in bp['boxes']:
        patch.set_facecolor('#2ca02c')
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linestyle='--', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='100% Oracle')
    ax3.set_ylabel('Taux de Capture')
    ax3.set_xlabel('Mois')
    ax3.set_title('Distribution du Taux de Capture par Mois (Médiane en rouge)', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax3.legend()
    ax3.set_ylim(0, 1.2)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print('\n' + '='*70)
    print('Analyse Saisonnière Sauvegardée (vs Oracle)')
    print('='*70)
    print(f'Chemin : {save_path}')
    print(f'Nombre de jours analysés : {len(daily_capture_vs_oracle)}')
    print(f'Taux de capture moyen (annuel) : {s_capture.mean():.2f}')
    print(f'Écart-type : {s_capture.std():.2f}')
    print(f'Min : {s_capture.min():.2f}, Max : {s_capture.max():.2f}')
    print('\nRésumé par mois :')
    for period, val in monthly_capture.items():
        print(f'  {period} : {val:.2f}')
    print('='*70 + '\n')


def generate_seasonal_analysis_simple(df_y_true, df_y_pred, save_path):
    """
    Analyse saisonnière SIMPLE et LISIBLE :
    - 1 graphe avec barres mensuelles colorées + courbe de tendance
    - Lignes de référence (80%, 90%, 100%)
    - Taux de capture vs Oracle en % grand sur chaque barre
    - Tableau récapitulatif des meilleurs/pires mois
    """
    from source.utils.performance_metric import optimal_strategy, strategy_profit
    
    dossier = os.path.dirname(save_path)
    if dossier:
        os.makedirs(dossier, exist_ok=True)
    
    df_true_aligned = df_y_true.loc[df_y_pred.index]
    
    # Calculer le taux de capture journalier vs Oracle
    daily_capture_vs_oracle = []
    
    for i in range(len(df_true_aligned)):
        profil_vrai = df_true_aligned.iloc[i, :].values
        profil_predit = df_y_pred.iloc[i, :].values
        
        actions_model = optimal_strategy(profil_predit)
        actions_oracle = optimal_strategy(profil_vrai)
        
        profit_m = strategy_profit(actions_model, profil_vrai)
        profit_o = strategy_profit(actions_oracle, profil_vrai)
        
        if profit_o != 0:
            capture = profit_m / profit_o
        else:
            capture = np.nan
        daily_capture_vs_oracle.append(capture)
    
    s_capture = pd.Series(daily_capture_vs_oracle, index=df_true_aligned.index)
    
    # Grouper par mois et calculer moyennes
    monthly_capture = s_capture.groupby(s_capture.index.to_period('M')).mean()
    monthly_capture_pct = monthly_capture * 100  # En pourcentage
    
    # Créer figure SIMPLE
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Couleurs en fonction de la performance
    def get_color(val):
        if val >= 100:
            return '#2ca02c'  # Vert
        elif val >= 90:
            return '#ff7f0e'  # Orange
        elif val >= 80:
            return '#ffbb78'  # Orange clair
        else:
            return '#d62728'  # Rouge
    
    colors = [get_color(v) for v in monthly_capture_pct.values]
    
    # Barres mensuelles
    x_pos = np.arange(len(monthly_capture))
    bars = ax.bar(x_pos, monthly_capture_pct.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter le texte % en gros sur les barres
    for bar, val in zip(bars, monthly_capture_pct.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Courbe de tendance (rolling mean sur les jours)
    s_capture_smooth = s_capture.rolling(window=14, center=True).mean() * 100
    dates_monthly = monthly_capture.index.to_timestamp()
    # Interpoler la courbe lissée pour chaque mois
    dates_daily = s_capture_smooth.index
    days_for_months = [dates_daily[d.strftime('%Y-%m') == dates_monthly[i].strftime('%Y-%m')].mean() 
                       for i, _ in enumerate(monthly_capture)]
    # Simplification : juste tracer la courbe lissée en arrière-plan
    ax.plot(dates_daily, s_capture_smooth.values, color='#1f77b4', linewidth=3, alpha=0.6, label='Tendance lissée (14j)', zorder=1)
    
    # Lignes de référence
    ax.axhline(y=100, color='black', linestyle='-', linewidth=2, alpha=0.7, label='100% Oracle (parfait)')
    ax.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='90%')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='80%')
    
    # Configuration des axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in monthly_capture.index], fontsize=11, fontweight='bold')
    ax.set_ylabel('Taux de Capture vs Oracle (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mois 2026', fontsize=13, fontweight='bold')
    ax.set_title('Performance Mensuelle - Analyse Saisonnière 2026\\nComparaison vs Oracle (Meilleure Stratégie Possible)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, max(monthly_capture_pct.values) + 15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend(fontsize=11, loc='upper left')
    
    # Ajouter tableau récapitulatif en bas
    sorted_monthly = monthly_capture_pct.sort_values(ascending=False)
    best_month = sorted_monthly.index[0]
    worst_month = sorted_monthly.index[-1]
    best_val = sorted_monthly.values[0]
    worst_val = sorted_monthly.values[-1]
    
    textstr = f"🟢 MEILLEUR: {best_month} ({best_val:.1f}%)  |  🔴 PIRE: {worst_month} ({worst_val:.1f}%)"
    ax.text(0.5, -0.12, textstr, transform=ax.transAxes, fontsize=12, 
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print('\\n' + '='*70)
    print('✓ Analyse Saisonnière SIMPLE sauvegardée')
    print('='*70)
    print(f'Chemin : {save_path}')
    print(f'\\nStatistiques (vs Oracle) :')
    print(f'  Moyenne annuelle : {s_capture.mean()*100:.1f}%')
    print(f'  Min : {s_capture.min()*100:.1f}%')
    print(f'  Max : {s_capture.max()*100:.1f}%')
    print(f'\\nRésumé par mois :')
    for period in monthly_capture.index:
        val = monthly_capture[period]
        symbol = '🟢' if val >= 0.90 else '🟡' if val >= 0.80 else '🔴'
        print(f'  {symbol} {period} : {val*100:.1f}%')
    print('='*70 + '\\n')



def generate_seasonal_analysis_v2(df_y_true, df_y_pred, save_path):
    """
    Analyse saisonnielle version intermediaire : 2 graphes clairs.
    
    Graphe 1 (haut) : Evolution journaliere lissee du taux de capture vs Oracle
                      + zones colorees meilleur/pire periode
    Graphe 2 (bas)  : Barres mensuelles avec ecart-type (barres d erreur),
                      taux en % sur chaque barre, couleurs semantiques
    """
    from source.utils.performance_metric import optimal_strategy, strategy_profit

    dossier = os.path.dirname(save_path)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    df_true_aligned = df_y_true.loc[df_y_pred.index]

    # --- Calcul journalier ---
    daily_capture = []
    for i in range(len(df_true_aligned)):
        pv = df_true_aligned.iloc[i, :].values
        pp = df_y_pred.iloc[i, :].values
        pm = strategy_profit(optimal_strategy(pp), pv)
        po = strategy_profit(optimal_strategy(pv), pv)
        daily_capture.append(pm / po if po != 0 else np.nan)

    s = pd.Series(daily_capture, index=df_true_aligned.index)
    s_pct = s * 100

    monthly_mean = s_pct.groupby(s_pct.index.to_period("M")).mean()
    monthly_std  = s_pct.groupby(s_pct.index.to_period("M")).std()
    month_labels = [str(m) for m in monthly_mean.index]

    # --- Figure 2 graphes ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12),
                                    gridspec_kw={"height_ratios": [2, 3], "hspace": 0.45})

    # ================================================================
    # GRAPHE 1 : Courbe journaliere lissee
    # ================================================================
    smooth = s_pct.rolling(window=14, center=True).mean()

    ax1.fill_between(s_pct.index, smooth, 100,
                     where=(smooth >= 100), alpha=0.15, color="green", interpolate=True)
    ax1.fill_between(s_pct.index, smooth, 100,
                     where=(smooth < 100), alpha=0.15, color="red", interpolate=True)
    ax1.plot(s_pct.index, s_pct.values,
             color="#aec6e8", linewidth=0.5, alpha=0.5, label="Taux journalier")
    ax1.plot(smooth.index, smooth.values,
             color="#1f77b4", linewidth=2.5, label="Tendance lissee (14j)")

    ax1.axhline(100, color="black",  ls="--", lw=1.5, alpha=0.6, label="100% Oracle")
    ax1.axhline(90,  color="green",  ls=":",  lw=1.2, alpha=0.5, label="90%")
    ax1.axhline(80,  color="orange", ls=":",  lw=1.2, alpha=0.5, label="80%")

    # Annoter le meilleur et pire mois directement sur la courbe
    best_m  = monthly_mean.idxmax()
    worst_m = monthly_mean.idxmin()
    for period, label, color in [(best_m, "Meilleur mois", "green"),
                                  (worst_m, "Pire mois", "red")]:
        mask = s_pct.index.to_period("M") == period
        mid_date = s_pct.index[mask][len(s_pct.index[mask])//2]
        val = monthly_mean[period]
        ax1.annotate(f"{label}\n{period}\n({val:.0f}%)",
                     xy=(mid_date, smooth[mask].mean()),
                     xytext=(0, 25), textcoords="offset points",
                     fontsize=9, fontweight="bold", color=color,
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                     ha="center")

    ax1.set_title("Evolution du Taux de Capture vs Oracle (tendance 14j)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Taux de Capture (%)", fontsize=11)
    ax1.set_ylim(max(0, s_pct.min() - 10), min(150, s_pct.max() + 15))
    ax1.grid(True, axis="y", ls=":", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper left", ncol=2)

    # ================================================================
    # GRAPHE 2 : Barres mensuelles + ecart-type
    # ================================================================
    x = np.arange(len(monthly_mean))
    colors = []
    for v in monthly_mean.values:
        if v >= 90:   colors.append("#2ca02c")   # vert
        elif v >= 80: colors.append("#ff7f0e")   # orange
        elif v >= 70: colors.append("#ffbb78")   # jaune
        else:         colors.append("#d62728")   # rouge

    bars = ax2.bar(x, monthly_mean.values, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=1.2, width=0.6, zorder=3)
    ax2.errorbar(x, monthly_mean.values, yerr=monthly_std.values,
                 fmt="none", color="black", capsize=6, capthick=1.5,
                 elinewidth=1.5, zorder=4, label="Ecart-type (variabilite)")

    # Valeur + nombre de jours sur chaque barre
    n_jours = s_pct.groupby(s_pct.index.to_period("M")).count()
    for i, (bar, mean, std, nj) in enumerate(zip(bars, monthly_mean.values,
                                                   monthly_std.values, n_jours.values)):
        y_text = mean + std + 2.5
        ax2.text(bar.get_x() + bar.get_width()/2, y_text,
                 f"{mean:.1f}%", ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color="black")
        ax2.text(bar.get_x() + bar.get_width()/2, 3,
                 f"n={nj}j", ha="center", va="bottom",
                 fontsize=8, color="white", fontweight="bold")

    ax2.axhline(100, color="black", ls="--", lw=1.5, alpha=0.6, label="100% Oracle")
    ax2.axhline(90,  color="green", ls=":",  lw=1.2, alpha=0.5, label="Seuil 90%")
    ax2.axhline(80,  color="orange",ls=":",  lw=1.2, alpha=0.5, label="Seuil 80%")

    # Legende couleurs
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#2ca02c", label=">= 90% (Bon)"),
        Patch(color="#ff7f0e", label="80-90% (Correct)"),
        Patch(color="#ffbb78", label="70-80% (Faible)"),
        Patch(color="#d62728", label="< 70% (Mauvais)"),
    ]
    l1 = ax2.legend(handles=legend_patches, fontsize=9, loc="upper right", title="Performance")
    ax2.add_artist(l1)
    ax2.legend(fontsize=9, loc="upper left")

    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Taux de Capture Moyen (%)", fontsize=11)
    ax2.set_xlabel("Mois", fontsize=11)
    ax2.set_title("Performance Mensuelle vs Oracle  |  barres d erreur = variabilite dans le mois",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(monthly_mean.values + monthly_std.values) + 20)
    ax2.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

    plt.suptitle(f"Analyse Saisonniere — Taux de Capture vs Oracle\n"
                 f"Periode : {df_true_aligned.index[0].strftime('%d/%m/%Y')} -> "
                 f"{df_true_aligned.index[-1].strftime('%d/%m/%Y')}  |  "
                 f"Moyenne annuelle : {s_pct.mean():.1f}%",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "="*65)
    print("Analyse Saisonniere v2 sauvegardee")
    print("="*65)
    print(f"Chemin : {save_path}")
    print(f"Moyenne annuelle : {s_pct.mean():.1f}%  |  Ecart-type : {s_pct.std():.1f}%")
    print("\nMois (taux moyen +/- std) :")
    for p in monthly_mean.index:
        flag = "MEILLEUR" if p == best_m else ("PIRE" if p == worst_m else "")
        print(f"  {p} : {monthly_mean[p]:.1f}% +/- {monthly_std[p]:.1f}%  {flag}")
    print("="*65 + "\n")


def generate_seasonal_analysis_v3(df_y_true, df_y_pred, save_path):
    """
    Analyse saisonniere — 3 graphes :
      1. Moyenne glissante 7j du ratio profit_modele / profit_oracle
      2. Bar chart mensuel : profit total Oracle vs Modele (EUR) cote a cote
      3. Boxplot mensuel du taux de capture (ratio journalier) + ligne 100% Oracle
    """
    from source.utils.performance_metric import optimal_strategy, strategy_profit

    dossier = os.path.dirname(save_path)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    df_true_aligned = df_y_true.loc[df_y_pred.index]

    # --- Calculs journaliers ---
    daily_ratio   = []   # profit_modele / profit_oracle
    daily_profit_model  = []
    daily_profit_oracle = []

    for i in range(len(df_true_aligned)):
        pv = df_true_aligned.iloc[i, :].values
        pp = df_y_pred.iloc[i, :].values
        pm = strategy_profit(optimal_strategy(pp), pv)
        po = strategy_profit(optimal_strategy(pv), pv)
        daily_profit_model.append(pm)
        daily_profit_oracle.append(po)
        daily_ratio.append(pm / po if po != 0 else np.nan)

    idx = df_true_aligned.index
    s_ratio  = pd.Series(daily_ratio,          index=idx)
    s_pm     = pd.Series(daily_profit_model,   index=idx)
    s_po     = pd.Series(daily_profit_oracle,  index=idx)

    # Agregats mensuels
    monthly_pm    = s_pm.groupby(s_pm.index.to_period("M")).sum()
    monthly_po    = s_po.groupby(s_po.index.to_period("M")).sum()
    monthly_ratio_groups = [
        s_ratio[s_ratio.index.to_period("M") == p].dropna().values
        for p in s_ratio.index.to_period("M").unique()
    ]
    month_labels = [str(p) for p in s_ratio.index.to_period("M").unique()]

    # --- Figure ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={"hspace": 0.5})
    ax1, ax2, ax3 = axes

    # ================================================================
    # GRAPHE 1 : Moyenne glissante 7j du ratio
    # ================================================================
    roll7 = s_ratio.rolling(window=7, center=True).mean()

    ax1.plot(s_ratio.index, s_ratio.values,
             color="#aec6e8", lw=0.6, alpha=0.45, label="Ratio journalier")
    ax1.plot(roll7.index, roll7.values,
             color="#1f77b4", lw=2.5, label="Moyenne glissante 7j")
    ax1.axhline(1.0, color="black", ls="--", lw=1.5, alpha=0.7,
                label="100 % Oracle")
    ax1.fill_between(roll7.index, roll7.values, 1.0,
                     where=(roll7 >= 1.0), alpha=0.12, color="green", interpolate=True)
    ax1.fill_between(roll7.index, roll7.values, 1.0,
                     where=(roll7 < 1.0),  alpha=0.12, color="red",   interpolate=True)
    ax1.set_ylabel("Ratio  Modele / Oracle", fontsize=11)
    ax1.set_title("Taux de Capture vs Oracle — Moyenne Glissante 7 Jours",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(axis="y", ls=":", alpha=0.4)
    ymin = max(0, roll7.dropna().min() - 0.05)
    ymax = min(1.5, roll7.dropna().max() + 0.1)
    ax1.set_ylim(ymin, ymax)

    # ================================================================
    # GRAPHE 2 : Bar chart mensuel Oracle vs Modele (profit total EUR)
    # ================================================================
    x   = np.arange(len(monthly_pm))
    w   = 0.38
    b_o = ax2.bar(x - w/2, monthly_po.values, width=w, color="#1f77b4",
                  alpha=0.85, edgecolor="black", lw=1.0, label="Oracle (max possible)")
    b_m = ax2.bar(x + w/2, monthly_pm.values, width=w, color="#ff7f0e",
                  alpha=0.85, edgecolor="black", lw=1.0, label="Modele")

    for bar in b_o:
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, v + max(monthly_po.values)*0.01,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#1f77b4", fontweight="bold")
    for bar in b_m:
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, v + max(monthly_po.values)*0.01,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#d62728", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Profit Total (EUR)", fontsize=11)
    ax2.set_title("Profit Mensuel Total : Oracle vs Modele (EUR)",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    # ================================================================
    # GRAPHE 3 : Boxplot mensuel du taux de capture
    # ================================================================
    bp = ax3.boxplot(monthly_ratio_groups, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2ca02c")
        patch.set_alpha(0.65)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", lw=1.5, ls="--")
    for cap in bp["caps"]:
        cap.set(color="black", lw=1.5)
    for median in bp["medians"]:
        median.set(color="red", lw=2.5)
    for flier in bp["fliers"]:
        flier.set(marker="o", color="gray", alpha=0.4, markersize=3)

    ax3.axhline(1.0, color="black", ls="--", lw=1.8, alpha=0.7, label="100 % Oracle")
    ax3.set_xticks(range(1, len(month_labels) + 1))
    ax3.set_xticklabels(month_labels, fontsize=10, fontweight="bold")
    ax3.set_ylabel("Taux de Capture  (Modele / Oracle)", fontsize=11)
    ax3.set_xlabel("Mois", fontsize=11)
    ax3.set_title("Distribution Mensuelle du Taux de Capture  —  Mediane en rouge",
                  fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(axis="y", ls=":", alpha=0.4)

    # Titre global
    periode = (f"{idx[0].strftime('%d/%m/%Y')} -> {idx[-1].strftime('%d/%m/%Y')}")
    fig.suptitle(
        f"Analyse Saisonniere  |  Periode : {periode}  |  "
        f"Capture annuelle moyenne : {s_ratio.mean()*100:.1f} %",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "="*65)
    print("Analyse Saisonniere v3 sauvegardee")
    print("="*65)
    print(f"Chemin : {save_path}")
    print(f"Capture annuelle moyenne : {s_ratio.mean()*100:.1f} %")
    print("\nMois  |  Profit Oracle (EUR)  |  Profit Modele (EUR)  |  Capture moy")
    for p, po, pm in zip(monthly_po.index, monthly_po.values, monthly_pm.values):
        cap = pm / po * 100 if po != 0 else float("nan")
        print(f"  {p}  |  {po:>10.0f}  |  {pm:>10.0f}  |  {cap:.1f} %")
    print("="*65 + "\n")
