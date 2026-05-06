"""
critical_state_detector.py - Module à importer dans test_models.py

Ajoute une détection d'état critique basée sur des règles simples
pour compléter les prédictions ML.
"""

def check_critical_state(current_row):
    """
    Vérifie si l'état actuel du système justifie une alerte immédiate
    
    Args:
        current_row: Ligne pandas avec les métriques actuelles
    
    Returns:
        dict: Informations sur l'alerte critique
    """
    cpu = current_row.get('CPU_USAGE_PERCENT', 0)
    mem = current_row.get('MEM_USAGE_PERCENT', 0)
    status = current_row.get('LOCAL_STATUS', 'UNKNOWN')
    
    alert = {
        'has_alert': False,
        'level': 'OK',
        'message': '',
        'icon': '✓',
        'color': 'green'
    }
    
    # Règle 1: État CRITICAL dans les données
    if status in ['CRITICAL', 'pre_crash']:
        alert['has_alert'] = True
        alert['level'] = 'CRITICAL'
        alert['message'] = f"État système CRITICAL détecté"
        alert['icon'] = '🔴'
        alert['color'] = 'red'
        return alert
    
    # Règle 2: CPU critique (≥90%)
    if cpu >= 90:
        alert['has_alert'] = True
        alert['level'] = 'CRITICAL'
        alert['message'] = f"CPU critique ({cpu}% ≥ 90%)"
        alert['icon'] = '🔴'
        alert['color'] = 'red'
        return alert
    
    # Règle 3: Mémoire critique (≥95%)
    if mem >= 95:
        alert['has_alert'] = True
        alert['level'] = 'CRITICAL'
        alert['message'] = f"Mémoire critique ({mem}% ≥ 95%)"
        alert['icon'] = '🔴'
        alert['color'] = 'red'
        return alert
    
    # Règle 4: CPU + MEM tous deux élevés (≥85% et ≥90%)
    if cpu >= 85 and mem >= 90:
        alert['has_alert'] = True
        alert['level'] = 'CRITICAL'
        alert['message'] = f"CPU ({cpu}%) et MEM ({mem}%) critiques"
        alert['icon'] = '🔴'
        alert['color'] = 'red'
        return alert
    
    # Règle 5: CPU élevé (≥85%)
    if cpu >= 85:
        alert['has_alert'] = True
        alert['level'] = 'WARNING'
        alert['message'] = f"CPU élevé ({cpu}% ≥ 85%)"
        alert['icon'] = '🟡'
        alert['color'] = 'yellow'
        return alert
    
    # Règle 6: Mémoire élevée (≥90%)
    if mem >= 90:
        alert['has_alert'] = True
        alert['level'] = 'WARNING'
        alert['message'] = f"Mémoire élevée ({mem}% ≥ 90%)"
        alert['icon'] = '🟡'
        alert['color'] = 'yellow'
        return alert
    
    # Règle 7: État WARNING dans les données
    if status == 'WARNING':
        alert['has_alert'] = True
        alert['level'] = 'INFO'
        alert['message'] = f"État système WARNING"
        alert['icon'] = '🔵'
        alert['color'] = 'blue'
        return alert
    
    return alert


def format_output_with_critical_check(idx, ts, current, predictions):
    """
    Affiche les résultats avec vérification d'état critique
    
    Args:
        idx: Index du test
        ts: Timestamp
        current: État actuel (pandas Series)
        predictions: Résultats des prédictions ML
    """
    print("\n" + "="*75)
    print(f"  PRÉDICTION À T = {ts}")
    print("="*75)
    
    # Vérifier l'état critique AVANT d'afficher les prédictions ML
    critical_alert = check_critical_state(current)
    
    # État réel
    status = current.get('LOCAL_STATUS', 'UNKNOWN')
    reason = current.get('STATUS_REASON', 'unknown')
    cpu = current.get('CPU_USAGE_PERCENT', 0)
    mem = current.get('MEM_USAGE_PERCENT', 0)
    
    print(f"\n  État réel à cet instant : {status} ({reason})")
    print(f"  CPU={cpu}%  MEM={mem}%")
    
    # NOUVEAU: Afficher l'alerte d'état critique si détectée
    if critical_alert['has_alert']:
        print(f"\n  {critical_alert['icon']} ALERTE ÉTAT ACTUEL : {critical_alert['level']}")
        print(f"     → {critical_alert['message']}")
        if critical_alert['level'] == 'CRITICAL':
            print(f"     → ⚡ ACTION REQUISE IMMÉDIATEMENT")
    
    # Prédictions ML
    print("\n  PRÉDICTIONS ML/DL :")
    print("  " + "-"*71)
    print(f"  {'Horizon':<15} {'Probabilité':<15} {'Seuil':<15} {'Alerte':<15}")
    print("  " + "-"*71)
    
    for horizon, pred in predictions.items():
        prob = pred['probability']
        threshold = pred['threshold']
        alert = pred['alert']
        
        if alert:
            icon = "⚠️ ALERTE"
        elif prob > threshold * 0.7:
            icon = "◯ surveillé"
        else:
            icon = "✓ OK"
        
        print(f"  {horizon:<15} {prob:>8.4f}   {threshold:>8.4f}    {icon}")
    
    # NOUVEAU: Résumé des alertes
    ml_alerts = [h for h, p in predictions.items() if p['alert']]
    
    print("\n  " + "="*71)
    if critical_alert['has_alert'] and critical_alert['level'] == 'CRITICAL':
        print(f"  🚨 SYNTHÈSE : ALERTE CRITIQUE (état actuel)")
        print(f"  └─ Raison : {critical_alert['message']}")
    elif ml_alerts:
        print(f"  ⚠️  SYNTHÈSE : Incident prédit dans {ml_alerts[0]}")
    elif critical_alert['has_alert']:
        print(f"  {critical_alert['icon']} SYNTHÈSE : {critical_alert['level']} (état actuel)")
    else:
        print(f"  ✅ SYNTHÈSE : Système normal")
    print("  " + "="*71)


# ============================================================================
# Fonction helper pour calculer un score composite simple
# ============================================================================

def compute_simple_composite_score(current, predictions):
    """
    Calcule un score composite simple : 60% ML + 40% état actuel
    
    Returns:
        float: Score entre 0 et 1
    """
    # Score ML : moyenne pondérée des probabilités
    weights = {'15min': 1.5, '30min': 1.3, '60min': 1.0, '360min': 0.7, '3 jours': 0.5}
    ml_score = 0
    total_weight = 0
    
    for horizon, pred in predictions.items():
        if pred['alert']:
            ml_score += pred['probability'] * weights.get(horizon, 1.0)
            total_weight += weights.get(horizon, 1.0)
    
    if total_weight > 0:
        ml_score /= total_weight
    
    # Score état actuel
    cpu = current.get('CPU_USAGE_PERCENT', 0)
    mem = current.get('MEM_USAGE_PERCENT', 0)
    status = current.get('LOCAL_STATUS', 'NORMAL')
    
    state_score = 0
    if status == 'CRITICAL':
        state_score = 1.0
    elif cpu >= 90 or mem >= 95:
        state_score = 0.95
    elif cpu >= 85 or mem >= 90:
        state_score = 0.7
    elif status == 'WARNING':
        state_score = 0.5
    elif cpu >= 70 or mem >= 80:
        state_score = 0.3
    
    # Composite : 60% ML + 40% état
    composite = ml_score * 0.6 + state_score * 0.4
    
    return composite


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    # Exemple : Cas 8 (CRITICAL non détecté)
    example_critical = pd.Series({
        'timestamp': '2026-05-02 07:45:00',
        'LOCAL_STATUS': 'CRITICAL',
        'STATUS_REASON': 'pre_crash',
        'CPU_USAGE_PERCENT': 88,
        'MEM_USAGE_PERCENT': 95
    })
    
    example_predictions = {
        '15 min': {'probability': 0.1794, 'threshold': 0.6248, 'alert': False},
        '30 min': {'probability': 0.1899, 'threshold': 0.7833, 'alert': False},
        '1 heure': {'probability': 0.0266, 'threshold': 0.5598, 'alert': False},
        '6 heures': {'probability': 0.1936, 'threshold': 0.7749, 'alert': False},
        '3 jours': {'probability': 0.0004, 'threshold': 0.5135, 'alert': False}
    }
    
    # Test de détection
    critical = check_critical_state(example_critical)
    print("Test détection état critique:")
    print(f"  Alerte: {critical['has_alert']}")
    print(f"  Niveau: {critical['level']}")
    print(f"  Message: {critical['message']}")
    
    # Test affichage complet
    print("\n" + "="*75)
    format_output_with_critical_check(
        idx=8,
        ts=example_critical['timestamp'],
        current=example_critical,
        predictions=example_predictions
    )
    
    # Test score composite
    composite = compute_simple_composite_score(example_critical, example_predictions)
    print(f"\nScore composite: {composite:.1%}")
