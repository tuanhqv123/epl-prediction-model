"""
Final test for xg_form_diff window
"""
import sys
sys.path.insert(0, '.')

# Run the base setup
exec(open('experiments/test_rolling_xg.py').read().split('print("\\nTesting larger windows:")')[0])

print("="*60)
print("FINAL WINDOW OPTIMIZATION")
print("="*60)

for window in [8, 9, 10, 11, 12, 13]:
    feat_df = create_features_v2(df, defaults, window)
    
    train_mask = df['season'].isin(train_seasons).values
    test_mask = df['season'].isin(test_seasons).values
    
    X_train = feat_df[train_mask][COLS].fillna(0).replace([float('inf'), float('-inf')], 0).values
    y_train = feat_df[train_mask]['FTR'].values
    X_test = feat_df[test_mask][COLS].fillna(0).replace([float('inf'), float('-inf')], 0).values
    y_test = feat_df[test_mask]['FTR'].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, max_iter=2000)
    model.fit(X_train_s, y_train)
    
    proba = model.predict_proba(X_test_s)
    proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
    
    loss = log_loss(y_test, proba_ord, labels=CLASSES)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    
    print(f'Window={window}: Loss={loss:.4f}, Acc={acc*100:.1f}%')
